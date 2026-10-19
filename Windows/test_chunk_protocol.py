import io
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
import uuid
import numpy as np
from PIL import Image
import zstandard as zstd
from chunk_protocol import decode_lossless_packet,BlockStore
from test_offline_pipeline import raw_packet
from offline_pipeline import validate_session,records
from ai_alignment import robust_pose


def encode(packet,repeat=False):
    size=struct.unpack_from('<I',packet,84)[0]; offset=88+size
    w,h,rgb_size=struct.unpack_from('<HHI',packet,offset)
    if repeat: depth=b'';png=b''
    else:
        shuffled=np.frombuffer(packet[88:offset],np.uint8).reshape(-1,4).T.copy().tobytes()
        depth=zstd.ZstdCompressor(level=1).compress(shuffled)
        bgra=np.frombuffer(packet,np.uint8,count=rgb_size,offset=offset+8).reshape(h,w,4)
        image=Image.fromarray(bgra[:,:,[2,1,0,3]],'RGBA'); output=io.BytesIO();image.save(output,format='PNG');png=output.getvalue()
    header=bytearray(packet[:88]);struct.pack_into('<I',header,0,0xB1C01704 if repeat else 0xB1C01703);struct.pack_into('<I',header,84,len(depth))
    return bytes(header)+depth+struct.pack('<HHI',w,h,len(png))+png+packet[offset+8+rgb_size:]


def message(seq,payload): return struct.pack('<I',seq)+hashlib.sha256(payload).digest()+payload


class BlockTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.folder=Path(self.temp.name);self.store=BlockStore(self.folder);self.session=str(uuid.uuid4())
    def tearDown(self):self.temp.cleanup()

    def test_png_and_float32_are_bit_exact(self):
        raw=bytearray(raw_packet())
        for index,bits in enumerate([0x80000000,0x7fc01234,0x00000001,0x3e99999a]): struct.pack_into('<I',raw,88+index*4,bits)
        self.assertEqual(decode_lossless_packet(encode(raw)),raw)

    def test_exact_repeat_preserves_new_header(self):
        a=raw_packet(0);b=raw_packet(1)
        previous=decode_lossless_packet(encode(a))
        self.assertEqual(decode_lossless_packet(encode(b,True),previous),b)
        self.assertLess(len(encode(b,True)),len(b)/10)

    def test_ack_and_retry_are_durable(self):
        packet=encode(raw_packet(0));ack=self.store.accept(self.session,message(0,packet))
        self.assertEqual(ack['sha256'],hashlib.sha256(packet).hexdigest())
        self.assertEqual(self.store.accept(self.session,message(0,packet)),ack)
        self.assertEqual((self.store.folder(self.session)/'000000000.b3dc').read_bytes(),packet)

    def test_corrupt_block_never_acknowledged(self):
        data=bytearray(message(0,encode(raw_packet())));data[-1]^=1
        with self.assertRaises(ValueError):self.store.accept(self.session,data)
        self.assertFalse(self.store.folder(self.session).exists())

    def test_commit_validates_all_sequences(self):
        self.store.accept(self.session,message(0,encode(raw_packet(0))))
        with self.assertRaises(FileNotFoundError):self.store.finish(self.session,2,0)
        self.store.accept(self.session,message(1,encode(raw_packet(1),True)))
        path,receipt=self.store.finish(self.session,2,0)
        self.assertEqual(receipt['op'],'committed')
        self.assertEqual(validate_session(path)['frames'],2)
        self.assertEqual(len(list(records(path))),2)
        self.assertEqual(self.store.finish(self.session,2,0)[0],path)

    def test_robust_neural_pose_rejects_outliers(self):
        rng=np.random.default_rng(2); source=rng.normal(size=(30,3))*.02
        target=source+[.01,-.02,.005];target[-8:]=rng.normal(size=(8,3))
        pose=robust_pose(source,target)
        np.testing.assert_allclose(pose[:3,3],[.01,-.02,.005],atol=1e-6)


if __name__=='__main__':unittest.main()
