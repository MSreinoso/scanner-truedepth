import io
import json
import hashlib
import struct
import tempfile
from pathlib import Path
import unittest
from unittest.mock import Mock
import zlib
import numpy as np
from offline_pipeline import (HEADER, RAW_MAGIC, Settings, decode_frame, records,
    validate_session, receive_archive, relative_rotation, TensorFusion, reconstruct, StableViewer)


def raw_packet(index=0,width=24,height=24):
    y,x=np.mgrid[:height,:width]
    depth=(.3+.003*np.sin(x/3)*np.cos(y/3)).astype('<f4').tobytes()
    color=np.zeros((height,width,4),np.uint8)
    color[:,:,0]=40; color[:,:,1]=(x*7)%255; color[:,:,2]=(y*7)%255; color[:,:,3]=255
    meta=json.dumps({'referenceSize':[width,height],'distortionCenter':[width/2,height/2],'lensTable':[0,0]}).encode()
    header=struct.pack('<IId9f3f4fHHI',RAW_MAGIC,index,index/15+1,
        1,0,0,0,1,0,0,0,1,0,0,0,120,120,width/2,height/2,width,height,len(depth))
    return header+depth+struct.pack('<HHI',width,height,color.nbytes)+color.tobytes()+struct.pack('<I',len(meta))+meta


def session(count=3,dropped=0):
    data=bytearray(HEADER)
    for index in range(count):
        packet=raw_packet(index)
        data.extend(struct.pack('<I',len(packet))); data.extend(packet)
    meta=json.dumps({'frames':count,'dropped':dropped,'version':2}).encode()
    return bytes(data)+struct.pack('<II',0,len(meta))+meta


def compressed(data):
    encoder=zlib.compressobj(wbits=-15)
    return encoder.compress(data)+encoder.flush()


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.folder=Path(self.temp.name)

    def tearDown(self): self.temp.cleanup()

    def test_native_float32_depth_and_bgra_color(self):
        frame=decode_frame(raw_packet())
        self.assertEqual(frame.depth.dtype,np.float32)
        np.testing.assert_array_equal(frame.rgb[0,0],[0,0,40])
        self.assertAlmostEqual(frame.depth[0,0],.3,places=6)

    def test_complete_session_and_dropped_count(self):
        path=self.folder/'scan.b3draw'; path.write_bytes(session(dropped=2))
        self.assertEqual(validate_session(path),{'frames':3,'dropped':2,'version':2})
        self.assertEqual(len(list(records(path))),3)

    def test_truncation_is_not_accepted(self):
        for cut in (1,8,100,len(session())-1):
            path=self.folder/'bad'; path.write_bytes(session()[:cut])
            with self.assertRaises(ValueError): validate_session(path)

    def test_checksum_failure_leaves_no_session(self):
        body=compressed(session()); path=self.folder/'scan'
        with self.assertRaises(ValueError): receive_archive(io.BytesIO(body),len(body),'0'*64,path)
        self.assertFalse(path.exists()); self.assertEqual(list(self.folder.iterdir()),[])

    def test_upload_retry_is_idempotent(self):
        body=compressed(session()); digest=hashlib.sha256(body).hexdigest(); path=self.folder/'scan'
        for _ in range(2):
            receipt=receive_archive(io.BytesIO(body),len(body),digest,path)
            self.assertEqual(receipt['frames'],3)
        self.assertEqual(path.read_bytes(),session())

    def test_different_session_cannot_overwrite_uuid(self):
        path=self.folder/'scan'; path.write_bytes(session())
        body=compressed(session(dropped=1))
        with self.assertRaises(ValueError): receive_archive(io.BytesIO(body),len(body),hashlib.sha256(body).hexdigest(),path)
        self.assertEqual(path.read_bytes(),session())

    def test_expansion_limit_and_corrupted_trailer(self):
        body=compressed(session()); path=self.folder/'scan'
        with self.assertRaises(ValueError): receive_archive(io.BytesIO(body),len(body),hashlib.sha256(body).hexdigest(),path,max_size=3000)
        self.assertFalse(path.exists())

    def test_sequence_and_calibration_are_validated(self):
        packet=bytearray(raw_packet()); struct.pack_into('<f',packet,64,0)
        with self.assertRaises(ValueError): decode_frame(packet)
        data=bytearray(session()); struct.pack_into('<I',data,12+4,8)
        path=self.folder/'bad'; path.write_bytes(data)
        with self.assertRaises(ValueError): validate_session(path)

    def test_settings_default_and_reject_nonfinite(self):
        from offline_pipeline import PRESETS
        self.assertEqual(Settings().validate().resolution_mm,.5)
        self.assertEqual(list(PRESETS.values())[-1].resolution_mm,.05)
        for kwargs in ({'resolution_mm':float('nan')},{'stride':0},{'memory_mb':0},{'min_depth_m':.8},{'device':'fake'}):
            with self.assertRaises(ValueError): Settings(**kwargs).validate()

    def test_static_imu_does_not_rotate_camera(self):
        angle=.8; R=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
        np.testing.assert_allclose(relative_rotation(R,R),np.eye(3),atol=1e-12)

    def test_viewer_render_preserves_manual_view(self):
        viewer=object.__new__(StableViewer)
        viewer.points=np.array([[0.,0.,0.],[1.,1.,1.]])
        viewer.colors=np.ones((2,3)); viewer.radius=2.; viewer.yaw=.4; viewer.pitch=.2; viewer.zoom=1.3
        for name in ('delete','create_oval','create_text'): setattr(viewer,name,Mock())
        viewer.winfo_width=lambda:600; viewer.winfo_height=lambda:400
        viewer.render(); viewer.render()
        self.assertEqual((viewer.yaw,viewer.pitch,viewer.zoom),(.4,.2,1.3))


try:
    import open3d
    HAS_O3D=True
except ImportError:
    HAS_O3D=False


@unittest.skipUnless(HAS_O3D,'Open3D not installed in local test runtime')
class ReconstructionTests(unittest.TestCase):
    def test_tensor_fusion_and_colored_alignment_cpu(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'session.b3draw'; path.write_bytes(session(4))
            settings=Settings(resolution_mm=1.,memory_mb=128,min_observations=1.,device='CPU')
            mesh,report=reconstruct(path,settings)
            self.assertGreater(len(mesh.triangles),0)
            self.assertEqual(report['accepted'],4)
            self.assertEqual(report['device'],'CPU:0')

    def test_memory_limit_does_not_silently_change_resolution(self):
        from offline_pipeline import prepare_frame
        settings=Settings(resolution_mm=.01,memory_mb=128,device='CPU')
        fusion=TensorFusion(settings); fusion.max_blocks=0
        frame=decode_frame(raw_packet()); depth,rgb=prepare_frame(frame,settings)
        with self.assertRaises(MemoryError): fusion.integrate(depth,rgb,frame.intrinsics,np.eye(4))
        self.assertEqual(fusion.settings.resolution_mm,.01)


if __name__=='__main__': unittest.main()
