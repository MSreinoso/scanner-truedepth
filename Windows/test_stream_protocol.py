import io
import queue
import struct
import unittest
import zlib

import numpy as np
from PIL import Image
from Bichito3D_ReconstructionEngine import FramePacket, TCPFrameReceiver, MAGIC


def packet(raw_deflate=True):
    depth = np.array([[0.2, 0.3], [0.4, 0.5]], dtype='<f2').tobytes()
    encoder = zlib.compressobj(wbits=-15 if raw_deflate else 15)
    compressed = encoder.compress(depth) + encoder.flush()
    jpeg = io.BytesIO()
    Image.new('RGB', (2, 2), 'red').save(jpeg, format='JPEG')
    rgb = jpeg.getvalue()
    header = struct.pack('<IId9f3f4fHHI', MAGIC, 7, 1.25,
                         1, 0, 0, 0, 1, 0, 0, 0, 1,
                         0, 0, 0, 100, 100, 1, 1, 2, 2, len(compressed))
    return header + compressed + struct.pack('<HHI', 2, 2, len(rgb)) + rgb


class ChunkSocket:
    def __init__(self, chunks):
        self.chunks = iter(chunks)
        self.closed = False

    def settimeout(self, timeout):
        pass

    def recv(self, size):
        return next(self.chunks, b'')

    def close(self):
        self.closed = True


class StreamProtocolTests(unittest.TestCase):
    def test_apple_raw_deflate(self):
        frame = FramePacket.deserialize(packet())
        self.assertEqual(frame.frame_id, 7)
        np.testing.assert_allclose(frame.depth_map, [[0.2, 0.3], [0.4, 0.5]], atol=0.001)
        self.assertEqual(frame.rgb_image.shape, (2, 2, 3))

    def test_wrapped_zlib_compatibility(self):
        self.assertEqual(FramePacket.deserialize(packet(False)).depth_map.shape, (2, 2))

    def test_invalid_depth_dimensions(self):
        malformed = bytearray(packet())
        struct.pack_into('<HH', malformed, 80, 65535, 65535)
        with self.assertRaises(ValueError):
            FramePacket.deserialize(malformed)

    def test_tcp_fragmentation_and_multiple_frames(self):
        frames = queue.Queue(maxsize=5)
        receiver = TCPFrameReceiver(frames)
        receiver.running = True
        data = packet() + packet(False)
        conn = ChunkSocket(data[offset:offset + 13] for offset in range(0, len(data), 13))
        receiver._receive_loop(conn)
        self.assertEqual(receiver.frames_received, 2)
        self.assertEqual(frames.qsize(), 2)
        self.assertTrue(conn.closed)

    def test_tcp_rejects_unbounded_payload(self):
        receiver = TCPFrameReceiver(queue.Queue())
        receiver.running = True
        header = bytearray(packet()[:88])
        struct.pack_into('<I', header, 84, 0xFFFFFFFF)
        conn = ChunkSocket([header])
        receiver._receive_loop(conn)
        self.assertEqual(receiver.frames_received, 0)
        self.assertTrue(conn.closed)


if __name__ == '__main__':
    unittest.main()
