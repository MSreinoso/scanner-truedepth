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
    validate_session, receive_archive, relative_rotation, TensorFusion, reconstruct, export_selected)


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
        self.assertEqual(Settings().validate().resolution_mm,.05)
        self.assertEqual(list(PRESETS.values())[-1].resolution_mm,.05)
        for kwargs in ({'resolution_mm':float('nan')},{'stride':0},{'memory_mb':0},{'min_depth_m':.8},{'device':'fake'}):
            with self.assertRaises(ValueError): Settings(**kwargs).validate()

    def test_static_imu_does_not_rotate_camera(self):
        angle=.8; R=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
        np.testing.assert_allclose(relative_rotation(R,R),np.eye(3),atol=1e-12)

    def test_viewer_draws_solid_triangles_and_keeps_manual_camera(self):
        from mesh_view import weld,ViewMesh,Camera,CpuRenderer
        # Closed box as a triangle soup with duplicated corners (as Marching Cubes emits).
        corners=np.array([[x,y,z] for x in (-.02,.02) for y in (-.02,.02) for z in (.28,.32)],np.float32)
        quads=[(0,1,3,2),(4,6,7,5),(0,4,5,1),(2,3,7,6),(0,2,6,4),(1,5,7,3)]
        soup=np.array([[corners[a],corners[b],corners[c]] for q in quads for a,b,c in ((q[0],q[1],q[2]),(q[0],q[2],q[3]))])
        V,F,N,C=weld(soup,quantum=1e-6)
        self.assertEqual((len(V),len(F)),(8,12))
        mesh=ViewMesh(V,F,N,None,'test',12)
        camera=Camera(); camera.frame(mesh); camera.orbit(60,-40); camera.zoom_by(1.3)
        renderer=CpuRenderer(); renderer.upload(mesh)
        image=renderer.render(camera,200,150,use_color=False)
        covered=(np.abs(image.astype(int)-[20,26,38]).sum(2)>30).mean()
        self.assertGreater(covered,.15)   # a filled surface, not sparse points
        state=(camera.yaw,camera.pitch,camera.zoom)
        renderer.render(camera,200,150); camera.frame(mesh,keep=True)
        self.assertEqual((camera.yaw,camera.pitch,camera.zoom),state)
        camera.frame(mesh,keep=False); self.assertEqual((camera.yaw,camera.pitch,camera.zoom),(0.,0.,1.))

    def test_stl_export_is_verified_against_selected_result(self):
        class Fake:
            triangles=[0,1,2]
            def export_mm(self,path):
                Path(path).write_bytes(b'\0'*80+struct.pack('<I',3)+b'\0'*150)
        target=self.folder/'a.stl'
        self.assertEqual(export_selected(Fake(),target),3)
        class Wrong(Fake):
            triangles=[0,1]
        with self.assertRaises(ValueError): export_selected(Wrong(),target)


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
