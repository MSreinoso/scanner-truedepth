import os
from pathlib import Path
import tempfile
import unittest
import numpy as np
from gpu_backend import CudaTile,cuda_module,TiledModel

try:
    import cupy
    GPU=True
except ImportError:
    GPU=False


@unittest.skipUnless(GPU,'CUDA runtime optional; tested separately on RTX 5060')
class CudaTests(unittest.TestCase):
    def test_cuda_registration_recovers_plane_translation_and_rejects_empty(self):
        from cuda_registration import register,surface_tile_keys
        from offline_pipeline import Settings
        cp=cuda_module();settings=Settings()
        target=np.full((80,80),.3,np.float32)
        source=target+.001
        valid,pose,fitness,rmse=register(source,target,[200,200,40,40],[200,200,40,40],np.eye(4),settings)
        self.assertTrue(valid);self.assertGreater(fitness,.8)
        self.assertAlmostEqual(pose[2,3],-.001,places=5)
        self.assertLess(rmse,.001)
        keys=surface_tile_keys(source,[200,200,40,40],pose,.0032)
        self.assertIsInstance(keys,cp.ndarray);self.assertGreater(len(keys),0)
        with self.assertRaises(ValueError):register(np.zeros_like(source),target,[200,200,40,40],[200,200,40,40],np.eye(4),settings)

    def test_batched_integration_equals_sequential(self):
        cp=cuda_module(); rng=np.random.default_rng(5)
        frames=[]
        for k in range(4):
            depth=(.3+rng.normal(0,.0005,(40,48))).astype(np.float32); depth[rng.random((40,48))<.1]=0
            rgb=rng.integers(0,255,(40,48,3),dtype=np.uint8)
            pose=np.eye(4); pose[:3,3]=[.001*k,-.0005*k,.0007*k]
            frames.append((depth,rgb,pose))
        a=CudaTile(16,.001,[-.008,-.008,.292]); b=CudaTile(16,.001,[-.008,-.008,.292])
        K=[120,120,24,20]
        for depth,rgb,pose in frames: a.integrate(depth,rgb,K,pose)
        b.integrate_many([(cp.asarray(d),cp.asarray(r),np.linalg.inv(p)[:3,:].astype(np.float32),K,48,40) for d,r,p in frames])
        for name in ('field','weights','colors'):
            np.testing.assert_allclose(getattr(a,name).get(),getattr(b,name).get(),rtol=0,atol=2e-6)

    def test_disk_estimate_stops_and_removes_partial_result(self):
        import gpu_backend
        from unittest import mock
        from offline_pipeline import Settings,reconstruct
        from test_offline_pipeline import session
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'s.b3draw'; path.write_bytes(session(4))
            calls=[]
            def free(_):
                calls.append(1); return 10**13 if len(calls)==1 else 3*1024**3
            with mock.patch.object(gpu_backend,'free_bytes',free),mock.patch.object(gpu_backend,'estimate_mesh_bytes',lambda *a:10**11):
                with self.assertRaises(gpu_backend.DiskSpaceError) as caught:
                    reconstruct(path,Settings(resolution_mm=1.,device='CUDA',min_observations=1.),lambda *a:None,None)
            self.assertIn('No se reduce la resolución',str(caught.exception))
            self.assertEqual([p.name for p in Path(folder).iterdir()],['s.b3draw'])

    def test_rgb_guided_filter_matches_cpu(self):
        from offline_pipeline import rgb_guided_filter
        rng=np.random.default_rng(3)
        depth=(.3+rng.normal(0,.001,(48,64))).astype(np.float32); depth[5:9,5:9]=0
        rgb=rng.integers(0,255,(48,64,3),dtype=np.uint8)
        np.testing.assert_allclose(rgb_guided_filter(depth,rgb,True),rgb_guided_filter(depth,rgb,False),atol=2e-6)

    def test_cuda_initialization_is_idempotent(self):
        first=cuda_module();path=os.environ.get('PATH','')
        for _ in range(500):self.assertIs(cuda_module(),first)
        self.assertEqual(path,os.environ.get('PATH',''))

    def test_tsdf_and_marching_cubes_on_known_plane(self):
        depth=np.full((32,32),.3,np.float32);rgb=np.full((32,32,3),128,np.uint8)
        tile=CudaTile(16,.001,[-.008,-.008,.292])
        for _ in range(3):tile.integrate(depth,rgb,[120,120,16,16],np.eye(4))
        vertices,colors=tile.mesh(2)
        self.assertGreater(len(vertices),0)
        np.testing.assert_allclose(vertices[:,:,2],.3,atol=1e-6)
        self.assertTrue(np.isfinite(colors).all())


if __name__=='__main__': unittest.main()
