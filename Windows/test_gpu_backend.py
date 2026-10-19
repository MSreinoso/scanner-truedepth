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
