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
