import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import importlib.util
import numpy as np

class TrainedArtifactTests(unittest.TestCase):
    def test_weight_integrity_and_independent_evaluation(self):
        folder=Path(__file__).resolve().parent/'models'
        card=json.loads((folder/'depth_denoiser_v1.json').read_text(encoding='utf-8'))
        self.assertEqual(hashlib.sha256((folder/'depth_denoiser_v1.pt').read_bytes()).hexdigest(),card['sha256'])
        groups=[set(card['splits'][name]) for name in ('train','validation','test')]
        self.assertTrue(all(not groups[a]&groups[b] for a,b in ((0,1),(0,2),(1,2))))
        metrics=card['held_out_synthetic_noise_metrics']
        self.assertLess(metrics['model_rmse_mm'],metrics['noisy_rmse_mm'])
        self.assertFalse(card['truedepth_validated'])

@unittest.skipUnless(importlib.util.find_spec('torch'),'Neural runtime optional; tested locally on RTX')
class NeuralDepthTests(unittest.TestCase):
    def test_reconstruction_preparation_uses_network_without_mutating_capture(self):
        from offline_pipeline import prepare_frame,decode_frame,Settings
        from test_offline_pipeline import raw_packet
        frame=decode_frame(raw_packet()); original=frame.depth.copy()
        depth,rgb=prepare_frame(frame,Settings(neural_denoising=True))
        np.testing.assert_array_equal(frame.depth,original)
        self.assertEqual(depth.dtype,np.float32);self.assertEqual(rgb.shape,(*depth.shape,3))
        self.assertTrue(np.isfinite(depth).all());self.assertTrue(np.any(depth!=original))

    def test_trained_inference_preserves_holes_and_source(self):
        from neural_depth import DepthDenoiser
        denoiser=DepthDenoiser()
        rng=np.random.default_rng(23)
        depth=(.3+rng.normal(0,.002,(97,99))).astype(np.float32)
        depth[:4]=0;depth[4,0]=np.nan; original=depth.copy()
        rgb=np.full((97,99,3),128,np.uint8)
        output=denoiser.denoise(depth,rgb)
        np.testing.assert_equal(depth,original)
        self.assertEqual(output.shape,depth.shape);self.assertTrue(np.isfinite(output).all())
        self.assertTrue((output[:4]==0).all());self.assertEqual(output[4,0],0)
        self.assertLess(np.std(output[20:-20,20:-20]),np.std(depth[20:-20,20:-20]))
        self.assertLessEqual(np.max(np.abs(output[5:]-depth[5:])),.010001)
        self.assertLess(abs(np.mean(output[20:-20,20:-20])-.3),.001)

    def test_tampered_weights_rejected(self):
        from neural_depth import DepthDenoiser
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'model.pt';path.write_bytes(b'altered')
            path.with_suffix('.json').write_text(json.dumps({'sha256':hashlib.sha256(b'original').hexdigest()}))
            with self.assertRaisesRegex(ValueError,'alterados'):DepthDenoiser(path)

    def test_scene_disjoint_splits(self):
        from train_depth import split_scenes
        scenes=np.array([f'scene{i//3}' for i in range(120)])
        groups=split_scenes({'scene':scenes})
        names=[set(scenes[group]) for group in groups]
        self.assertTrue(all(not names[a]&names[b] for a,b in ((0,1),(0,2),(1,2))))
        self.assertEqual(sum(len(group) for group in groups),len(scenes))

if __name__=='__main__':unittest.main()
