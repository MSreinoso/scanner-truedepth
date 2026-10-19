import json,os,tempfile,unittest
from unittest.mock import patch
from pathlib import Path
from unittest.mock import Mock
import model_registry as registry

class ModelTests(unittest.TestCase):
    def test_comparison_selection_changes_export_mesh(self):
        from offline_pipeline import OfflineStudio
        studio=OfflineStudio.__new__(OfflineStudio);studio.busy=False
        first,second=object(),object()
        studio.results={'RTX sola':(first,{'triangles':10}),'RTX + IA':(second,{'triangles':20})}
        studio.viewer=Mock();studio.status=Mock();studio.result_choice=Mock()
        studio.result_choice.get.return_value='RTX sola';studio.select_result()
        self.assertIs(studio.mesh,first)
        studio.result_choice.get.return_value='RTX + IA';studio.select_result()
        self.assertIs(studio.mesh,second)
        studio.viewer.set_mesh.assert_called_with(second)

    def test_default_persists_and_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as folder,patch.dict(os.environ,{'LOCALAPPDATA':folder}):
            self.assertEqual(registry.default(),registry.BUILTIN)
            registry.set_default(registry.BUILTIN)
            self.assertEqual(registry.default(),registry.BUILTIN)
            bad=Path(folder)/'bad.pt';bad.write_bytes(b'invalid')
            bad.with_suffix('.json').write_text(json.dumps({'architecture':'residual-unet-v1','sha256':'bad'}))
            with self.assertRaisesRegex(ValueError,'Checksum'):registry.set_default(bad)
            self.assertEqual(registry.default(),registry.BUILTIN)

    def test_incompatible_architecture_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            bad=Path(folder)/'bad.pt';bad.write_bytes(b'invalid')
            bad.with_suffix('.json').write_text(json.dumps({'architecture':'unknown'}))
            with self.assertRaisesRegex(ValueError,'incompatible'):registry.validate(bad)
