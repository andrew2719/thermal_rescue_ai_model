"""Verify loader correctness against a synthetic checkpoint, not a validation-accuracy test."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import torch
import numpy as np
from sphere_scout.cnn import CNNPredictor, BouaziziBinaryCNN


class TestLoader(unittest.TestCase):
    def test_strict_checkpoint_weights_and_probability(self):
        root = Path(__file__).resolve().parent.parent
        config = json.loads((root / 'config.json').read_text())
        with TemporaryDirectory() as td:
            root2 = Path(td)
            (root2 / 'models').mkdir()
            m = BouaziziBinaryCNN()
            torch.save({'model_state_dict': m.state_dict()}, root2 / config['model_file'])
            config['model_sha256'] = hashlib.sha256((root2 / config['model_file']).read_bytes()).hexdigest()
            cp = CNNPredictor(root2, config)
            p = cp.predict_probability(np.full((24,32),22.5,dtype=np.float32))
            self.assertTrue(0 <= p <= 1)
            cp.model.eval()
            self.assertTrue(all(not mod.training for mod in cp.model.modules()))

    def test_fails_on_wrong_weights(self):
        root = Path(__file__).resolve().parent.parent
        config = json.loads((root / 'config.json').read_text())
        with TemporaryDirectory() as td:
            root2 = Path(td)
            (root2 / 'models').mkdir()
            (root2 / config['model_file']).write_text('not a checkpoint')
            with self.assertRaisesRegex(ValueError, 'hash'):
                CNNPredictor(root2, config)
