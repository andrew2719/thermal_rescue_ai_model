"""Faithful standalone implementation of the existing 419,841-parameter Bouazizi CNN.

The checkpoint comes from the earlier training project; this package does not train.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn


class BouaziziBinaryCNN(nn.Module):
    def __init__(self, dropout: float = 0.6):
        super().__init__()
        # This matches the published training report's layer order, kernel sizes and dimensions.
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(dropout),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(dropout),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(dropout),
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(dropout),
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.classifier(self.features(x)).flatten()


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as file:
        for b in iter(lambda: file.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def _read_original_preprocessing(root: Path, cfg: dict) -> tuple[float, float]:
    """Prefer original precise preprocessing file if user copies it; support common key spellings."""
    file = root / 'models' / 'preprocessing.json'
    defaults = cfg['normalization']
    if not file.exists():
        return float(defaults['mean_c']), float(defaults['std_c'])
    values = json.loads(file.read_text())
    # The original training Preprocessing class stores the raw Celsius scalar statistics.
    def pick(keys):
        for key in keys:
            if key in values:
                return float(values[key])
        return None
    mean = pick(('mean_c', 'mean', 'global_mean_c', 'train_mean_c'))
    std = pick(('std_c', 'std', 'global_std_c', 'train_std_c'))
    if mean is None or std is None:
        raise ValueError('models/preprocessing.json exists but its Celsius mean/std keys are not recognized; see README.')
    if abs(mean - defaults['mean_c']) > 0.001 or abs(std - defaults['std_c']) > 0.001:
        raise ValueError('The supplied preprocessing.json does not match the trained model statistics')
    return mean, std


def _load_state_dict(checkpoint) -> dict:
    if not isinstance(checkpoint, dict):
        raise ValueError('Not a compatible state-dict checkpoint; expected a saved Bouazizi CNN, not YOLO')
    for key in ('model_state_dict', 'state_dict'):
        if key in checkpoint and isinstance(checkpoint[key], dict):
            return checkpoint[key]
    if checkpoint and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
        return checkpoint
    raise ValueError('Checkpoint does not contain model_state_dict. Did you accidentally copy YOLO best.pt?')


def _strict_weight_transfer(model: nn.Module, state: dict) -> None:
    """Support layer-name changes ONLY when tensor shapes and registration order match exactly.

    The original training code is external; refusing mismatched layers is safer than silent partial loading.
    """
    desired = model.state_dict()
    cleaned = {k.removeprefix('module.'): v for k, v in state.items()}
    if list(cleaned) == list(desired):
        model.load_state_dict(cleaned, strict=True)
        return
    if len(cleaned) != len(desired):
        raise ValueError(f'Checkpoint has {len(cleaned)} tensors; expected {len(desired)} for Bouazizi CNN')
    mismatches = [(src, dst, tuple(value.shape), tuple(desired[dst].shape))
                  for (src, value), dst in zip(cleaned.items(), desired)
                  if tuple(value.shape) != tuple(desired[dst].shape)]
    if mismatches:
        raise ValueError(f'Checkpoint architecture is incompatible. First mismatch: {mismatches[0]}')
    model.load_state_dict({new: tensor for (_, tensor), new in zip(cleaned.items(), desired)}, strict=True)


class CNNPredictor:
    def __init__(self, root: Path, config: dict):
        path = root / config['model_file']
        if not path.exists():
            raise FileNotFoundError(
                f'Missing trained CNN checkpoint: {path}\n'
                'Copy models/bouazizi_binary_best.pt from your ORIGINAL CNN project. '
                'The YOLO best.pt is NOT compatible.')
        expected_sha = config.get('model_sha256')
        if expected_sha:
            got = _hash_file(path)
            if got != expected_sha:
                raise ValueError('The CNN checkpoint hash does not match the original trained model. '
                                 f'Expected {expected_sha}, got {got}. Verify you copied the right .pt file.')
        self.mean_c, self.std_c = _read_original_preprocessing(root, config)
        if self.std_c <= 0:
            raise ValueError('Invalid normalization std')
        self.model = BouaziziBinaryCNN()
        if sum(p.numel() for p in self.model.parameters()) != 419841:
            raise AssertionError('Model architecture changed unexpectedly')
        # Always load only trusted weights, on CPU, in eval mode (dropout disabled).
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        _strict_weight_transfer(self.model, _load_state_dict(checkpoint))
        self.model.eval()

    @torch.inference_mode()
    def predict_probability(self, frame: np.ndarray) -> float:
        norm = (frame.astype(np.float32) - self.mean_c) / self.std_c
        x = torch.from_numpy(norm[None, None, :, :])
        logit = self.model(x)
        p = torch.sigmoid(logit).item()
        if not np.isfinite(p):
            raise RuntimeError('CNN returned non-finite output')
        return float(p)
