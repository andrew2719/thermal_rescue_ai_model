"""One stable public interface, independently runnable without the old project directory."""
from __future__ import annotations
import json
from pathlib import Path
import time
import numpy as np
from .io import validate_frame
from .localization import locate_one_hot_region

ROOT = Path(__file__).resolve().parent.parent


class SphereScoutDetector:
    def __init__(self, config_file: str | Path | None = None, predictor=None):
        file = Path(config_file) if config_file else ROOT / 'config.json'
        self.config = json.loads(file.read_text())
        self.threshold = float(self.config['decision_threshold'])
        self.model_version = self.config['model_version']
        if predictor is None:
            from .cnn import CNNPredictor
            predictor = CNNPredictor(file.resolve().parent, self.config)
        self.predictor = predictor

    def predict(self, frame, frame_id=None):
        started = time.perf_counter()
        spec = self.config['input']
        arr = validate_frame(frame, spec['height'], spec['width'], spec['min_c'], spec['max_c'])
        p = float(self.predictor.predict_probability(arr))
        if not np.isfinite(p) or not 0 <= p <= 1:
            raise RuntimeError(f'Invalid CNN human score {p}')
        detected = p >= self.threshold
        # Independent thermal candidate helps diagnose misses on out-of-distribution simulation frames.
        # It is NOT a person detection and is never used to override the CNN decision.
        candidate = locate_one_hot_region(arr, self.config.get('localizer'))
        region = candidate if detected else None
        grid = [[0 for _ in range(4)] for _ in range(3)]
        if region is not None:
            row, col = region['grid_cell_row_col']
            grid[row][col] = 1
        candidate_grid = [[0 for _ in range(4)] for _ in range(3)]
        if candidate is not None:
            cr, cc = candidate['grid_cell_row_col']
            candidate_grid[cr][cc] = 1
        warnings = []
        frame_median = float(np.median(arr))
        if not 18.04 <= frame_median <= 24.83:
            warnings.append('Frame median differs from CNN training environment; prediction may not generalize')
        if float(arr.max()) > 41.57:
            warnings.append('Hotter than any training pixel; prediction may not generalize')
        if detected and region is None:
            warnings.append('CNN predicts HUMAN but no unambiguous warm region was found')
        response = {
            'human_detected': bool(detected),
            'human_probability': p,  # classifier sigmoid, NOT externally calibrated confidence
            'human_score': p,
            'model_version': self.model_version,
            'threshold': self.threshold,
            'detections': [region] if region else [],
            'grid_3x4': grid,
            'thermal_candidate': candidate,  # raw heat region, NOT verified human position
            'candidate_grid_3x4': candidate_grid,
            'out_of_training_temperature_range': not 18.04 <= frame_median <= 24.83,
            'localization_method': 'heuristic_thermal_blob',
            'localization_available': region is not None,
            'inference_ms': round((time.perf_counter() - started) * 1000, 3),
            'warnings': warnings,
        }
        if frame_id is not None:
            if not isinstance(frame_id, (str, int, float)) or isinstance(frame_id, bool):
                raise ValueError('frame_id must be a string or number')
            response['frame_id'] = frame_id
        return response
