"""Validate Celsius grids and decode both canonical and real simulation JSON packets.

Canonical client: {"frame": [[... 32 numbers] ... 24 rows]}
Simulation: {"t": 5.375, "frame": 0, "width": 32, "height": 24,
             "celsius": [[...]], "pose": ..., "state": ...}
The simulation's `frame` field is an INDEX, not the temperature matrix.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np


class InvalidFrameError(ValueError):
    pass


def parse_packet(payload):
    """Return (temperature_matrix, safe metadata to return with inference)."""
    if isinstance(payload, list):
        return payload, {}
    if not isinstance(payload, dict):
        raise InvalidFrameError('Expected a JSON object with celsius/frame temperatures, or a nested grid')

    meta = {}
    if 'celsius' in payload:
        # The *actual* teammate format. Here "frame" is a numeric frame index.
        if any(key in payload for key in ('thermal_grid', 'temperatures', 'thermal')):
            raise InvalidFrameError('Provide one temperature matrix; do not mix celsius with aliases')
        matrix = payload['celsius']
        if 'width' in payload and payload['width'] != 32:
            raise InvalidFrameError(f"Simulation width must be 32, got {payload['width']!r}")
        if 'height' in payload and payload['height'] != 24:
            raise InvalidFrameError(f"Simulation height must be 24, got {payload['height']!r}")
        frame_num = payload.get('frame_id', payload.get('frame'))
        if frame_num is not None:
            if isinstance(frame_num, bool) or not isinstance(frame_num, (int, float, str)):
                raise InvalidFrameError('frame/frame_id must be a number or string')
            meta['frame_id'] = frame_num
        if 't' in payload:
            t = payload['t']
            if isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t):
                raise InvalidFrameError('t must be a finite numeric simulation timestamp')
            meta['timestamp_s'] = t
        if isinstance(payload.get('pose'), dict):
            pose = payload['pose']
            if all(isinstance(pose.get(k), (int, float)) and not isinstance(pose.get(k), bool)
                   and math.isfinite(pose[k]) for k in ('x', 'y', 'yaw_deg')):
                meta['robot_pose'] = {k: pose[k] for k in ('x', 'y', 'yaw_deg')}
        if isinstance(payload.get('state'), str):
            meta['simulation_state'] = payload['state'][:40]
        return matrix, meta

    options = [key for key in ('frame', 'thermal_grid', 'temperatures', 'thermal') if key in payload]
    if len(options) != 1:
        raise InvalidFrameError('Include one matrix key: celsius, frame, thermal_grid, temperatures or thermal')
    matrix = payload[options[0]]
    frame_id = payload.get('frame_id')
    if frame_id is not None:
        if isinstance(frame_id, bool) or not isinstance(frame_id, (str, int, float)):
            raise InvalidFrameError('frame_id must be a string or number')
        meta['frame_id'] = frame_id
    return matrix, meta


def extract_payload_frame(payload):
    return parse_packet(payload)[0]


def validate_frame(frame, h=24, w=32, low=-40, high=300) -> np.ndarray:
    try:
        arr = np.asarray(frame)
    except (TypeError, ValueError) as exc:
        raise InvalidFrameError(f'Cannot convert frame to array: {exc}') from None
    if arr.dtype.kind not in 'ifu':
        raise InvalidFrameError('Frame values must be real numeric Celsius temperatures')
    if arr.shape != (h, w):
        raise InvalidFrameError(f'Expected {h} rows x {w} columns (shape ({h},{w})), got {arr.shape}')
    arr = arr.astype(np.float32, copy=True)
    if not np.isfinite(arr).all():
        raise InvalidFrameError('Frame has NaN or infinite values')
    if float(arr.min()) < low or float(arr.max()) > high:
        raise InvalidFrameError(f'Frame contains temperatures outside [{low},{high}] Celsius')
    if float(np.median(arr)) > 200:
        raise InvalidFrameError('Frame may be in Kelvin; send Celsius, not Kelvin')
    return arr


def predict_packet(detector, payload):
    """Adapter used by HTTP and JSONL replay. Does not need the original project."""
    frame, meta = parse_packet(payload)
    result = detector.predict(frame, frame_id=meta.get('frame_id'))
    result.update({k: v for k, v in meta.items() if k != 'frame_id'})
    return result


def load_frame_file(path: str | Path):
    p = Path(path)
    if p.suffix.lower() == '.npy':
        return np.load(p, allow_pickle=False)
    if p.suffix.lower() in ('.csv', '.txt'):
        return np.loadtxt(p, delimiter=',' if p.suffix.lower() == '.csv' else None)
    if p.suffix.lower() == '.json':
        return extract_payload_frame(json.loads(p.read_text()))
    raise ValueError('Input file must be .json, .csv, .txt or .npy; for .jsonl use replay')
