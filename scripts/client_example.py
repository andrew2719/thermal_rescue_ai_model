"""Minimal Python simulation client, standard library only.
Run the server first; then: python scripts/client_example.py
"""
from pathlib import Path
import json
import urllib.request
root = Path(__file__).resolve().parent.parent
frame = json.loads((root / 'examples' / 'synthetic_human.json').read_text())['frame']
request = urllib.request.Request(
    'http://127.0.0.1:8765/predict',
    data=json.dumps({'frame': frame, 'frame_id': 'demo-001'}).encode('utf-8'),
    headers={'Content-Type': 'application/json'}, method='POST')
with urllib.request.urlopen(request, timeout=5) as response:
    result = json.load(response)
print('HUMAN:', result['human_detected'])
print('LOCATION:', result['detections'][0]['center_uv_normalized'] if result['detections'] else None)
print('GRID:', result['grid_3x4'])
