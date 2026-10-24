"""Test a teammate-sent file as soon as it arrives; no model retraining needed.
Run from repository root:
    python scripts/test_sim_input.py /path/to/simulation_frame.json
"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sphere_scout import SphereScoutDetector
from sphere_scout.io import load_frame_file

if len(sys.argv) != 2:
    raise SystemExit('Usage: python scripts/test_sim_input.py /path/to/simulation_frame.json')
detector = SphereScoutDetector()
frame = load_frame_file(sys.argv[1])
result = detector.predict(frame)
print(json.dumps(result, indent=2))
