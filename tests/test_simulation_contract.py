"""Tests using four real packets from teammate's example run (no real CNN required)."""
import http.client
import json
import threading
import unittest
from pathlib import Path
import numpy as np
from sphere_scout.io import parse_packet, InvalidFrameError, predict_packet
from sphere_scout.pipeline import SphereScoutDetector
from sphere_scout.localization import locate_one_hot_region
from sphere_scout.server import create_server

ROOT = Path(__file__).resolve().parent.parent
PACKETS = [json.loads(l) for l in (ROOT / 'examples/simulation_sample.jsonl').read_text().splitlines()]


class FixedPredictor:
    def __init__(self, value): self.value = value
    def predict_probability(self, _frame): return self.value


class SimulationContractTests(unittest.TestCase):
    def test_teammate_packets_are_valid_and_8fps_compatible(self):
        for item in PACKETS:
            matrix, meta = parse_packet(item)
            self.assertEqual(np.asarray(matrix).shape, (24, 32))
            self.assertEqual(meta['frame_id'], item['frame'])
            self.assertEqual(meta['timestamp_s'], item['t'])
            self.assertEqual(item['width'], 32)
            self.assertEqual(item['height'], 24)

    def test_index_is_not_temperature_frame(self):
        item = PACKETS[0]
        arr, meta = parse_packet(item)
        self.assertIsInstance(arr, list)
        self.assertEqual(meta['frame_id'], 0)
        self.assertEqual(len(arr), 24)

    def test_reject_wrong_dimensions(self):
        item = dict(PACKETS[0], width=64)
        with self.assertRaises(InvalidFrameError): parse_packet(item)
        item = dict(PACKETS[0], height=32)
        with self.assertRaises(InvalidFrameError): parse_packet(item)

    def test_empty_initial_band_is_not_human_box(self):
        frame = np.asarray(PACKETS[0]['celsius'], dtype=np.float32)
        self.assertIsNone(locate_one_hot_region(frame))

    def test_late_warm_blob_can_produce_approximate_box(self):
        frame = np.asarray(PACKETS[-1]['celsius'], dtype=np.float32)
        region = locate_one_hot_region(frame)
        self.assertIsNotNone(region)
        self.assertGreater(len(region['edge_pixels_xy']), 1)

    def test_simulation_packet_preserves_metadata_and_candidate(self):
        detector = SphereScoutDetector(predictor=FixedPredictor(0.99))
        result = predict_packet(detector, PACKETS[-1])
        self.assertTrue(result['human_detected'])
        self.assertEqual(result['frame_id'], PACKETS[-1]['frame'])
        self.assertEqual(result['timestamp_s'], PACKETS[-1]['t'])
        self.assertEqual(result['robot_pose'], PACKETS[-1]['pose'])
        self.assertEqual(sum(map(sum, result['grid_3x4'])), 1)
        self.assertTrue(result['out_of_training_temperature_range'])
        self.assertEqual(len(result['detections']), 1)
        detector = SphereScoutDetector(predictor=FixedPredictor(0.1))
        result = predict_packet(detector, PACKETS[-1])
        self.assertFalse(result['human_detected'])
        self.assertEqual(sum(map(sum, result['grid_3x4'])), 0)
        self.assertIsNotNone(result['thermal_candidate'])
        self.assertEqual(sum(map(sum, result['candidate_grid_3x4'])), 1)

    def test_http_accepts_unmodified_simulator_json(self):
        det = SphereScoutDetector(predictor=FixedPredictor(0.99))
        server = create_server(det, host='127.0.0.1', port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            connection = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=5)
            for packet in PACKETS:
                payload = json.dumps(packet).encode()
                connection.request('POST','/predict', payload, {'Content-Type':'application/json'})
                resp = connection.getresponse()
                result = json.loads(resp.read())
                self.assertEqual(resp.status, 200, result)
                self.assertEqual(result['frame_id'], packet['frame'])
            connection.close()
        finally:
            server.shutdown(); server.server_close()

if __name__ == '__main__': unittest.main()
