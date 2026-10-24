"""Run: python -m unittest discover -s tests -v
Tests can run WITHOUT the real CNN checkpoint; mocked scores only test interface wiring.
"""
import http.client
import json
import tempfile
import threading
import unittest
from pathlib import Path
import numpy as np
from sphere_scout.pipeline import SphereScoutDetector
from sphere_scout.io import InvalidFrameError, validate_frame, load_frame_file
from sphere_scout.localization import locate_one_hot_region
from sphere_scout.server import create_server
from sphere_scout.cnn import BouaziziBinaryCNN


class FixedPredictor:
    def __init__(self, score):
        self.score = score

    def predict_probability(self, frame):
        return self.score


class UnitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parent.parent
        cls.human = load_frame_file(cls.root / 'examples' / 'synthetic_human.json')
        cls.empty = load_frame_file(cls.root / 'examples' / 'synthetic_empty.json')

    def test_cnn_architecture(self):
        net = BouaziziBinaryCNN()
        self.assertEqual(sum(p.numel() for p in net.parameters()), 419841)
        import torch
        with torch.no_grad():
            self.assertEqual(tuple(net(torch.zeros(2, 1, 24, 32)).shape), (2,))

    def test_positive_has_one_region_and_edges(self):
        r = locate_one_hot_region(validate_frame(self.human))
        self.assertIsNotNone(r)
        self.assertLess(abs(r['center_pixel_xy'][0]-16), 5)
        self.assertLess(abs(r['center_pixel_xy'][1]-12), 5)
        self.assertGreater(len(r['edge_pixels_xy']), 2)
        self.assertEqual(len(r['bbox_xyxy_normalized']), 4)
        self.assertTrue(all(0 <= v <= 1 for v in r['bbox_xyxy_normalized']))

    def test_no_hotspot_for_empty(self):
        self.assertIsNone(locate_one_hot_region(validate_frame(self.empty)))

    def test_detected_response_contains_grid_and_box(self):
        det = SphereScoutDetector(predictor=FixedPredictor(0.99))
        out = det.predict(self.human, frame_id='sim-42')
        self.assertTrue(out['human_detected'])
        self.assertEqual(out['frame_id'], 'sim-42')
        self.assertEqual(len(out['detections']), 1)
        self.assertEqual(sum(map(sum, out['grid_3x4'])), 1)
        self.assertTrue(out['localization_available'])

    def test_no_detection_has_no_box(self):
        det = SphereScoutDetector(predictor=FixedPredictor(0.1))
        out = det.predict(self.human)
        self.assertFalse(out['human_detected'])
        self.assertEqual(out['detections'], [])
        self.assertEqual(sum(map(sum, out['grid_3x4'])), 0)

    def test_cnn_says_human_but_no_warm_region(self):
        det = SphereScoutDetector(predictor=FixedPredictor(0.99))
        out = det.predict(self.empty)
        self.assertTrue(out['human_detected'])
        self.assertFalse(out['localization_available'])
        self.assertEqual(out['detections'], [])

    def test_input_validation(self):
        bad = [np.zeros((32,24)), np.zeros((768,)), np.full((24,32), np.nan),
               [['hello']*32 for _ in range(24)], np.full((24,32), 500.0)]
        for item in bad:
            with self.subTest(type=str(type(item))):
                with self.assertRaises(InvalidFrameError):
                    validate_frame(item)
        self.assertEqual(validate_frame(self.human).shape, (24,32))

    def test_server_roundtrip_and_errors(self):
        d = SphereScoutDetector(predictor=FixedPredictor(0.99))
        server = create_server(d, '127.0.0.1', port=0, cors=True)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        try:
            connection = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=3)
            connection.request('GET', '/health')
            resp = connection.getresponse()
            self.assertEqual(resp.status, 200)
            self.assertEqual(json.loads(resp.read())['status'],'ok')
            data = json.dumps({'frame': self.human, 'frame_id':'3'}).encode()
            connection.request('POST','/predict',data,{'Content-Type':'application/json'})
            resp = connection.getresponse()
            self.assertEqual(resp.status,200)
            result = json.loads(resp.read())
            self.assertTrue(result['human_detected'])
            self.assertEqual(result['frame_id'], '3')
            connection.request('POST','/predict',b'{"frame":[1,2,3]}',{'Content-Type':'application/json'})
            resp = connection.getresponse()
            self.assertEqual(resp.status,400)
            self.assertIn('Expected', json.loads(resp.read())['error'])
            connection.close()
        finally:
            server.shutdown()
            server.server_close()


if __name__ == '__main__':
    unittest.main()
