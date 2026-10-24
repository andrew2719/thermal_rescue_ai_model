"""Standard-library HTTP bridge. No Flask/FastAPI needed."""
from __future__ import annotations
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from .io import InvalidFrameError, predict_packet

MAX_BODY_BYTES = 256_000


def create_server(detector, host='127.0.0.1', port=8765, cors=False):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, *_args):
            return

        def send_json(self, code, obj):
            data = json.dumps(obj, allow_nan=False).encode('utf-8')
            self.send_response(code)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(data)))
            if cors:
                self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(data)

        def do_OPTIONS(self):
            if not cors:
                return self.send_json(405, {'error': 'Start with --cors to enable browser access'})
            self.send_response(204)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
            self.send_header('Content-Length', '0')
            self.end_headers()

        def do_GET(self):
            if self.path == '/health':
                return self.send_json(200, {'status': 'ok', 'model_version': detector.model_version,
                                            'expected_shape': [24, 32], 'units': 'Celsius', 'max_fps_required': 8})
            if self.path == '/schema':
                return self.send_json(200, {'request': {'celsius': 'simulation 24 rows x 32 Celsius numbers', 'frame': 'simulation numeric index', 't': 'timestamp in seconds', 'width': 32, 'height': 24, 'alternate': {'frame': 'canonical 24x32 grid'}},
                                            'method': 'POST', 'path': '/predict',
                                            'response_fields': ['human_detected','human_probability','detections','grid_3x4']})
            return self.send_json(404, {'error': 'Use GET /health, GET /schema, POST /predict'})

        def do_POST(self):
            if self.path != '/predict':
                self.close_connection = True
                return self.send_json(404, {'error': 'Use POST /predict'})
            try:
                size = int(self.headers['Content-Length'])
            except (ValueError, TypeError, KeyError):
                self.close_connection = True
                return self.send_json(411, {'error': 'Content-Length required'})
            if not 0 < size <= MAX_BODY_BYTES:
                self.close_connection = True
                return self.send_json(413, {'error': f'Request must be between 1 and {MAX_BODY_BYTES} bytes'})
            try:
                body = json.loads(self.rfile.read(size))
                result = predict_packet(detector, body)
                return self.send_json(200, result)
            except (ValueError, InvalidFrameError, TypeError, KeyError, json.JSONDecodeError) as exc:
                return self.send_json(400, {'error': str(exc)})
            except Exception as exc:
                return self.send_json(500, {'error': f'{type(exc).__name__}: {exc}'})
    return ThreadingHTTPServer((host, port), Handler)
