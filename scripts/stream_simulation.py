"""Send the teammate's actual thermal_run.jsonl packets to the running server.

Run the server in terminal 1:
  python -m sphere_scout serve
Run this from repository root in terminal 2:
  python scripts/stream_simulation.py thermal_run.jsonl --limit 16

Default 8 FPS: 0.125 seconds between POSTs. Set --fps 0 for fast testing.
"""
from __future__ import annotations
import argparse
import http.client
import json
import time
from pathlib import Path
from urllib.parse import urlsplit


def main():
    p = argparse.ArgumentParser()
    p.add_argument('file', help='Teammate .jsonl file')
    p.add_argument('--url', default='http://127.0.0.1:8765/predict')
    p.add_argument('--fps', type=float, default=8.0)
    p.add_argument('--limit', type=int, default=16)
    args = p.parse_args()
    if args.fps < 0 or args.limit < 1:
        p.error('--fps must be >= 0, --limit must be >= 1')
    parsed = urlsplit(args.url)
    if parsed.scheme != 'http' or not parsed.hostname:
        p.error('Use a regular http://host:port/predict URL')
    conn = http.client.HTTPConnection(parsed.hostname, parsed.port or 80, timeout=10)
    total = 0
    next_at = time.monotonic()
    try:
        for line in Path(args.file).open(encoding='utf-8'):
            if not line.strip():
                continue
            body = line.strip().encode()
            if args.fps:
                delay = next_at - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                next_at += 1 / args.fps
            conn.request('POST', parsed.path or '/predict', body, {'Content-Type': 'application/json'})
            response = conn.getresponse()
            result = json.loads(response.read())
            if response.status != 200:
                raise RuntimeError(f'HTTP {response.status}: {result}')
            total += 1
            if total <= 3 or total % 8 == 0:
                print(f"frame {result.get('frame_id')}: human={result['human_detected']}, "
                      f"grid={result['grid_3x4']}, candidate={result['thermal_candidate'] is not None}, "
                      f"warnings={result['warnings']}", flush=True)
            if total >= args.limit:
                break
        print(f'Successfully sent {total} simulation frames.')
    finally:
        conn.close()

if __name__ == '__main__':
    main()
