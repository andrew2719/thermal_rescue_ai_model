"""Entrypoint: python -m sphere_scout [check | predict FILE | replay FILE.jsonl | serve]."""
import argparse
import json
import sys
from .pipeline import SphereScoutDetector
from .io import load_frame_file, predict_packet


def main():
    p = argparse.ArgumentParser(description='Sphere-Scout CNN inference + one estimated thermal region')
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('check', help='Verify the CNN checkpoint loads (no training)')
    pr = sub.add_parser('predict', help='Predict a saved 24x32 Celsius sample')
    pr.add_argument('file', help='.json, .csv, .txt or .npy')
    pr.add_argument('--visualize', metavar='IMAGE.png', help='Save annotated thermal PNG (matplotlib)')
    rp = sub.add_parser('replay', help='Offline: run CNN on real simulation .jsonl frames')
    rp.add_argument('file', help='Teammate thermal_run.jsonl')
    rp.add_argument('--output', default='simulation_predictions.jsonl', help='JSONL prediction output')
    rp.add_argument('--limit', type=int, default=None, help='Only process first N frames (default all)')
    sv = sub.add_parser('serve', help='HTTP server')
    sv.add_argument('--host', default='127.0.0.1')
    sv.add_argument('--port', type=int, default=8765)
    sv.add_argument('--cors', action='store_true', help='Enable CORS for browser simulations')
    args = p.parse_args()
    try:
        det = SphereScoutDetector()
        if args.command == 'check':
            print(json.dumps({'status': 'ready', 'model_version': det.model_version,
                              'cnn_parameters': 419841, 'input_shape': [24, 32]}, indent=2))
        elif args.command == 'predict':
            frame = load_frame_file(args.file)
            result = det.predict(frame)
            print(json.dumps(result, indent=2))
            if args.visualize:
                from .io import validate_frame
                from .visualize import save_debug_image
                save_debug_image(validate_frame(frame), result, args.visualize)
                print(f'Saved {args.visualize}', file=sys.stderr)
        elif args.command == 'replay':
            from pathlib import Path
            if args.limit is not None and args.limit < 1:
                raise ValueError('--limit must be >= 1')
            frames, positives, located, warnings, candidates = 0, 0, 0, 0, 0
            with Path(args.file).open(encoding='utf-8') as source, Path(args.output).open('w', encoding='utf-8') as sink:
                for line_no, line in enumerate(source, 1):
                    if not line.strip():
                        continue
                    try:
                        result = predict_packet(det, json.loads(line))
                    except (ValueError, TypeError, KeyError) as exc:
                        raise ValueError(f'JSONL line {line_no}: {exc}') from exc
                    sink.write(json.dumps(result, allow_nan=False) + '\n')
                    frames += 1
                    positives += int(result['human_detected'])
                    located += int(result['localization_available'])
                    warnings += int(result['out_of_training_temperature_range'])
                    candidates += int(result['thermal_candidate'] is not None)
                    if args.limit is not None and frames >= args.limit:
                        break
            print(json.dumps({'frames_processed': frames, 'cnn_predicted_human': positives,
                              'cnn_confirmed_localizations': located,
                              'unconfirmed_warm_region_candidates': candidates,
                              'frames_outside_training_temperature_range': warnings,
                              'results_file': args.output,
                              'note': 'Predictions are NOT accuracy metrics; JSONL has no ground-truth human labels.'}, indent=2))
        elif args.command == 'serve':
            from .server import create_server
            srv = create_server(det, args.host, args.port, args.cors)
            print(f'Sphere-Scout ready at http://{args.host}:{srv.server_port} (POST /predict)', flush=True)
            try:
                srv.serve_forever()
            except KeyboardInterrupt:
                pass
            finally:
                srv.server_close()
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f'ERROR: {e}', file=sys.stderr)
        sys.exit(2)

if __name__ == '__main__':
    main()
