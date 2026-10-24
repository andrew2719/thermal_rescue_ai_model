# Sphere-Scout Human Detection — Clean Inference Server

A small **standalone** repository for your one-person simulated rescue-robot demo.
No previous project's Python modules, manifests or 160k-frame dataset are required.

**Input:** one `24 × 32` array of **float temperatures in Celsius**.

**Output:** CNN HUMAN/NO HUMAN decision, score, **at most one estimated warm-region** bounding box and its thermal boundary pixels, and a 3×4 grid cell. The box is **not learned human segmentation** and may fail near hot walls or other warm objects.

## Folder layout

```text
SphereScout_Clean_CNN/
├── config.json                     # Original trained CNN threshold and global Celsius normalization
├── requirements.txt
├── models/
│   ├── PUT_CNN_CHECKPOINT_HERE.txt
│   └── bouazizi_binary_best.pt     # COPY YOUR ORIGINAL CNN WEIGHTS HERE; not included in this ZIP
├── sphere_scout/
│   ├── cnn.py                      # 419,841-parameter CNN definition + checkpoint loader
│   ├── localization.py             # Approximate warm-blob edge and bounding box
│   ├── pipeline.py                 # One predict(frame) function
│   ├── io.py                       # Celsius input adapter
│   ├── server.py                   # Local HTTP bridge
│   ├── visualize.py                # Optional thermal image with box and edges
│   └── __main__.py                 # CLI
├── examples/                      # Synthetic + real simulation JSONL samples; no human labels
├── scripts/                       # Simulation JSONL sender (8 FPS) + Python/browser clients
└── tests/                         # Unit tests / HTTP contract checks
```

## 1. Add your CNN weights (important)

Copy the following file **from the old CNN training project** into this new repo:

```text
OLD_PROJECT/models/bouazizi_binary_best.pt
    --> NEW_REPO/models/bouazizi_binary_best.pt
```

**Do not use `best.pt` from Colab/YOLO.** Its weights are incompatible with this CNN. `config.json` checks the known original CNN checkpoint SHA-256 and stops if the wrong file was copied. This archive does **not** contain the original CNN `.pt` because that specific file was not uploaded.

Optionally, copy `OLD_PROJECT/models/preprocessing.json` to `NEW_REPO/models/preprocessing.json`. That retains the exact saved training normalization; `config.json` already has the reported normalization mean **22.657346°C**, std **1.571359°C** rounded to six decimals. Saved CNN decision threshold: **0.8452931642532349**.

## 2. Install and run

Use Python 3.10 or newer. From inside this new repository:

```bash
# You can use your existing conda environment, if torch/numpy/matplotlib are already installed:
conda activate ml
# or use a clean environment and install dependencies:
python -m pip install -r requirements.txt

python -m sphere_scout check
python -m sphere_scout predict examples/synthetic_human.json --visualize output/demo.png
python -m sphere_scout serve --cors
```

Create the output directory before using `--visualize`:

```bash
mkdir -p output
```

When the server prints `Sphere-Scout ready`, leave the terminal running. To run on another trusted laptop in the same LAN, use `--host 0.0.0.0 --cors`; teammates must connect to your LAN IP, not their `127.0.0.1`. Do not expose this unauthenticated demo server to the internet.

## 3. Simple HTTP contract (suggested stable version)

`GET http://127.0.0.1:8765/health` — reports model ready.

`POST http://127.0.0.1:8765/predict` — JSON request:

```json
{
  "frame_id": "camera-007",
  "frame": [
    "24 rows total, each containing 32 Celsius numbers"
  ]
}
```

The above shows the **shape**, not valid JSON input. Use `examples/synthetic_human.json` for a real syntactically valid full request (it contains 768 temperatures). You can try:

```bash
curl -s -H 'Content-Type: application/json' --data-binary @examples/synthetic_human.json http://127.0.0.1:8765/predict
```

Response (illustrative fields/numbers only):

```json
{
  "human_detected": true,
  "human_probability": 0.93,
  "human_score": 0.93,
  "model_version": "bouazizi-binary-cnn-1.0.0",
  "threshold": 0.8452931642532349,
  "detections": [
    {
      "type": "thermal_hotspot_estimate",
      "bbox_xyxy_pixels": [11, 5, 21, 19],
      "bbox_xyxy_normalized": [0.34375, 0.20833, 0.65625, 0.79167],
      "center_pixel_xy": [16, 12],
      "center_uv_normalized": [0.5, 0.5],
      "grid_cell_row_col": [1, 2],
      "edge_pixels_xy": [[11, 8], [12, 7]]
    }
  ],
  "grid_3x4": [[0,0,0,0],[0,0,1,0],[0,0,0,0]],
  "localization_method": "heuristic_thermal_blob",
  "localization_available": true,
  "warnings": []
}
```

If the CNN says HUMAN but no distinct warm region exists, `detections` is empty and `localization_available=false`. A zero in `grid_3x4` means **no localized blob**, not necessarily NO HUMAN. The `human_probability` is a sigmoid score, **not a calibrated probability**.

### Coordinate convention

- `frame[y][x]` with `y=0..23`, `x=0..31`; `(0,0)` is the top-left.
- `bbox_xyxy_pixels=[left,top,right,bottom]` — right/bottom **exclusive** (coordinates refer to pixel boundaries).
- `center_pixel_xy=[x,y]` in the original sensor's pixel-boundary coordinate system (range 0–32, 0–24). Divide by `[32,24]` for normalized `[u,v]`.
- `edge_pixels_xy` are **warm-component boundary pixels**, not verified human silhouette/body edges. Order is not a connected contour path.
- `grid_3x4` is three **rows** by four **columns**, with a 1 in the cell containing the selected blob **center**; it is not the original input grid.
- The box is a 2D region. The teammate's depth calculation needs separate geometry/calibration/range assumptions.

### Actual simulation input (VERIFIED: 966 frames at 8 FPS)

Your teammate's `thermal_run.jsonl` is supported **as-is**, no wrapper or conversion:

```json
{"t": 5.375, "frame": 0, "width": 32, "height": 24, "celsius": [["24 rows of 32 Celsius values"]], "pose": null, "state": null}
```

The line above is illustrative; `examples/simulation_sample.jsonl` contains four **real complete** packets from the teammate's upload. Each line of `.jsonl` is one JSON object. `celsius` is the temperature matrix; `frame` is the **numeric frame index** (not the matrix!). `t` is simulation time in seconds, and `pose` is the robot pose (`x`, `y`, `yaw_deg`), not the person's coordinates. `state` describes the robot's control state and is **not a human label**. The model only receives `celsius`. `frame_id`, `timestamp_s`, `robot_pose`, and `simulation_state` are echoed in responses for easy alignment.

The old canonical `{"frame": [[32 floats], ... 24 rows]}` is also accepted, as are the earlier aliases. Input is always 24 rows × 32 columns in degrees Celsius.

**Important domain mismatch:** All 966 frames have median room temperatures between **12.12°C and 17.26°C**, below the original CNN's observed training frame-median range **18.04–24.83°C**. Frames are flagged with `out_of_training_temperature_range=true`. The supplied file contains no human ground truth labels, so it cannot establish accuracy. The simulated scene also uses only a few distinct temperature levels, unlike ordinary MLX90640 measurements. You can run the model for a hackathon demonstration, but presence decisions should be treated as experimental. Do not silently alter Celsius values just to force a desired output.

### Replay the actual JSONL file

Copy `thermal_run.jsonl` into the repo root (or use its absolute path). With the **original CNN checkpoint installed**:

```bash
# Test first 16 frames, no simulated wait; predictions are written to JSONL
python -m sphere_scout replay thermal_run.jsonl --limit 16 --output sample_predictions.jsonl

# Evaluate all 966 frames (no human ground-truth metrics available)
python -m sphere_scout replay thermal_run.jsonl --output predictions.jsonl
```

To send the real packets to the HTTP server at **8 frames per second**, start `python -m sphere_scout serve` in one terminal and run:

```bash
python scripts/stream_simulation.py thermal_run.jsonl --fps 8 --limit 16
```

Omit `--limit`? By default the example streams only 16 frames to keep testing fast. Use `--limit 966` to replay the whole run at 8 FPS (around two minutes). Set `--fps 0` for an unthrottled smoke test.

The server can process one packet per POST and returns its corresponding `frame_id`, `timestamp_s`, `human_detected`, one confirmed-region `grid_3x4`, and optional box. No person can be confirmed from an empty grid: a warm thermal candidate is separate from the CNN's decision.

#### Thermal candidates versus HUMAN detections

- `human_detected`: CNN-only decision using the frozen original threshold.
- `detections`: at most one approximate warm-region box **only when the CNN predicts HUMAN**.
- `grid_3x4`: 1 in the selected cell only if the CNN predicts HUMAN **and** the warm region is found; otherwise all zeros.
- `thermal_candidate` and `candidate_grid_3x4`: raw candidate hotspot and its grid independent of CNN; **not** evidence a human is present. Handy for debugging out-of-domain frames.
- Estimated `edge_pixels_xy` are boundary pixels of a warm component, not a person's physical edges.

The region finder rejects nearly full-width warm background bands seen at the start of the actual run. This is a heuristic, not trained or evaluated human localization.

## 4. Test the teammate's sample (already included)

For a single real simulation packet (24×32 Celsius), save one JSONL line as `simulation_frame.json`, or use CSV/NPY, then:

```bash
python scripts/test_sim_input.py simulation_frame.json
python -m sphere_scout predict simulation_frame.json --visualize output/actual_simulation.png
```

For a Python sim: `scripts/client_example.py`. For a browser/Three.js sim: `scripts/browser_example.js` (needs server `--cors`).

## 5. Verify tests and limitations

```bash
python -m unittest discover -s tests -v
```

Tests cover the CNN's 419,841-parameter architecture, a synthetic checkpoint's strict loader behavior, input validation, thermal-region boxes, JSON contract, the real simulation packet structure, and local HTTP round trip **without** requiring your original checkpoint. Run `python -m sphere_scout check` after copying the real `.pt` to verify actual checkpoint loading.

The previously trained CNN's KU Leuven validation results were approximately 90.0% human recall, 61.0% no-human recall and 75.5% balanced accuracy using its saved threshold. **Those are not performance guarantees for simulation frames.** Hot objects can cause false alarms; people close to warm backgrounds may have no blob box. We have not validated the new localizer on human ground-truth boxes; this is a demonstration heuristic.

## 6. Push the clean repository

Inside `SphereScout_Clean_CNN` (after copying the original CNN weights):

```bash
git init
git add .
git commit -m "Ship standalone Sphere-Scout CNN inference server"
git branch -M main
git remote add origin https://github.com/YOUR_TEAM/YOUR_NEW_REPO.git
git push -u origin main
```

Only the inference runtime + the one CNN checkpoint are needed. **Don't push the huge training dataset, notebooks, or YOLO experiments.** If `.pt` distribution is restricted in your event, share the checkpoint with teammates through the event's approved method instead.
# thermal_rescue_ai_model
