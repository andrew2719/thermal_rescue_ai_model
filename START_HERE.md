# START HERE — One person, one 24×32 Celsius grid, 8 FPS

**This is the clean independent repo.** No Claude, Colab, YOLO, old `inference` folder, or training dataset needed.

1. Copy the **original CNN** `bouazizi_binary_best.pt` into `models/` (NOT Colab YOLO `best.pt`). If available also copy `models/preprocessing.json`.

2. In the repository root (conda `ml` environment):

   ```bash
   python -m pip install -r requirements.txt
   python -m unittest discover -s tests -v
   python -m sphere_scout check
   ```

3. Test one actual frame, then the teammate's packet stream. Four **actual** packets are included in `examples/simulation_sample.jsonl`:

   ```bash
   python -m sphere_scout predict examples/simulation_frame.json
   python -m sphere_scout replay examples/simulation_sample.jsonl --output sample_predictions.jsonl
   ```

   To test the complete 966-frame run, copy the supplied `thermal_run.jsonl` into this repository and run:

   ```bash
   python -m sphere_scout replay thermal_run.jsonl --output predictions.jsonl
   ```

4. Start the server and test a live 8-FPS stream:

   ```bash
   # Terminal 1
   python -m sphere_scout serve --cors

   # Terminal 2
   python scripts/stream_simulation.py thermal_run.jsonl --fps 8 --limit 16
   ```

**HTTP:** `POST http://127.0.0.1:8765/predict`, with an unmodified JSON object from one JSONL line. The server returns a CNN yes/no, an optional one-person approximate warm-region box, one 3×4 grid, plus a separate *unconfirmed* heat-region candidate and original frame/timestamp/pose.

**Accuracy warning:** All 966 uploaded simulation frames are cooler than the CNN's known training room medians, so presence accuracy is **not validated** on these frames. The file has no known human labels. A hotspot is not proof of a person. Binary correctness requires labeled simulator examples.

See `README.md` for full response schema, GitHub push, coordinate convention, and LAN networking.
