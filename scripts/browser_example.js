// Browser-based simulation. Start server: python -m sphere_scout serve --cors
// frame24x32 must be 24 arrays of 32 Celsius values.
async function classify(frame24x32, frameId = null) {
  const response = await fetch('http://127.0.0.1:8765/predict', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({frame: frame24x32, frame_id: frameId})
  });
  const result = await response.json();
  if (!response.ok) throw Error(result.error || 'Prediction request failed');
  return result;
}
