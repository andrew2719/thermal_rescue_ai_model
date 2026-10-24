"""Optional PNG heatmap overlay for debugging real simulation frames."""
from pathlib import Path
import numpy as np

def save_debug_image(frame: np.ndarray, result: dict, output: str | Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(frame, interpolation='nearest', cmap='inferno', origin='upper')
    fig.colorbar(im, ax=ax, label='°C')
    for region in result['detections']:
        x1, y1, x2, y2 = region['bbox_xyxy_pixels']
        ax.add_patch(Rectangle((x1-0.5, y1-0.5), x2-x1, y2-y1,
                               fill=False, edgecolor='cyan', linewidth=2))
        if region.get('edge_pixels_xy'):
            xs, ys = zip(*region['edge_pixels_xy'])
            ax.scatter(xs, ys, s=10, c='lime', marker='s')
        cx, cy = region['center_pixel_xy']
        ax.plot(cx-0.5, cy-0.5, marker='+', color='white', markersize=14)
    status = 'HUMAN' if result['human_detected'] else 'NO HUMAN'
    ax.set_title(f'{status} | score={result["human_probability"]:.3f} | '
                 f'{"estimated box" if result["detections"] else "no estimated box"}')
    ax.set_xlabel('column (0..31)')
    ax.set_ylabel('row (0..23)')
    fig.tight_layout()
    fig.savefig(output, dpi=140)
    plt.close(fig)
