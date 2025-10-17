import os
import re
import matplotlib

if os.environ.get("DISPLAY", "") == "" and os.environ.get("MPLBACKEND") is None:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

def visualize_grid(grid, title="Neural Grid", output_dir="grids", show=False):
    try:
        import torch
        if isinstance(grid, torch.Tensor):
            grid = grid.detach().cpu().numpy()
    except Exception:
        pass
    
    os.makedirs(output_dir, exist_ok=True)
    safe_title = re.sub(r"[^A-Za-z0-9._-]+", "_", title).strip("_").lower()
    save_path = os.path.join(output_dir, f"{safe_title}.png")

    plt.figure(figsize=(6,6))
    plt.imshow(grid, cmap="viridis", interpolation="nearest")
    plt.title(title)
    plt.colorbar(label="Weight magnitude")
    plt.tight_layout()

    plt.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    print(f"Saved visualization: {os.path.abspath(save_path)}")
    
    return save_path