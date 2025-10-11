import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import os

def visualize_grid(grid, title="Neural Grid", output_dir="grids", show=False):
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{title.replace(' ', '_').lower()}.png")
    
    plt.figure(figsize=(6,6))
    plt.imshow(grid, cmap="viridis", interpolation="nearest")
    plt.title(title)
    plt.colorbar(label="Weight magnitude")
    plt.tight_layout()


    plt.savefig(save_path)
    plt.close()
    print(f"Saved visualization: {os.path.abspath(save_path)}")

    if show:
        plt.show()