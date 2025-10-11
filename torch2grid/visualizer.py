import matplotlib.pyplot as plt

def visualize_grid(grid, title="Neural Grid"):
    plt.figure(figsize=(6,6))
    plt.imshow(grid, cmap="viridis", interpolation="nearest")
    plt.title(title)
    plt.colorbar(label="Weight magnitude")
    plt.show()