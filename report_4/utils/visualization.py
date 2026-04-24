import matplotlib.pyplot as plt
from pathlib import Path


def visualize_matrix(matrix, title, filename):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.spy(matrix, markersize=5, aspect='auto')
    ax1.set_title(f'{title} - Spy Plot')
    ax1.set_xlabel('Columns')
    ax1.set_ylabel('Rows')

    im = ax2.imshow(matrix, cmap='RdBu_r', aspect='auto', interpolation='nearest')
    ax2.set_title(f'{title} - Heatmap')
    ax2.set_xlabel('Columns')
    ax2.set_ylabel('Rows')
    plt.colorbar(im, ax=ax2)

    plt.tight_layout()

    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization: {filename}")
