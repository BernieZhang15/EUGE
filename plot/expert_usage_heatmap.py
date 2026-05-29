import os
from typing import List, Optional


def plot_expert_usage_heatmap(
    usage_matrix: List[List[float]],
    output_path: str,
    title: Optional[str] = None,
    inactive_threshold: float = 1.0 / 64.0,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    matrix = np.asarray(usage_matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(
            f"usage_matrix must be 2D, got shape {tuple(matrix.shape)}"
        )

    display_matrix = matrix.copy()
    display_matrix[display_matrix < inactive_threshold] = np.nan
    finite_values = display_matrix[np.isfinite(display_matrix)]
    vmax = (
        max(float(finite_values.max()), inactive_threshold)
        if finite_values.size > 0
        else inactive_threshold
    )

    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="white")

    num_layers, num_experts = display_matrix.shape
    fig_width = max(8.0, num_experts * 0.75)
    fig_height = max(6.0, num_layers * 0.42)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(
        display_matrix,
        aspect="auto",
        cmap=cmap,
        interpolation="nearest",
        vmin=inactive_threshold,
        vmax=vmax,
        origin="upper",
    )

    ax.set_xlabel("Expert")
    ax.set_ylabel("Layer")
    ax.set_xticks(range(num_experts))
    ax.set_yticks(range(num_layers))
    if title:
        ax.set_title(title)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Average Routed Tokens Ratio")

    ax.tick_params(axis="both", labelsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
