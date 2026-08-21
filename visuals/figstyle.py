"""
Shared heatmap style for the paper's table-to-figure conversions.

Colour convention:
    POSITIVE ("Blues")  -> higher is better  (accuracy, agreement,
                           rationale provision, similarity)
    NEGATIVE ("Reds")   -> higher is worse   (ASR, benign-flip rate,
                           delta-accuracy degradation)
Keeping these two families apart means a reader never has to check the
caption to know whether a dark cell is good or bad.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

POSITIVE = "Blues"
NEGATIVE = "Reds"

FONT_SIZE = 8
GRID_COLOR = "white"
GRID_LW = 1.4

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "font.size": FONT_SIZE,
    "axes.linewidth": 0.6,
    "pdf.fonttype": 42,   # IEEE rejects Type 3 fonts
    "ps.fonttype": 42,
})


def draw_heatmap(data, row_labels, col_labels, ax, cmap=POSITIVE,
                 vmin=None, vmax=None, cbar_label=None, fmt="{:.1f}",
                 sep_after=None):
    """Annotated heatmap. sep_after = column index to draw a rule after."""
    data = np.asarray(data, dtype=float)
    vmin = np.nanmin(data) if vmin is None else vmin
    vmax = np.nanmax(data) if vmax is None else vmax
    norm = Normalize(vmin=vmin, vmax=vmax)

    im = ax.imshow(data, cmap=plt.get_cmap(cmap), norm=norm, aspect="auto")

    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_xticklabels(col_labels, fontsize=FONT_SIZE)
    ax.set_yticklabels(row_labels, fontsize=FONT_SIZE)
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False,
                   length=0)

    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color=GRID_COLOR, linestyle="-", linewidth=GRID_LW)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    thresh = vmin + 0.60 * (vmax - vmin)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            ax.text(j, i, fmt.format(val), ha="center", va="center",
                    fontsize=FONT_SIZE,
                    color="white" if val > thresh else "black")

    if sep_after is not None:
        ax.axvline(sep_after + 0.5, color="0.25", lw=1.0)

    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.030, pad=0.025)
    cbar.outline.set_linewidth(0.5)
    cbar.ax.tick_params(labelsize=FONT_SIZE - 1, width=0.5)
    if cbar_label:
        cbar.set_label(cbar_label, fontsize=FONT_SIZE)
    return im


def save(fig, stem, outdir="/mnt/user-data/outputs"):
    fig.tight_layout(pad=0.3)
    for ext in ("pdf", "png"):
        fig.savefig(f"{outdir}/{stem}.{ext}", dpi=400, bbox_inches="tight")
