"""Pictures of an episode: the map, a heat map by cell, and a short GIF drawn with matplotlib."""

import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from PIL import Image

MAP = ListedColormap(["#ffffff", "#4a5568", "#dbeafe", "#fde68a"])     # free, shelf, parking, station


def draw_map(ax, wh):
    ax.imshow(wh.grid, cmap=MAP, vmin=0, vmax=3, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])


def heatmap(wh, counts, title, path, log=True):
    mask = wh.grid == 1
    show = np.ma.masked_where(mask, np.log1p(counts) if log else counts)
    fig, ax = plt.subplots(figsize=(max(6, wh.width / 10), max(3.6, wh.height / 10)), dpi=130)
    ax.imshow(np.where(mask, 0.25, 1.0), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    im = ax.imshow(show, cmap="inferno", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.8, label="log(1 + count)" if log else "count")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def gif(wh, traj, targets, path, first=0, frames=160, every=1, ms=110):
    """Robots as dots: blue going, orange in service, grey at rest."""
    out = []
    fig, ax = plt.subplots(figsize=(wh.width / 5, wh.height / 5 + 0.3), dpi=70)
    colours = {True: "#ea580c", False: "#2563eb"}
    for t in range(first, min(first + frames, len(targets)), every):
        ax.clear()
        draw_map(ax, wh)
        r, c = traj[t] // wh.width, traj[t] % wh.width
        col = ["#ea580c" if g == -2 else "#9ca3af" if g == -1 else "#2563eb" for g in targets[t]]
        ax.scatter(c, r, s=46, c=col, edgecolors="black", linewidths=0.4, zorder=3)
        ax.set_title(f"step {t}", fontsize=9)
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        out.append(Image.open(buf).convert("P", palette=Image.ADAPTIVE, colors=32))
    plt.close(fig)
    out[0].save(path, save_all=True, append_images=out[1:], duration=ms, loop=0, optimize=True)
