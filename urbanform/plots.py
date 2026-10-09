"""Maps and charts. Colour-blind-safe qualitative palette; no basemap tiles (no network needed)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442", "#7F7F7F"]
SPARSE = "#E6E6E6"


def _outline(ax, boundary):
    boundary.boundary.plot(ax=ax, color="#333333", linewidth=0.6)
    ax.set_axis_off()
    ax.set_aspect("equal")


def typology_map(grid, boundary, path, names=None):
    fig, ax = plt.subplots(figsize=(9, 9))
    colors = grid["type"].map(lambda t: SPARSE if t == 0 else PALETTE[(int(t) - 1) % len(PALETTE)])
    grid.plot(ax=ax, color=colors, linewidth=0)
    _outline(ax, boundary)
    types = sorted(t for t in grid["type"].unique() if t > 0)
    handles = [Patch(color=PALETTE[(t - 1) % len(PALETTE)], label=(names or {}).get(t, f"Type {t}")) for t in types]
    handles.append(Patch(color=SPARSE, label="Sparse / unbuilt (not clustered)"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=9)
    ax.set_title(f"Urban form types ({int(grid['cell_size'].iloc[0])} m cells)")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def gap_map(grid, boundary, path):
    fig, ax = plt.subplots(figsize=(9, 9))
    grid[~grid["built"]].plot(ax=ax, color=SPARSE, linewidth=0)
    b = grid[grid["built"]]
    b.plot(ax=ax, column="gap", cmap="RdBu_r", vmin=-1, vmax=1, linewidth=0, legend=True,
           legend_kwds={"label": "Building-density rank minus intersection-density rank", "shrink": 0.6})
    grid[grid["dense_poorly_connected"]].boundary.plot(ax=ax, color="black", linewidth=0.6)
    _outline(ax, boundary)
    ax.set_title("Density vs street-connectivity gap (black outline = dense & poorly connected)")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def profile_heatmap(zmean, path, names=None):
    fig, ax = plt.subplots(figsize=(1.0 * len(zmean.columns) + 3, 0.7 * len(zmean) + 2))
    im = ax.imshow(zmean.to_numpy(), cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax.set_xticks(range(len(zmean.columns)))
    ax.set_xticklabels(zmean.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(zmean)))
    ax.set_yticklabels([(names or {}).get(t, f"Type {t}") for t in zmean.index])
    for i in range(zmean.shape[0]):
        for j in range(zmean.shape[1]):
            ax.text(j, i, f"{zmean.iat[i, j]:.1f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Mean z-score", shrink=0.8)
    ax.set_title("Type profiles (z-scores across built cells)")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def subcity_types(share_table, path, names=None):
    fig, ax = plt.subplots(figsize=(10, 0.5 * len(share_table) + 2.5))
    left = np.zeros(len(share_table))
    for t in share_table.columns:
        ax.barh(share_table.index, share_table[t].to_numpy(), left=left,
                color=PALETTE[(int(t) - 1) % len(PALETTE)], label=(names or {}).get(t, f"Type {t}"))
        left += share_table[t].to_numpy()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of built cells")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.15), frameon=False, fontsize=9)
    ax.set_title("Urban form type mix by subcity")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def lisa_map(sub, col, boundary, path):
    colors = {"HH": "#B2182B", "LL": "#2166AC", "HL": "#EF8A62", "LH": "#67A9CF", "ns": SPARSE}
    fig, ax = plt.subplots(figsize=(9, 9))
    sub.plot(ax=ax, color=sub[f"lisa_{col}"].map(colors), linewidth=0)
    _outline(ax, boundary)
    ax.legend(handles=[Patch(color=c, label=k) for k, c in colors.items()], loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=9)
    ax.set_title(f"Local Moran's I clusters: {col}")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
