"""Vessel morphometry figures for the 5-patient cohort (SVC only).

Uses the validated palette from the dataviz skill: fixed categorical order
for the 5 patients (never re-cycled across figures), single-hue sequential
blue for all magnitude/heatmap encodings -- no rainbow colormap.
"""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy import ndimage as ndi

OUT = Path("results_v6")
PIDS = [1, 2, 3, 4, 5]

# Validated palette (references/palette.md) -- fixed order, never cycled.
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
SEQ_BLUE_STEPS = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ_BLUE = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE_STEPS)

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.edgecolor": GRID, "axes.labelcolor": INK_SECONDARY,
    "xtick.color": INK_MUTED, "ytick.color": INK_MUTED,
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    "axes.grid": False,
})


def load_rows():
    with open(OUT / "vessel_morphometrics.csv") as fh:
        return list(csv.DictReader(fh))


def local_density_map(vessel_mask: np.ndarray, cell: int = 16, smooth_sigma: float = 1.0) -> np.ndarray:
    h, w = vessel_mask.shape
    nh, nw = h // cell, w // cell
    grid = np.zeros((nh, nw))
    for i in range(nh):
        for j in range(nw):
            block = vessel_mask[i * cell:(i + 1) * cell, j * cell:(j + 1) * cell]
            grid[i, j] = block.mean()
    if smooth_sigma > 0:
        grid = ndi.gaussian_filter(grid, smooth_sigma)
    return grid


# ---------------------------------------------------------------------------
# Figure 1: spatial vessel-density heatmaps, all 5 patients
# ---------------------------------------------------------------------------
def fig_density_heatmaps():
    fig, axes = plt.subplots(2, 5, figsize=(22, 9.5),
                             gridspec_kw={"height_ratios": [1, 1]})
    fig.suptitle("Local SVC vessel density -- 5-patient cohort", fontsize=19,
                color=INK, y=0.99, fontweight="bold")
    fig.text(0.5, 0.955,
             "top: SVC reconstruction with vessel mask  ·  bottom: local density (16×16px cells, Gaussian-smoothed)",
             ha="center", fontsize=11.5, color=INK_SECONDARY)

    im = None
    for i, pid in enumerate(PIDS):
        d = OUT / f"patient_{pid}"
        svc = cv2.imread(str(d / "svc_enface_v5.png"), 0)
        vessel = cv2.imread(str(d / "svc_vessels_model.png"), 0) > 127

        ax0 = axes[0, i]
        rgb = np.stack([svc, svc, svc], axis=-1).astype(np.uint8)
        rgb[vessel] = [227, 73, 72]  # categorical slot 8 (red) for contrast against blue ramp below
        ax0.imshow(rgb)
        ax0.set_title(f"Patient {pid}", fontsize=13, color=INK, fontweight="bold", pad=8)
        ax0.set_xticks([]); ax0.set_yticks([])
        for s in ax0.spines.values():
            s.set_visible(False)

        density = local_density_map(vessel)
        ax1 = axes[1, i]
        im = ax1.imshow(density, cmap=SEQ_BLUE, vmin=0, vmax=0.85, interpolation="bilinear")
        ax1.set_xticks([]); ax1.set_yticks([])
        for s in ax1.spines.values():
            s.set_visible(False)

    cbar = fig.colorbar(im, ax=axes[1, :], orientation="horizontal", fraction=0.045, pad=0.06, aspect=50)
    cbar.set_label("local vessel density (fraction of cell area)", color=INK_SECONDARY, fontsize=10.5)
    cbar.ax.xaxis.set_tick_params(color=INK_MUTED, labelcolor=INK_SECONDARY)
    cbar.outline.set_visible(False)
    fig.savefig(OUT / "vessel_density_heatmaps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved vessel_density_heatmaps.png")


# ---------------------------------------------------------------------------
# Figure 2: small-multiples bar comparison across morphometric features
# ---------------------------------------------------------------------------
FEATURES = [
    ("vessel_density_pct", "Vessel density", "%"),
    ("skeleton_density_per_1k_px", "Skeleton density", "px / 1k px²"),
    ("mean_caliber_px", "Mean vessel caliber", "px"),
    ("branchpoint_density_per_10k_px", "Branch-point density", "/ 10k px²"),
    ("endpoint_density_per_10k_px", "Free-end density", "/ 10k px²"),
    ("fractal_dimension", "Fractal dimension", "D"),
    ("mean_tortuosity", "Mean tortuosity", "ratio"),
    ("n_segments", "Vessel segment count", "n"),
]


def fig_feature_small_multiples(rows):
    fig, axes = plt.subplots(2, 4, figsize=(20, 9.5))
    fig.suptitle("SVC vessel morphometry -- feature comparison across patients", fontsize=19,
                color=INK, y=1.01, fontweight="bold")
    axes = axes.flatten()
    patients = [int(r["patient"]) for r in rows]

    for ax, (key, label, unit) in zip(axes, FEATURES):
        vals = [float(r[key]) for r in rows]
        bars = ax.bar([str(p) for p in patients], vals, color=CATEGORICAL, width=0.62,
                     edgecolor=SURFACE, linewidth=2)
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:,.2f}" if abs(v) < 100 else f"{v:,.0f}",
                       (b.get_x() + b.get_width() / 2, v),
                       xytext=(0, 4), textcoords="offset points",
                       ha="center", fontsize=9, color=INK_SECONDARY)
        ax.set_title(f"{label}", fontsize=12.5, color=INK, fontweight="bold", pad=10)
        ax.set_xlabel("patient", fontsize=9.5, color=INK_MUTED)
        ax.set_ylabel(unit, fontsize=9.5, color=INK_MUTED)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(GRID)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(length=0)
        ymax = max(vals)
        ax.set_ylim(0, ymax * 1.22 if ymax > 0 else 1)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "vessel_feature_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved vessel_feature_comparison.png")


# ---------------------------------------------------------------------------
# Figure 3: patients x features fingerprint heatmap (min-max normalised)
# ---------------------------------------------------------------------------
def fig_fingerprint_heatmap(rows):
    keys = [k for k, _, _ in FEATURES] + ["faz_area_px", "faz_circularity", "faz_acircularity"]
    labels = [l for _, l, _ in FEATURES] + ["FAZ area", "FAZ circularity", "FAZ acircularity"]
    patients = [int(r["patient"]) for r in rows]

    raw = np.array([[float(r[k]) for k in keys] for r in rows])  # (patients, features)
    norm = np.zeros_like(raw)
    for j in range(raw.shape[1]):
        col = raw[:, j]
        lo, hi = col.min(), col.max()
        norm[:, j] = 0.5 if hi == lo else (col - lo) / (hi - lo)

    fig, ax = plt.subplots(figsize=(13, 5.6))
    im = ax.imshow(norm.T, cmap=SEQ_BLUE, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(patients)))
    ax.set_xticklabels([f"Patient {p}" for p in patients], fontsize=11, color=INK)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10.5, color=INK)
    ax.set_xticks(np.arange(-0.5, len(patients), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2.5)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    for i in range(raw.shape[0]):
        for j in range(raw.shape[1]):
            v = raw[i, j]
            txt = f"{v:,.2f}" if abs(v) < 100 else f"{v:,.0f}"
            lum = norm[i, j]
            color = SURFACE if lum > 0.55 else INK
            ax.text(i, j, txt, ha="center", va="center", fontsize=9, color=color)

    ax.set_title("SVC vessel + FAZ morphometry — cohort fingerprint (column-normalised per feature)",
                fontsize=14.5, color=INK, fontweight="bold", pad=14)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("relative within feature (low → high)", color=INK_SECONDARY, fontsize=10)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["lowest", "highest"])
    cbar.outline.set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT / "vessel_fingerprint_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved vessel_fingerprint_heatmap.png")


if __name__ == "__main__":
    rows = load_rows()
    fig_density_heatmaps()
    fig_feature_small_multiples(rows)
    fig_fingerprint_heatmap(rows)
