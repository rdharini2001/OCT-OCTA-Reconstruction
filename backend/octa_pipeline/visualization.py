"""Figure generation: QC boundary overlays and per-patient composite panels."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .segmentation import VolumeBoundaries

_BOUNDARY_STYLE = {
    "ILM": "#e63946",
    "IPL/INL": "#ffd60a",
    "OPL/ONL": "#00b4d8",
    "RPE": "#57cc99",
}


def save_boundary_overlay(structural: np.ndarray, bounds: VolumeBoundaries, out_path: Path) -> None:
    n_bscans = structural.shape[0]
    rows = [n_bscans // 4, n_bscans // 2, 3 * n_bscans // 4]

    fig, axes = plt.subplots(1, len(rows), figsize=(6 * len(rows), 6))
    for ax, y in zip(axes, rows):
        b = structural[y]
        ax.imshow(b, cmap="gray", aspect="auto")
        x = np.arange(b.shape[1])
        ax.plot(x, bounds.ilm[y], color=_BOUNDARY_STYLE["ILM"], lw=1.3, label="ILM")
        ax.plot(x, bounds.ipl_inl[y], color=_BOUNDARY_STYLE["IPL/INL"], lw=1.3, label="IPL/INL")
        ax.plot(x, bounds.opl_onl[y], color=_BOUNDARY_STYLE["OPL/ONL"], lw=1.3, label="OPL/ONL")
        ax.plot(x, bounds.rpe[y], color=_BOUNDARY_STYLE["RPE"], lw=1.3, label="RPE")
        ax.set_title(f"B-scan {y}/{n_bscans}")
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0].legend(loc="upper right", fontsize=9, framealpha=0.85)
    fig.suptitle("Segmentation QC — ILM / IPL-INL / OPL-ONL / RPE", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def save_patient_composite(result, out_path: Path) -> None:
    mid = result.structural.shape[0] // 2
    b = result.structural[mid]
    bounds = result.bounds

    fig = plt.figure(figsize=(11, 11))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.3])

    ax0 = fig.add_subplot(gs[0, :])
    ax0.imshow(b, cmap="gray", aspect="auto")
    x = np.arange(b.shape[1])
    ax0.plot(x, bounds.ilm[mid], color=_BOUNDARY_STYLE["ILM"], lw=1.4, label="ILM")
    ax0.plot(x, bounds.ipl_inl[mid], color=_BOUNDARY_STYLE["IPL/INL"], lw=1.4, label="IPL/INL")
    ax0.plot(x, bounds.opl_onl[mid], color=_BOUNDARY_STYLE["OPL/ONL"], lw=1.4, label="OPL/ONL")
    ax0.plot(x, bounds.rpe[mid], color=_BOUNDARY_STYLE["RPE"], lw=1.4, label="RPE")
    ax0.fill_between(
        x, bounds.ilm[mid], bounds.ipl_inl[mid], color=_BOUNDARY_STYLE["IPL/INL"], alpha=0.15
    )
    ax0.fill_between(
        x, bounds.ipl_inl[mid], bounds.opl_onl[mid], color=_BOUNDARY_STYLE["OPL/ONL"], alpha=0.15
    )
    ax0.legend(loc="upper right", fontsize=9, framealpha=0.85)
    ax0.set_title(f"Patient {result.patient_id} — central B-scan {mid} with SVP (yellow fill) / DVP (cyan fill) slabs")
    ax0.set_xticks([])
    ax0.set_yticks([])

    panels = [
        ("Superficial (enhanced)", result.svp_display, f"density {result.svp_density_pct:.1f}%"),
        ("Deep (enhanced, shadow-corrected)", result.dvp_display, f"density {result.dvp_density_pct:.1f}%"),
    ]
    for i, (title, img, sub) in enumerate(panels):
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(img, cmap="gray")
        ax.set_title(title, fontsize=11)
        if sub:
            ax.set_xlabel(sub, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        f"OCTA Reconstruction — Patient {result.patient_id}  "
        f"(mean retinal thickness ≈ {np.mean(bounds.rpe - bounds.ilm):.0f} px)",
        fontsize=15,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_cohort_summary(entries: list, out_path: Path) -> None:
    """One row per patient: central B-scan with boundaries | SVP enhanced |
    DVP enhanced | SVP/DVP color overlay. A compact, at-a-glance comparison
    across the whole cohort, complementing each patient's full composite
    figure (which stays the place for per-case detail).

    Takes lightweight `CohortEntry` records (pipeline.py), not full
    `PatientResult`s -- holding all 5 patients' full structural/flow
    volumes in memory at once to build this one figure would be a
    several-gigabyte spike for no benefit, since only the central B-scan
    slice is actually drawn.
    """
    n = len(entries)
    fig, axes = plt.subplots(n, 4, figsize=(20, 5 * n))
    if n == 1:
        axes = axes[None, :]

    col_titles = ["Central B-scan", "Superficial (enhanced)", "Deep (enhanced)", "SVP / DVP overlay"]

    for row, entry in enumerate(entries):
        b = entry.mid_bscan
        mid = entry.mid_index
        bounds = entry.bounds
        x = np.arange(b.shape[1])

        ax = axes[row, 0]
        ax.imshow(b, cmap="gray", aspect="auto")
        ax.plot(x, bounds.ilm[mid], color=_BOUNDARY_STYLE["ILM"], lw=1.1)
        ax.plot(x, bounds.ipl_inl[mid], color=_BOUNDARY_STYLE["IPL/INL"], lw=1.1)
        ax.plot(x, bounds.opl_onl[mid], color=_BOUNDARY_STYLE["OPL/ONL"], lw=1.1)
        ax.plot(x, bounds.rpe[mid], color=_BOUNDARY_STYLE["RPE"], lw=1.1)
        qc_tag = "flagged" if bounds.qc["any_flagged"] else "clean"
        ax.set_ylabel(
            f"Patient {entry.patient_id}\nSVP {entry.svp_density_pct:.0f}% · "
            f"DVP {entry.dvp_density_pct:.0f}%\nQC: {qc_tag}",
            fontsize=11,
            rotation=0,
            ha="right",
            va="center",
            labelpad=15,
        )

        ax_svp = axes[row, 1]
        ax_svp.imshow(entry.svp_display, cmap="gray")

        ax_dvp = axes[row, 2]
        ax_dvp.imshow(entry.dvp_display, cmap="gray")

        h, w = entry.svp_display.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        rgb[..., 0] = entry.svp_display
        rgb[..., 1] = entry.dvp_display
        rgb[..., 2] = entry.svp_display
        ax_overlay = axes[row, 3]
        ax_overlay.imshow(rgb)

        for ax in (ax, ax_svp, ax_dvp, ax_overlay):
            ax.set_xticks([])
            ax.set_yticks([])
        if row == 0:
            for ax, title in zip(axes[row], col_titles):
                ax.set_title(title, fontsize=13)

    fig.suptitle("OCTA Reconstruction — Cohort Summary (5 patients)", fontsize=17, y=1.005)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def save_composite_color(svp: np.ndarray, dvp: np.ndarray, out_path: Path) -> None:
    """Two-color composite: SVP in magenta, DVP in green, to visualize how
    the two plexuses spatially relate (common in OCTA literature figures)."""
    h, w = svp.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[..., 0] = svp  # R
    rgb[..., 1] = dvp  # G
    rgb[..., 2] = svp  # B (svp -> magenta, dvp -> green, overlap -> white-ish)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(rgb)
    ax.set_title("SVP (magenta) vs DVP (green) composite")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
