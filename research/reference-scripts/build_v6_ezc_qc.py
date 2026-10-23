"""v6: adds the EZC boundary to the QC overlay (new, informational only -- no
reconstruction output changes, since OPL/ONL and AVC tests found no usable
improvement; see results_v6/METHOD.md). Writes per-patient QC overlays with
EZC and a cohort EZC-reliability figure.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from octa_pipeline import io_utils, segmentation, ezc
from octa_pipeline.config import PipelineConfig

STYLE = {"ILM": "#e63946", "IPL/INL": "#ffd60a", "OPL/ONL": "#00b4d8", "EZC": "#ff6fd8", "RPE": "#57cc99"}
OUT = Path("results_v6")
cfg = PipelineConfig()

jump_rows = []
fig_cohort, axes_cohort = plt.subplots(1, 5, figsize=(30, 6))

for i, pid in enumerate(range(1, 6)):
    d = OUT / f"patient_{pid}"
    pdata = io_utils.load_patient(".", pid)
    bounds = segmentation.segment_volume(pdata.structural, cfg.segmentation)
    e = ezc.detect_ezc_volume(pdata.structural, bounds, cfg.segmentation)
    jump = float(np.abs(np.diff(e, axis=0)).max())
    jump_rows.append((pid, jump))

    n_bscans = pdata.structural.shape[0]
    rows_idx = [n_bscans // 4, n_bscans // 2, 3 * n_bscans // 4]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, y in zip(axes, rows_idx):
        b = pdata.structural[y]
        ax.imshow(b, cmap="gray", aspect="auto")
        x = np.arange(b.shape[1])
        ax.plot(x, bounds.ilm[y], color=STYLE["ILM"], lw=1.2, label="ILM")
        ax.plot(x, bounds.ipl_inl[y], color=STYLE["IPL/INL"], lw=1.2, label="IPL/INL")
        ax.plot(x, bounds.opl_onl[y], color=STYLE["OPL/ONL"], lw=1.2, label="OPL/ONL (unchanged, fraction est.)")
        ax.plot(x, e[y], color=STYLE["EZC"], lw=1.4, label="EZC (new)")
        ax.plot(x, bounds.rpe[y], color=STYLE["RPE"], lw=1.2, label="RPE")
        ax.set_title(f"B-scan {y}/{n_bscans}")
        ax.set_xticks([]); ax.set_yticks([])
    axes[0].legend(loc="upper right", fontsize=8, framealpha=0.85)
    fig.suptitle(f"Patient {pid} -- v6: ILM/IPL-INL/OPL-ONL/EZC/RPE", fontsize=13)
    fig.tight_layout()
    fig.savefig(d / "qc_boundary_overlay_v6.png", dpi=140)
    plt.close(fig)

    mid = n_bscans // 2
    b = pdata.structural[mid]
    ax = axes_cohort[i]
    ax.imshow(b, cmap="gray", aspect="auto")
    x = np.arange(b.shape[1])
    ax.plot(x, bounds.ilm[mid], color=STYLE["ILM"], lw=1)
    ax.plot(x, e[mid], color=STYLE["EZC"], lw=1.3, label="EZC")
    ax.plot(x, bounds.rpe[mid], color=STYLE["RPE"], lw=1, label="RPE")
    ax.set_ylim(bounds.rpe[mid].max() + 100, max(bounds.ilm[mid].min() - 100, 0))
    ax.set_title(f"P{pid}: max frame-jump {jump:.0f}px")
    ax.set_xticks([]); ax.set_yticks([])

axes_cohort[0].legend(fontsize=9)
fig_cohort.suptitle("v6: EZC boundary reliability across the cohort (central B-scan)", fontsize=15)
fig_cohort.tight_layout()
fig_cohort.savefig(OUT / "ezc_reliability_cohort.png", dpi=110)
plt.close(fig_cohort)

print("jump summary:", jump_rows)
