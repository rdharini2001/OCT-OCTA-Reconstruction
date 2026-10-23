"""Polished layer-segmentation figures for all 5 patients: ILM/IPL-INL/OPL-ONL/EZC/RPE
on three B-scan positions each, with shaded SVC/DVC/AVC slabs, clean typography, and a
consistent high-end look. Writes results_v6/patient_<N>/layers_beautiful_v6.png and a
per-patient single-B-scan hero image, plus a cohort montage.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from octa_pipeline import io_utils, segmentation, ezc
from octa_pipeline.config import PipelineConfig

plt.rcParams.update({
    "figure.facecolor": "#0b0f12",
    "axes.facecolor": "#0b0f12",
    "savefig.facecolor": "#0b0f12",
    "text.color": "#e8edf0",
    "axes.edgecolor": "#2a3338",
})

COLORS = {
    "ILM": "#ff5a6e",
    "IPL/INL": "#ffd166",
    "OPL/ONL": "#4fd1c5",
    "EZC": "#c792ea",
    "RPE": "#6ee7a0",
}
FILL = {
    "SVC": "#ffb703",   # ILM -> IPL/INL
    "DVC": "#4fd1c5",   # IPL/INL -> OPL/ONL
    "AVC": "#7aa2ff",   # OPL/ONL -> RPE-ish (avascular outer retina)
}

OUT = Path("results_v6")
cfg = PipelineConfig()


def draw_bscan(ax, b, x, bounds_y, title, show_legend=False):
    ax.imshow(b, cmap="bone", aspect="auto")
    ilm, ipl, opl, ez, rpe = bounds_y
    ax.fill_between(x, ilm, ipl, color=FILL["SVC"], alpha=0.16, lw=0)
    ax.fill_between(x, ipl, opl, color=FILL["DVC"], alpha=0.16, lw=0)
    ax.fill_between(x, opl, rpe, color=FILL["AVC"], alpha=0.12, lw=0)
    for name, y, lw in [("ILM", ilm, 1.6), ("IPL/INL", ipl, 1.6),
                        ("OPL/ONL", opl, 1.6), ("EZC", ez, 1.3), ("RPE", rpe, 1.6)]:
        ax.plot(x, y, color=COLORS[name], lw=lw, solid_capstyle="round",
                path_effects=None, label=name, alpha=0.95)
    ax.set_title(title, fontsize=12, color="#e8edf0", pad=8)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if show_legend:
        leg = ax.legend(loc="upper right", fontsize=9, framealpha=0.25,
                        facecolor="#141a1e", edgecolor="none", labelcolor="#e8edf0")


def main():
    cohort_fig, cohort_axes = plt.subplots(1, 5, figsize=(27, 6.2))
    cohort_fig.suptitle("Retinal layer segmentation -- all patients (central B-scan)",
                        fontsize=18, color="#f3f6f7", y=1.02)

    for i, pid in enumerate(range(1, 6)):
        d = OUT / f"patient_{pid}"
        pdata = io_utils.load_patient(".", pid)
        bounds = segmentation.segment_volume(pdata.structural, cfg.segmentation)
        e = ezc.detect_ezc_volume(pdata.structural, bounds, cfg.segmentation)

        n_bscans, width = bounds.ilm.shape
        x = np.arange(width)
        rows_idx = [n_bscans // 4, n_bscans // 2, 3 * n_bscans // 4]

        fig, axes = plt.subplots(1, 3, figsize=(19, 6.4))
        fig.suptitle(f"Patient {pid} -- OCT B-scan layer segmentation", fontsize=15, color="#f3f6f7", y=1.03)
        for j, (ax, y) in enumerate(zip(axes, rows_idx)):
            b = pdata.structural[y]
            draw_bscan(ax, b, x,
                      (bounds.ilm[y], bounds.ipl_inl[y], bounds.opl_onl[y], e[y], bounds.rpe[y]),
                      f"B-scan {y} / {n_bscans}", show_legend=(j == 0))
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(d / "layers_beautiful_v6.png", dpi=160, facecolor=fig.get_facecolor())
        plt.close(fig)

        # single hero image: central B-scan only, larger
        mid = n_bscans // 2
        hero, hax = plt.subplots(figsize=(10, 7))
        draw_bscan(hax, pdata.structural[mid], x,
                  (bounds.ilm[mid], bounds.ipl_inl[mid], bounds.opl_onl[mid], e[mid], bounds.rpe[mid]),
                  f"Patient {pid} -- central B-scan", show_legend=True)
        hero.tight_layout()
        hero.savefig(d / "layers_hero_v6.png", dpi=170, facecolor=hero.get_facecolor())
        plt.close(hero)

        ax = cohort_axes[i]
        draw_bscan(ax, pdata.structural[mid], x,
                  (bounds.ilm[mid], bounds.ipl_inl[mid], bounds.opl_onl[mid], e[mid], bounds.rpe[mid]),
                  f"Patient {pid}", show_legend=(i == 0))
        print(f"patient {pid} done")

    cohort_fig.tight_layout(rect=[0, 0, 1, 0.93])
    cohort_fig.savefig(OUT / "layers_beautiful_cohort_v6.png", dpi=150, facecolor=cohort_fig.get_facecolor())
    plt.close(cohort_fig)
    print("cohort montage saved")


if __name__ == "__main__":
    main()
