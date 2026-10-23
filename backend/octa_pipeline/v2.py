"""v2 reconstruction: extra anatomical slabs (avascular complex, choriocapillaris,
choroid) and a contrast-restoring enhancement chain for every enface image.

Written as a separate module so the validated v1 outputs in results/ are not
touched -- v2 writes to its own results folder.

Slab definitions (pixel offsets are uncalibrated, see README):
    SVC   ILM -> IPL/INL                   (Spaide et al., 2015; Campbell et al., 2017)
    DVC   IPL/INL -> OPL/ONL               (Spaide et al., 2015)
    AVC   OPL/ONL -> RPE                   (avascular outer retina, Spaide 2015 / Campbell 2017)
    CC    RPE + cc_gap .. RPE + cc_depth   (choriocapillaris, a few tens of um below RPE)
    Choroid  RPE + choroid_gap onward      (larger choroidal vessels, Haller / Sattler layers)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from . import io_utils, postprocessing, projection, segmentation

logger = logging.getLogger("octa_pipeline")


@dataclass
class V2Config:
    # Choriocapillaris: a thin slab just beneath the RPE / Bruch's membrane.
    cc_gap_px: int = 2
    cc_depth_px: int = 14
    # Choroid: deeper stroma with larger vessels, beginning past the CC.
    choroid_gap_px: int = 16
    choroid_depth_px: int = 120
    # Background flattening: dividing by a large-scale local mean lifts dim
    # regions so they're not crushed to black by the global percentile stretch.
    background_sigma_px: float = 24.0
    # Ridge enhancement blended into the flattened image (kept moderate on
    # purpose; the earlier review flagged over-enhancement as making noise
    # look like capillaries).
    ridge_sigmas: tuple = (1.0, 1.5, 2.0)
    ridge_blend: float = 0.45
    clahe_clip: float = 2.0
    clahe_tile: tuple = (8, 8)
    # Target median brightness after the midtone lift (see _midtone_lift).
    display_target_median: float = 60.0
    robust_low_pct: float = 0.5
    robust_high_pct: float = 99.7


@dataclass
class ExtraSlabs:
    avc_top: np.ndarray
    avc_bottom: np.ndarray
    cc_top: np.ndarray
    cc_bottom: np.ndarray
    choroid_top: np.ndarray
    choroid_bottom: np.ndarray


def build_extra_slabs(bounds: segmentation.VolumeBoundaries, depth: int, cfg: V2Config) -> ExtraSlabs:
    avc_top = bounds.opl_onl
    avc_bottom = bounds.rpe - 1
    cc_top = bounds.rpe + cfg.cc_gap_px
    cc_bottom = cc_top + cfg.cc_depth_px
    choroid_top = bounds.rpe + cfg.choroid_gap_px
    choroid_bottom = choroid_top + cfg.choroid_depth_px
    clip = lambda a: np.clip(a, 0, depth)
    return ExtraSlabs(
        avc_top=clip(avc_top), avc_bottom=clip(avc_bottom),
        cc_top=clip(cc_top), cc_bottom=clip(cc_bottom),
        choroid_top=clip(choroid_top), choroid_bottom=clip(choroid_bottom),
    )


def normalize_robust(img: np.ndarray, cfg: V2Config) -> np.ndarray:
    lo, hi = np.percentile(img, [cfg.robust_low_pct, cfg.robust_high_pct])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((img - lo) / (hi - lo), 0.0, 1.0)


def enhance_v2(img: np.ndarray, cfg: V2Config) -> np.ndarray:
    """img: float enface, any scale. Returns uint8 display image.

    1. Background flattening: divide by a large-scale local mean so dim but
       real texture in low-signal regions is lifted, rather than clipped.
    2. Robust percentile normalization to [0, 1].
    3. Multiscale ridge (Frangi) response blended in at a moderate weight.
    4. CLAHE for local contrast on the 8-bit result.
    """
    img = img.astype(np.float64)
    background = gaussian_filter(img, cfg.background_sigma_px)
    flat = img / (background + np.percentile(background, 5) + 1e-6)
    norm = normalize_robust(flat, cfg)

    from skimage.filters import frangi

    ridge = frangi(norm, sigmas=cfg.ridge_sigmas, black_ridges=False)
    ridge = normalize_robust(ridge, cfg)
    blended = (1 - cfg.ridge_blend) * norm + cfg.ridge_blend * ridge
    u8 = np.clip(blended * 255.0, 0, 255).astype(np.uint8)

    import cv2

    clahe = cv2.createCLAHE(clipLimit=cfg.clahe_clip, tileGridSize=cfg.clahe_tile)
    out = clahe.apply(u8)
    return _midtone_lift(out, cfg.display_target_median)


def _midtone_lift(img_u8: np.ndarray, target_median: float) -> np.ndarray:
    """Adaptive gamma that moves this image's median to target_median, never darkening.
    Same technique as postprocessing.apply_display_gamma, applied here so the
    darkest slabs (DVC, choroid) keep visible background texture."""
    med = max(float(np.median(img_u8)), 1.0)
    gamma = np.log(target_median / 255.0) / np.log(med / 255.0)
    gamma = float(np.clip(gamma, 0.3, 1.0))
    lifted = 255.0 * (img_u8.astype(np.float64) / 255.0) ** gamma
    return np.clip(lifted, 0, 255).astype(np.uint8)


def vessel_contrast(img_u8: np.ndarray) -> float:
    p50, p99 = np.percentile(img_u8.astype(float), [50, 99])
    return float((p99 - p50) / (p99 + p50 + 1e-6))


def run_patient_v2(data_dir, patient_id: int, cfg, out_dir, v2: V2Config) -> dict:
    out = Path(out_dir) / f"patient_{patient_id}"
    out.mkdir(parents=True, exist_ok=True)

    pdata = io_utils.load_patient(data_dir, patient_id)
    bounds = segmentation.segment_volume(pdata.structural, cfg.segmentation)
    slabs = projection.build_slabs(bounds, cfg.segmentation)
    depth = pdata.flow.shape[1]
    extra = build_extra_slabs(bounds, depth, v2)

    flow_ds = projection.despeckle_flow(pdata.flow, cfg.projection.flow_median_ksize)
    bulk_floor = projection.estimate_bulk_motion_floor(
        flow_ds, bounds.opl_onl, bounds.rpe, cfg.projection.bulk_motion_ref_margin
    )
    flow_c = projection.suppress_bulk_motion(
        flow_ds, bulk_floor, cfg.projection.bulk_motion_baseline_percentile
    )

    svc_f = projection.project_slab(flow_c, slabs.svp_top, slabs.svp_bottom, cfg.projection)
    dvc_f = projection.suppress_projection_artifact_3d(
        flow_c, slabs.svp_top, slabs.svp_bottom, slabs.dvp_top, slabs.dvp_bottom,
        cfg.projection.projection_artifact_suppression,
        cfg.projection.projection_artifact_decay_px,
    )
    avc_f = projection.project_slab(flow_c, extra.avc_top, extra.avc_bottom, cfg.projection)
    cc_f = projection.project_slab(flow_c, extra.cc_top, extra.cc_bottom, cfg.projection)
    ch_f = projection.project_slab(flow_c, extra.choroid_top, extra.choroid_bottom, cfg.projection)

    results = {}
    for name, f in [("svc", svc_f), ("dvc", dvc_f), ("avc", avc_f), ("cc", cc_f), ("choroid", ch_f)]:
        enh = enhance_v2(f, v2)
        Image.fromarray(enh).save(out / f"{name}_enface_v2.png")
        results[name] = {
            "contrast": round(vessel_contrast(enh), 3),
            "median": int(np.median(enh)),
        }

    metrics = {"patient_id": patient_id, "enface_v2": results,
               "segmentation_qc": bounds.qc,
               "mean_retinal_thickness_px": round(float(np.mean(bounds.rpe - bounds.ilm)), 1)}
    with open(out / "metrics_v2.json", "w") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info("[patient %d] v2 done -> %s", patient_id, out)
    return metrics


def save_cohort_v2(out_dir: Path, patient_ids) -> None:
    rows = []
    for pid in patient_ids:
        d = Path(out_dir) / f"patient_{pid}"
        rows.append((pid, [np.array(Image.open(d / f"{n}_enface_v2.png")) for n in ["svc", "dvc", "cc", "choroid", "avc"]]))
    titles = ["Superficial (SVC)", "Deep (DVC)", "Choriocapillaris", "Choroid", "Avascular complex"]
    fig, axes = plt.subplots(len(rows), 5, figsize=(25, 5 * len(rows)))
    for r, (pid, imgs) in enumerate(rows):
        for c, im in enumerate(imgs):
            axes[r, c].imshow(im, cmap="gray")
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(titles[c], fontsize=13)
        axes[r, 0].set_ylabel(f"Patient {pid}", fontsize=12, rotation=0, ha="right", va="center", labelpad=15)
    fig.suptitle("OCTA v2 reconstruction — SVC / DVC / choriocapillaris / choroid / avascular complex", fontsize=16)
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "cohort_v2.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
