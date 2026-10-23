"""v3 with two changes, no inpainting and no masking:

  1. Avascular complex uses a median projection over its slab instead of a
     mean. A few bright speckles inside the slab then no longer raise the
     projected value, so the slab stays sparse. Nothing is filled or removed
     spatially; every pixel is still the measured signal of its A-scan.
  2. Choriocapillaris and choroid keep the v3 row flat-fielding, which
     removes B-scan-to-B-scan gain bands. Shadowed regions below
     hyperreflective tissue are left as measured (dark), and their fraction
     is reported as a QC number instead of being painted over.

Writes to results_v3_fixed/. results/, results_v2/ and results_v3/ are untouched.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from . import io_utils, projection, segmentation, v2, v3

logger = logging.getLogger("octa_pipeline")


def median_projection(volume: np.ndarray, top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    gathered, valid, _ = projection._gather_slab(volume, top, bottom)
    g = np.where(valid, gathered, np.nan)
    return np.nan_to_num(np.nanmedian(g, axis=1), nan=0.0)


def shadow_fraction(structural: np.ndarray, top: np.ndarray, bottom: np.ndarray) -> float:
    """QC only: fraction of choroid-window A-scans that look shadowed. Not used to modify the image."""
    ch = projection._mean_projection(structural.astype(np.float32), top, bottom)
    raw = ch < 0.8 * np.median(ch)
    return float(ndi.binary_opening(raw, iterations=2).mean())


def run_patient_v3_fixed(data_dir, patient_id: int, cfg, out_dir, v2cfg) -> dict:
    out = Path(out_dir) / f"patient_{patient_id}"
    out.mkdir(parents=True, exist_ok=True)

    pdata = io_utils.load_patient(data_dir, patient_id)
    bounds = segmentation.segment_volume(pdata.structural, cfg.segmentation)
    slabs = projection.build_slabs(bounds, cfg.segmentation)
    extra = v2.build_extra_slabs(bounds, pdata.flow.shape[1], v2cfg)

    flow_ds = projection.despeckle_flow(pdata.flow, cfg.projection.flow_median_ksize)
    bulk = projection.estimate_bulk_motion_floor(
        flow_ds, bounds.opl_onl, bounds.rpe, cfg.projection.bulk_motion_ref_margin
    )
    flow_c = projection.suppress_bulk_motion(flow_ds, bulk, cfg.projection.bulk_motion_baseline_percentile)

    svc_f = projection.project_slab(flow_c, slabs.svp_top, slabs.svp_bottom, cfg.projection)
    dvc_f = projection.suppress_projection_artifact_3d(
        flow_c, slabs.svp_top, slabs.svp_bottom, slabs.dvp_top, slabs.dvp_bottom,
        cfg.projection.projection_artifact_suppression, cfg.projection.projection_artifact_decay_px,
    )
    avc_f = v3.avascular_void(median_projection(flow_c, extra.avc_top, extra.avc_bottom))
    cc_f = projection._mean_projection(flow_c, extra.cc_top, extra.cc_bottom)
    ch_f = projection._mean_projection(flow_c, extra.choroid_top, extra.choroid_bottom)

    layers = {
        "svc": v3.row_flat_field(svc_f),
        "dvc": v3.row_flat_field(dvc_f),
        "avc": v3.row_flat_field(avc_f),
        "cc": v3.row_flat_field(cc_f),
        "choroid": v3.row_flat_field(ch_f),
    }
    results = {}
    for name, f in layers.items():
        enh = v2.enhance_v2(f, v2cfg)
        Image.fromarray(enh).save(out / f"{name}_enface_fixed.png")
        results[name] = {
            "median": int(np.median(enh)),
            "contrast": round(v2.vessel_contrast(enh), 3),
            "banding_index": round(v3.banding_index(enh), 3),
        }
    results["choroid_shadow_fraction_qc"] = round(
        shadow_fraction(pdata.structural, extra.choroid_top, extra.choroid_bottom), 4
    )
    metrics = {"patient_id": patient_id, "enface_fixed": results, "segmentation_qc": bounds.qc}
    with open(out / "metrics_fixed.json", "w") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info("[patient %d] fixed v3 done -> %s", patient_id, out)
    return metrics


def save_cohort_fixed(out_dir: Path, pids) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ["svc", "dvc", "avc", "cc", "choroid"]
    titles = ["Superficial (SVC)", "Deep (DVC)", "Avascular", "Choriocapillaris", "Choroid"]
    fig, axes = plt.subplots(len(pids), 5, figsize=(25, 5 * len(pids)), squeeze=False)
    for r, pid in enumerate(pids):
        for c, n in enumerate(names):
            im = np.array(Image.open(Path(out_dir) / f"patient_{pid}" / f"{n}_enface_fixed.png"))
            axes[r, c].imshow(im, cmap="gray", vmin=0, vmax=255)
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(titles[c], fontsize=13)
        axes[r, 0].set_ylabel(f"Patient {pid}", fontsize=12, rotation=0, ha="right", va="center", labelpad=15)
    fig.suptitle("OCTA v3-fixed: median avascular projection, no inpainting", fontsize=16, y=1.0)
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "cohort_fixed.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
