"""v3 reconstruction: row-wise flat-fielding for banding, mean projection for
diffuse deep slabs (CC, choroid), and an explicit void constraint for the
avascular complex.

Writes to its own results folder; results/ and results_v2/ are untouched.

Rationale per slab (references checked against OCTA literature that uses
the same slab names, e.g. Spaide et al. 2015/2018, Campbell et al. 2017):
  SVC / DVC     max projection (vessels are sparse bright features); the 3D
                DVC shadow suppression from v1 is kept.
  Avascular     should be void of perfusion by definition, so any signal
                there is noise or projection tail. Mean projection plus a
                noise-floor subtraction is used so the slab stays sparse.
  Choriocapillaris, Choroid   diffuse, fine-textured perfusion. Mean
                projection, plus row-wise flat-fielding to remove the
                horizontal bands that come from B-scan-to-B-scan gain.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from . import io_utils, projection, segmentation, v2

logger = logging.getLogger("octa_pipeline")


def row_flat_field(img: np.ndarray) -> np.ndarray:
    """Remove B-scan-to-B-scan (row) gain differences.

    Each enface row is one B-scan. Dividing each row by its own median and
    rescaling to the global median keeps spatial vessel structure while
    removing the horizontal banding that appears in the deep slabs.
    """
    img = img.astype(np.float64)
    row_level = np.median(img, axis=1, keepdims=True)
    # Floor the per-row estimate: near-empty B-scans would otherwise divide by
    # a near-zero median and amplify noise into bright rows.
    row_level = np.maximum(row_level, np.percentile(row_level, 10) + 1e-6)
    global_level = np.median(row_level)
    return img / row_level * global_level


def banding_index(img: np.ndarray) -> float:
    """std of per-row medians relative to the image mean. Lower = less banding."""
    rows = np.median(img.astype(float), axis=1)
    return float(rows.std() / (img.mean() + 1e-6))


def avascular_void(avc: np.ndarray, floor_pct: float = 90.0) -> np.ndarray:
    """Enforce the avascular void: subtract a noise floor estimated from the
    avascular slab's own distribution, then clip. Residual specks above this
    floor are treated as tail/noise and removed."""
    floor = np.percentile(avc, floor_pct)
    return np.clip(avc - floor, 0, None)


def run_patient_v3(data_dir, patient_id: int, cfg, out_dir, v2cfg) -> dict:
    from .projection import _mean_projection

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
    avc_f = _mean_projection(flow_c, extra.avc_top, extra.avc_bottom)
    cc_f = _mean_projection(flow_c, extra.cc_top, extra.cc_bottom)
    ch_f = _mean_projection(flow_c, extra.choroid_top, extra.choroid_bottom)

    avc_f = avascular_void(avc_f)

    layers = {
        "svc": row_flat_field(svc_f),
        "dvc": row_flat_field(dvc_f),
        "avc": avc_f,
        "cc": row_flat_field(cc_f),
        "choroid": row_flat_field(ch_f),
    }
    results = {}
    for name, f in layers.items():
        enh = v2.enhance_v2(f, v2cfg)
        Image.fromarray(enh).save(out / f"{name}_enface_v3.png")
        results[name] = {
            "median": int(np.median(enh)),
            "contrast": round(v2.vessel_contrast(enh), 3),
            "banding_index": round(banding_index(enh), 3),
            "mean_raw": round(float(np.mean(f)), 3),
        }

    results["avc_to_svc_raw_ratio"] = round(float(np.mean(avc_f) / (np.mean(svc_f) + 1e-6)), 4)
    metrics = {"patient_id": patient_id, "enface_v3": results, "segmentation_qc": bounds.qc}
    with open(out / "metrics_v3.json", "w") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info("[patient %d] v3 done -> %s", patient_id, out)
    return metrics


def save_cohort_v3(out_dir: Path, patient_ids) -> None:
    rows = []
    for pid in patient_ids:
        d = Path(out_dir) / f"patient_{pid}"
        rows.append((pid, [np.array(Image.open(d / f"{n}_enface_v3.png")) for n in ["svc", "dvc", "avc", "cc", "choroid"]]))
    titles = ["Superficial (SVC)", "Deep (DVC)", "Avascular", "Choriocapillaris", "Choroid"]
    fig, axes = plt.subplots(len(rows), 5, figsize=(25, 5 * len(rows)))
    for r, (pid, imgs) in enumerate(rows):
        for c, im in enumerate(imgs):
            axes[r, c].imshow(im, cmap="gray")
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(titles[c], fontsize=13)
        axes[r, 0].set_ylabel(f"Patient {pid}", fontsize=12, rotation=0, ha="right", va="center", labelpad=15)
    fig.suptitle("OCTA v3 reconstruction", fontsize=16, y=1.0)
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "cohort_v3.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
