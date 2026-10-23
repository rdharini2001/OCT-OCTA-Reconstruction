"""v4: v3 with three targeted changes. SVC and DVC are exactly v3.

  1. Avascular complex is rendered as a void. Its image is normalised by its
     own high percentile (no CLAHE, no midtone lift) so the background stays
     black, and a 3x3 median filter removes isolated speckles while keeping
     any connected structure. The v3-fixed version lifted this slab to grey,
     which was wrong for a slab that should be avascular.
  2. Choroid and choriocapillaris keep v3's per-row flat-fielding. A smoothed
     row gain was tested and rejected: on Patient 4 it raised banding from
     0.14 to 0.40.
  3. Choroid shadow voids are left as measured (no fill, no inpainting).

Writes to results_v4/. results/, results_v2/, results_v3/ and results_v3_fixed/
are not touched.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from . import io_utils, projection, segmentation, v2, v3, v3_fixed

logger = logging.getLogger("octa_pipeline")


def smoothed_row_flat_field(img: np.ndarray, row_window: int = 5) -> np.ndarray:
    """Row flat-fielding with the per-row gain median-smoothed across neighbouring
    B-scans, so one bad B-scan cannot create a full-width streak by itself."""
    img = img.astype(np.float64)
    row_level = np.median(img, axis=1)
    row_level = ndi.median_filter(row_level, size=row_window, mode="nearest")
    row_level = np.maximum(row_level, np.percentile(row_level, 10) + 1e-6)
    gain = row_level / np.median(row_level)
    return img / gain[:, None]


def normalize_void(f: np.ndarray) -> np.ndarray:
    """Avascular display: percentile normalise, 3x3 median speckle removal, no lift."""
    lo = 0.0
    hi = np.percentile(f, 99.7) + 1e-6
    scaled = np.clip((f - lo) / (hi - lo), 0, 1)
    despeckled = ndi.median_filter(scaled, size=3, mode="nearest")
    return np.clip(despeckled * 255.0, 0, 255).astype(np.uint8)


def run_patient_v4(data_dir, patient_id: int, cfg, out_dir, v2cfg) -> dict:
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
    avc_f = v3.avascular_void(v3_fixed.median_projection(flow_c, extra.avc_top, extra.avc_bottom))
    cc_f = projection._mean_projection(flow_c, extra.cc_top, extra.cc_bottom)
    ch_f = projection._mean_projection(flow_c, extra.choroid_top, extra.choroid_bottom)

    results = {}
    for name, f in [("svc", svc_f), ("dvc", dvc_f)]:
        enh = v2.enhance_v2(v3.row_flat_field(f), v2cfg)
        results[name] = enh
    results["avc"] = normalize_void(smoothed_row_flat_field(avc_f))
    results["cc"] = v2.enhance_v2(v3.row_flat_field(cc_f), v2cfg)
    results["choroid"] = v2.enhance_v2(v3.row_flat_field(ch_f), v2cfg)

    metrics = {"patient_id": patient_id, "enface_v4": {}}
    for name, enh in results.items():
        Image.fromarray(enh).save(out / f"{name}_enface_v4.png")
        metrics["enface_v4"][name] = {
            "median": int(np.median(enh)),
            "banding_index": round(v3.banding_index(enh), 3),
        }
    metrics["enface_v4"]["choroid_shadow_fraction_qc"] = round(
        v3_fixed.shadow_fraction(pdata.structural, extra.choroid_top, extra.choroid_bottom), 4
    )
    metrics["segmentation_qc"] = bounds.qc
    with open(out / "metrics_v4.json", "w") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info("[patient %d] v4 done -> %s", patient_id, out)
    return metrics


def save_cohort_v4(out_dir: Path, pids) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ["svc", "dvc", "avc", "cc", "choroid"]
    titles = ["Superficial (SVC)", "Deep (DVC)", "Avascular", "Choriocapillaris", "Choroid"]
    fig, axes = plt.subplots(len(pids), 5, figsize=(25, 5 * len(pids)), squeeze=False)
    for r, pid in enumerate(pids):
        for c, n in enumerate(names):
            im = np.array(Image.open(Path(out_dir) / f"patient_{pid}" / f"{n}_enface_v4.png"))
            axes[r, c].imshow(im, cmap="gray", vmin=0, vmax=255)
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(titles[c], fontsize=13)
        axes[r, 0].set_ylabel(f"Patient {pid}", fontsize=12, rotation=0, ha="right", va="center", labelpad=15)
    fig.suptitle("OCTA v4 (final candidate)", fontsize=16, y=1.0)
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "cohort_v4.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
