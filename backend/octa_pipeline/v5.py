"""v5: final reconstruction method.

Processing per layer (see results_v5/METHOD.md for the full write-up):

  SVC        v3 chain: despeckle -> bulk-motion attenuation -> max projection of
             ILM..IPL/INL -> row flat-field -> v2 enhancement (background flatten,
             Frangi blend, CLAHE, midtone lift to median 60).
  DVC        as SVC, with the depth-resolved 3D shadow suppression (v1 method,
             decay anchored to the DVC slab top) in place of plain max projection.
  Avascular  median projection over OPL/ONL..RPE, noise-floor subtraction, then
             percentile normalisation and a 3x3 median filter. No midtone lift,
             no CLAHE, no fill: the slab stays black unless signal is present.
  CC, choroid  mean projection of their slabs -> v3 row flat-field -> v2
             enhancement.

Rejected in development (numbers from the Patient 4/5 choroid tests):
  smoothed row gain, 5-B-scan median  banding 0.40 (worse than 0.14)
  wide row gain, 31 and 61 B-scans     banding 0.51 (no better than raw)
  ONL-reference row gain               banding 0.67 / 0.55 (worse; also double-corrects)
  grey lift on avascular slab          avascular read as tissue, not void
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


def avascular_display(f: np.ndarray) -> np.ndarray:
    lo = 0.0
    hi = np.percentile(f, 99.7) + 1e-6
    scaled = np.clip((f - lo) / (hi - lo), 0, 1)
    despeckled = ndi.median_filter(scaled, size=3, mode="nearest")
    return np.clip(despeckled * 255.0, 0, 255).astype(np.uint8)


def run_patient_v5(data_dir, patient_id: int, cfg, out_dir, v2cfg) -> dict:
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

    images = {
        "svc": v2.enhance_v2(v3.row_flat_field(svc_f), v2cfg),
        "dvc": v2.enhance_v2(v3.row_flat_field(dvc_f), v2cfg),
        "avc": avascular_display(avc_f),
        "cc": v2.enhance_v2(v3.row_flat_field(cc_f), v2cfg),
        "choroid": v2.enhance_v2(v3.row_flat_field(ch_f), v2cfg),
    }

    metrics = {"patient_id": patient_id, "method": "v5", "enface": {}}
    for name, img in images.items():
        Image.fromarray(img).save(out / f"{name}_enface_v5.png")
        metrics["enface"][name] = {
            "median": int(np.median(img)),
            "contrast": round(v2.vessel_contrast(img), 3),
            "banding_index": round(v3.banding_index(img), 3),
        }
    metrics["enface"]["choroid_shadow_fraction_qc"] = round(
        v3_fixed.shadow_fraction(pdata.structural, extra.choroid_top, extra.choroid_bottom), 4
    )
    metrics["segmentation_qc"] = bounds.qc
    with open(out / "metrics_v5.json", "w") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info("[patient %d] v5 done -> %s", patient_id, out)
    return metrics


def save_cohort_v5(out_dir: Path, pids) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ["svc", "dvc", "avc", "cc", "choroid"]
    titles = ["Superficial (SVC)", "Deep (DVC)", "Avascular", "Choriocapillaris", "Choroid"]
    fig, axes = plt.subplots(len(pids), 5, figsize=(25, 5 * len(pids)), squeeze=False)
    for r, pid in enumerate(pids):
        for c, n in enumerate(names):
            im = np.array(Image.open(Path(out_dir) / f"patient_{pid}" / f"{n}_enface_v5.png"))
            axes[r, c].imshow(im, cmap="gray", vmin=0, vmax=255)
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(titles[c], fontsize=13)
        axes[r, 0].set_ylabel(f"Patient {pid}", fontsize=12, rotation=0, ha="right", va="center", labelpad=15)
    fig.suptitle("OCTA v5 (final method)", fontsize=16, y=1.0)
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "cohort_v5.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
