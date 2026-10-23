"""End-to-end per-patient OCTA reconstruction pipeline: load -> segment ->
project -> post-process -> save results + visualizations.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from . import io_utils, postprocessing, projection, segmentation, visualization
from .config import PipelineConfig

logger = logging.getLogger("octa_pipeline")


@dataclass
class CohortEntry:
    """Lightweight per-patient summary for the cross-patient cohort figure
    (visualization.save_cohort_summary) -- deliberately excludes the full
    structural/flow volumes a PatientResult carries, so building the cohort
    figure doesn't require holding every patient's full data in memory at
    once.
    """

    patient_id: int
    mid_index: int
    mid_bscan: np.ndarray
    bounds: segmentation.VolumeBoundaries
    svp_enhanced: np.ndarray
    dvp_enhanced: np.ndarray
    svp_vessel_enhanced: np.ndarray
    dvp_vessel_enhanced: np.ndarray
    svp_display: np.ndarray
    dvp_display: np.ndarray
    svp_density_pct: float
    dvp_density_pct: float


def to_cohort_entry(result: "PatientResult") -> CohortEntry:
    mid = result.structural.shape[0] // 2
    return CohortEntry(
        patient_id=result.patient_id,
        mid_index=mid,
        mid_bscan=result.structural[mid],
        bounds=result.bounds,
        svp_enhanced=result.svp_enhanced,
        dvp_enhanced=result.dvp_enhanced,
        svp_vessel_enhanced=result.svp_vessel_enhanced,
        dvp_vessel_enhanced=result.dvp_vessel_enhanced,
        svp_display=result.svp_display,
        dvp_display=result.dvp_display,
        svp_density_pct=result.svp_density_pct,
        dvp_density_pct=result.dvp_density_pct,
    )


@dataclass
class PatientResult:
    patient_id: int
    svp_raw: np.ndarray
    svp_enhanced: np.ndarray
    svp_vessel_enhanced: np.ndarray
    svp_display: np.ndarray
    dvp_raw: np.ndarray
    dvp_enhanced: np.ndarray
    dvp_vessel_enhanced: np.ndarray
    dvp_display: np.ndarray
    svp_density_pct: float
    dvp_density_pct: float
    bounds: segmentation.VolumeBoundaries
    slabs: projection.Slabs
    structural: np.ndarray
    flow_despeckled: np.ndarray
    elapsed_s: float


def run_patient(
    data_dir: str | Path,
    patient_id: int,
    cfg: PipelineConfig,
    out_dir: str | Path,
) -> PatientResult:
    t0 = time.time()
    out_dir = Path(out_dir) / f"patient_{patient_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("[patient %d] loading volumes", patient_id)
    pdata = io_utils.load_patient(data_dir, patient_id)

    logger.info("[patient %d] segmenting retinal layers", patient_id)
    bounds = segmentation.segment_volume(pdata.structural, cfg.segmentation)
    slabs = projection.build_slabs(bounds, cfg.segmentation)

    logger.info("[patient %d] despeckling + projecting flow slabs", patient_id)
    flow_ds = projection.despeckle_flow(pdata.flow, cfg.projection.flow_median_ksize)

    bulk_motion_floor = projection.estimate_bulk_motion_floor(
        flow_ds, bounds.opl_onl, bounds.rpe, cfg.projection.bulk_motion_ref_margin
    )
    flow_corrected = projection.suppress_bulk_motion(
        flow_ds, bulk_motion_floor, cfg.projection.bulk_motion_baseline_percentile
    )

    svp_raw_f = projection.project_slab(flow_corrected, slabs.svp_top, slabs.svp_bottom, cfg.projection)
    svp_raw_u8 = postprocessing.normalize_to_uint8(svp_raw_f)

    # Depth-resolved (3D) superficial-shadow suppression, done on the flow
    # volume before projecting -- see projection.suppress_projection_artifact_3d
    # for why this replaced an earlier enface-level (2D) version.
    dvp_corrected_f = projection.suppress_projection_artifact_3d(
        flow_corrected,
        slabs.svp_top, slabs.svp_bottom,
        slabs.dvp_top, slabs.dvp_bottom,
        cfg.projection.projection_artifact_suppression,
        cfg.projection.projection_artifact_decay_px,
    )
    dvp_corrected_u8 = postprocessing.normalize_to_uint8(dvp_corrected_f)

    svp_enhanced = postprocessing.enhance_clahe(svp_raw_u8, cfg.visualization)
    dvp_enhanced = postprocessing.enhance_clahe(dvp_corrected_u8, cfg.visualization)

    svp_vessel_enhanced = postprocessing.enhance_vesselness(svp_enhanced, cfg.visualization)
    dvp_vessel_enhanced = postprocessing.enhance_vesselness(dvp_enhanced, cfg.visualization)

    # Display-only brightness calibration (see postprocessing.apply_display_gamma).
    # Deliberately computed from -- and saved separately from -- the file
    # used for vessel-density thresholding just above and the file already
    # fed to the external validation model, so neither of those numbers
    # changes; this only affects what composite_figure.png/cohort_summary.png
    # show.
    svp_display = postprocessing.apply_display_gamma(svp_vessel_enhanced, cfg.visualization)
    dvp_display = postprocessing.apply_display_gamma(dvp_vessel_enhanced, cfg.visualization)

    svp_density, _ = postprocessing.vessel_density(svp_enhanced, cfg.visualization)
    dvp_density, _ = postprocessing.vessel_density(dvp_enhanced, cfg.visualization)

    logger.info("[patient %d] saving results to %s", patient_id, out_dir)
    Image.fromarray(svp_raw_u8).save(out_dir / "svp_enface_raw.png")
    Image.fromarray(dvp_corrected_u8).save(out_dir / "dvp_enface_raw.png")
    Image.fromarray(svp_enhanced).save(out_dir / "svp_enface_enhanced.png")
    Image.fromarray(dvp_enhanced).save(out_dir / "dvp_enface_enhanced.png")
    Image.fromarray(svp_vessel_enhanced).save(out_dir / "svp_enface_vessel_enhanced.png")
    Image.fromarray(dvp_vessel_enhanced).save(out_dir / "dvp_enface_vessel_enhanced.png")
    Image.fromarray(svp_display).save(out_dir / "svp_enface_display.png")
    Image.fromarray(dvp_display).save(out_dir / "dvp_enface_display.png")

    metrics = {
        "patient_id": patient_id,
        "svp_vessel_density_pct": round(float(svp_density), 2),
        "dvp_vessel_density_pct": round(float(dvp_density), 2),
        "volume_shape_yzx": list(pdata.structural.shape),
        "mean_retinal_thickness_px": round(float(np.mean(bounds.rpe - bounds.ilm)), 1),
        "segmentation_qc": bounds.qc,
    }
    with open(out_dir / "metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)

    if bounds.qc["any_flagged"]:
        logger.warning(
            "[patient %d] segmentation QC flagged this volume -- check "
            "metrics.json['segmentation_qc'] and qc_boundary_overlay.png "
            "before trusting the enface output",
            patient_id,
        )

    result = PatientResult(
        patient_id=patient_id,
        svp_raw=svp_raw_u8,
        svp_enhanced=svp_enhanced,
        svp_vessel_enhanced=svp_vessel_enhanced,
        svp_display=svp_display,
        dvp_raw=dvp_corrected_u8,
        dvp_enhanced=dvp_enhanced,
        dvp_vessel_enhanced=dvp_vessel_enhanced,
        dvp_display=dvp_display,
        svp_density_pct=svp_density,
        dvp_density_pct=dvp_density,
        bounds=bounds,
        slabs=slabs,
        structural=pdata.structural,
        flow_despeckled=flow_ds,
        elapsed_s=time.time() - t0,
    )

    visualization.save_boundary_overlay(
        pdata.structural, bounds, out_dir / "qc_boundary_overlay.png"
    )
    visualization.save_patient_composite(result, out_dir / "composite_figure.png")
    visualization.save_composite_color(
        svp_display, dvp_display, out_dir / "svp_dvp_color_composite.png"
    )

    logger.info("[patient %d] done in %.1fs", patient_id, result.elapsed_s)
    return result
