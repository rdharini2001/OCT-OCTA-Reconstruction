"""Research-only OCT boundary and OCTA projection analysis.

Retinal boundaries: imported user-supplied octa_pipeline heuristic method.
No clinically validated diagnoses or uncalibrated mm-derived quantities are produced.
"""
from __future__ import annotations

import os
import io
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.morphology import skeletonize, remove_small_objects
from octa_pipeline.config import PipelineConfig
from octa_pipeline import segmentation, projection, v2, v3, v3_fixed
from .fast_segmentation import segment_fast

LAYERS = {'structural': 'Structural reflectance', 'svc': 'Superficial vascular complex', 'dvc': 'Deep vascular complex', 'avc': 'Outer avascular retina', 'cc': 'Choriocapillaris', 'choroid': 'Choroidal proxy'}


def normalize(arr: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(arr.astype(np.float32))
    p1, p99 = np.percentile(arr, [1, 99.7])
    if p99 <= p1:
        return np.zeros(arr.shape, dtype=np.uint8)
    return np.uint8(np.clip((arr - p1) / (p99 - p1) * 255, 0, 255))


def png_bytes(gray: np.ndarray, palette: str = 'gray') -> bytes:
    out = io.BytesIO()
    if palette == 'gray':
        Image.fromarray(gray.astype(np.uint8), 'L').save(out, format='PNG')
    else:
        # Viridis / plasma gradients via OpenCV LUTs, RGB-ready
        cm = cv2.COLORMAP_TURBO if palette == 'turbo' else cv2.COLORMAP_VIRIDIS
        rgb = cv2.cvtColor(cv2.applyColorMap(gray.astype(np.uint8), cm), cv2.COLOR_BGR2RGB)
        Image.fromarray(rgb).save(out, format='PNG')
    return out.getvalue()


def segment_and_project(structural: np.ndarray, flow: np.ndarray | None, folder: Path, fov_x_mm: float | None, fov_y_mm: float | None, axial_um_per_px: float | None, progress=None) -> dict:
    folder.mkdir(exist_ok=True, parents=True)
    cfg = PipelineConfig()
    if progress: progress(0.16, 'Tracing retinal boundaries')
    bounds = segmentation.segment_volume(structural, cfg.segmentation) if os.environ.get("OCTAVE_SEGMENTER") == "legacy" else segment_fast(structural)
    b = {name: getattr(bounds, name).astype(np.float32) for name in ('ilm', 'ipl_inl', 'opl_onl', 'rpe')}
    np.savez_compressed(folder / 'boundaries.npz', **b)
    if progress: progress(0.58, 'Reconstructing en face projections')
    images = {'structural': normalize(structural.mean(axis=1))}
    if flow is not None:
        # Image-derived flow projection requires already-processed paired decorrelation data.
        slabs = projection.build_slabs(bounds, cfg.segmentation)
        extra = v2.build_extra_slabs(bounds, structural.shape[1], v2.V2Config())
        flow_ds = projection.despeckle_flow(flow, cfg.projection.flow_median_ksize)
        bulk = projection.estimate_bulk_motion_floor(flow_ds, bounds.opl_onl, bounds.rpe, cfg.projection.bulk_motion_ref_margin)
        corrected = projection.suppress_bulk_motion(flow_ds, bulk, cfg.projection.bulk_motion_baseline_percentile)
        svc = projection.project_slab(corrected, slabs.svp_top, slabs.svp_bottom, cfg.projection)
        dvc = projection.suppress_projection_artifact_3d(corrected, slabs.svp_top, slabs.svp_bottom, slabs.dvp_top, slabs.dvp_bottom, cfg.projection.projection_artifact_suppression, cfg.projection.projection_artifact_decay_px)
        avc = v3.avascular_void(v3_fixed.median_projection(corrected, extra.avc_top, extra.avc_bottom))
        cc = projection._mean_projection(corrected, extra.cc_top, extra.cc_bottom)
        # Note: offset proxy is intentionally labelled, not a segmented choroidal boundary.
        choroid = projection._mean_projection(corrected, extra.choroid_top, extra.choroid_bottom)
        images.update({'svc': normalize(v3.row_flat_field(svc)), 'dvc': normalize(v3.row_flat_field(dvc)), 'avc': normalize(avc), 'cc': normalize(cc), 'choroid': normalize(choroid)})
        del flow_ds, corrected
    if progress: progress(0.87, 'Calculating quality controls')
    for name, image in images.items():
        Image.fromarray(image).save(folder / f'{name}.png')
    # Real data not packed into source tree, kept local only.
    np.save(folder / 'structural.npy', structural)
    mm = fov_x_mm is not None and fov_y_mm is not None
    qc = bounds.qc
    warnings = []
    if not mm: warnings.append('Physical lateral scan dimensions missing: FAZ area and vascular length cannot be reported in mm units.')
    if axial_um_per_px is None: warnings.append('Axial calibration missing: retinal thickness is reported only in pixels.')
    if bool(qc['any_flagged']): warnings.append('Boundary segmentation flagged for review. Inspect overlaid B-scans before trusting slab analysis.')
    if flow is None: warnings.append('Structural-only mode: OCTA vessel density, FAZ and flow analysis are unavailable.')
    else: warnings.append('Flow volume assumed to be co-registered decorrelation/OCTA input; it cannot be inferred from structural OCT.')
    warnings.append('IPL/INL, OPL/ONL and choroidal estimates use geometric proxies in fast mode. Not validated tissue-class boundaries; research use only.')
    return {
        'shape': list(structural.shape), 'flow_available': flow is not None, 'layers': list(images),
        'fovea_seed': estimate_fovea_seed(b['ilm'],b['rpe']),
        'calibration': {'fov_x_mm': fov_x_mm, 'fov_y_mm': fov_y_mm, 'axial_um_per_px': axial_um_per_px},
        'retinal_thickness_px': round(float(np.median(bounds.rpe - bounds.ilm)), 1),
        'retinal_thickness_um': round(float(np.median(bounds.rpe - bounds.ilm) * axial_um_per_px), 1) if axial_um_per_px else None,
        'qc': qc, 'warnings': warnings,
        'pipeline': 'Fast vectorized heuristic boundaries (legacy DP optional) + user-provided octa_pipeline projection algorithms; OCTAVE v0.1',
    }


def estimate_fovea_seed(ilm: np.ndarray, rpe: np.ndarray) -> dict:
    """Thickness-minimum seed as in user-provided seg_v6_faz, with center fallback."""
    h,w=ilm.shape
    thickness=ndi.gaussian_filter(rpe-ilm,sigma=min(15,max(1,min(h,w)//12)))
    y0,y1,x0,x1=int(h*.2),int(h*.8),int(w*.2),int(w*.8)
    patch=thickness[y0:y1,x0:x1]
    y,x=np.unravel_index(np.argmin(patch),patch.shape)
    y,x=int(y+y0),int(x+x0)
    # Relative image-size guard: avoid random peripheral minima.
    if np.hypot((y-h/2)/max(h,1),(x-w/2)/max(w,1)) > .25:
        return {'x':w//2,'y':h//2,'source':'image centre: thickness minimum too distant'}
    return {'x':x,'y':y,'source':'smoothed retinal thickness minimum (heuristic)'}


def vessel_and_faz(image: np.ndarray, threshold: int = 52, seed_x: int | None = None, seed_y: int | None = None, fov_x_mm: float | None = None, fov_y_mm: float | None = None) -> tuple[dict, np.ndarray, np.ndarray]:
    """Thresholded displayed projection: exploratory mask, not physical perfusion/velocity.

    FAZ from foveal seed in enclosed non-vessel component following supplied
    seg_v6_faz approach. A failed enclosure returns detected=False, never area 0.
    """
    if not 1 <= threshold <= 99:
        raise ValueError('threshold must be between 1 and 99')
    h, w = image.shape
    # Positive vessel structure above percentile of contrast-enhanced view.
    # This is not a calibrated biological flow threshold.
    nonzero = image[image > 0]
    cut = np.percentile(nonzero, threshold) if nonzero.size else 255
    vessel = (image >= cut) & (image > 0)
    vessel = ndi.binary_opening(vessel, structure=np.ones((2, 2)))
    vessel = remove_small_objects(vessel, min_size=4)
    y = h // 2 if seed_y is None else int(seed_y)
    x = w // 2 if seed_x is None else int(seed_x)
    if not (0 <= x < w and 0 <= y < h):
        raise ValueError('Seed is outside en face image')
    detected = False
    component = np.zeros_like(vessel, dtype=bool)
    used_radius = None
    # Input might have very different pixel spacings; radii in px only.
    for r in (5, 3, 2):
        closed = ndi.binary_closing(vessel, structure=np.ones((2*r+1, 2*r+1)))
        labels, _ = ndi.label(~closed)
        seed_label = labels[y, x]
        if not seed_label:
            continue
        candidate = labels == seed_label
        frac = float(candidate.mean())
        if candidate[0].any() or candidate[-1].any() or candidate[:, 0].any() or candidate[:, -1].any():
            continue
        if 0.003 <= frac <= 0.12:
            detected, component, used_radius = True, candidate, r
            break
    px = int(component.sum())
    contours, _ = cv2.findContours(component.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    perimeter_px = float(max((cv2.arcLength(c, True) for c in contours), default=0)) if detected else None
    circularity = float(4 * np.pi * px / (perimeter_px * perimeter_px)) if perimeter_px and perimeter_px > 0 else None
    calibrated = fov_x_mm is not None and fov_y_mm is not None
    pixel_area = (fov_x_mm / w) * (fov_y_mm / h) if calibrated else None
    skeleton = skeletonize(vessel)
    skeleton_length_px = int(skeleton.sum())
    # Approximate vessel length: diagonal skeleton links are not corrected; document as a pixel proxy.
    vessel_length_mm = float(skeleton_length_px * ((fov_x_mm/w + fov_y_mm/h)/2)) if calibrated else None
    row_density = (100*vessel.mean(axis=1)).round(1).tolist()
    metrics = {
        'layer': 'svc', 'threshold_percentile': threshold, 'threshold_intensity': int(round(cut)),
        'vessel_density_pct': round(float(vessel.mean()*100), 2),
        'vessel_length_px_proxy': skeleton_length_px,
        'vessel_length_mm_proxy': round(vessel_length_mm, 3) if calibrated else None,
        'faz_detected': detected, 'faz_area_px': px if detected else None,
        'faz_area_mm2': round(float(px*pixel_area), 4) if detected and calibrated else None,
        'faz_perimeter_px': round(perimeter_px, 1) if perimeter_px is not None else None,
        'faz_circularity': round(circularity, 3) if circularity is not None else None,
        'faz_seed': {'x': x, 'y': y}, 'closing_radius_px': used_radius,
        'row_density_pct': row_density, 'calibrated': calibrated,
        'method_note': 'Displayed en face image percentile threshold + morphology. Skeleton length is a pixel-path proxy. No clinical validation.',
    }
    return metrics, vessel, component


def metrics_overlay(image: np.ndarray, vessel: np.ndarray, faz: np.ndarray, x: int, y: int) -> bytes:
    rgb = np.array(Image.fromarray(image.astype(np.uint8)).convert('RGB'))
    rgb[vessel] = (rgb[vessel].astype(float)*0.45 + np.array([23, 207, 199])*0.55).astype('uint8')
    if faz.any():
        contours, _ = cv2.findContours(faz.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(rgb, contours, -1, (255, 185, 113), 2)
    cv2.drawMarker(rgb, (x, y), (255, 145, 104), markerType=cv2.MARKER_CROSS, markerSize=12, thickness=1)
    return png_bytes_rgb(rgb)


def png_bytes_rgb(arr: np.ndarray) -> bytes:
    b = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8), 'RGB').save(b, format='PNG')
    return b.getvalue()
