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
from skimage.morphology import skeletonize
from octa_pipeline.config import PipelineConfig
from octa_pipeline import segmentation, projection, v2, v3, v3_fixed, v5, ezc
from .fast_segmentation import segment_fast

LAYERS = {'structural':'Structural OCT','svc':'SVC','dvc':'DVC','avc':'Avascular','cc':'Choriocapillaris','choroid':'Choroid'}


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


def segment_and_project(structural: np.ndarray, flow: np.ndarray | None, folder: Path, fov_x_mm: float | None, fov_y_mm: float | None, axial_um_per_px: float | None, progress=None, *, mode: str = 'v5-v6') -> dict:
    """Use the supplied v5 reconstruction exactly, with additive v6 EZC+FAZ.

    v5-v6: DP-based structural segmentation + v5 flow processing and enhancement.
    preview: faster OCT-only intensity-boundary proxy with v5 flow processing;
             NEVER presented as equivalent scientific quality.
    """
    folder.mkdir(exist_ok=True, parents=True)
    if mode not in ('v5-v6', 'preview'):
        raise ValueError('Invalid reconstruction mode')
    cfg = PipelineConfig()
    v2cfg = v2.V2Config()
    if progress: progress(.16, 'Segmenting anatomy · v5')
    bounds = segmentation.segment_volume(structural, cfg.segmentation) if mode == 'v5-v6' else segment_fast(structural)
    if progress: progress(.39, 'Tracing EZC · v6')
    # EZC is an informational boundary in v6, not a substitution for v5 slabs.
    ez = None
    if mode == 'v5-v6':
        try:
            ez = ezc.detect_ezc_volume(structural, bounds, cfg.segmentation).astype(np.float32)
        except (ValueError, IndexError) as e:
            # EZC is optional and must not invalidate otherwise valid v5 slabs.
            if progress: progress(.43, f'EZC unavailable ({type(e).__name__}); continuing v5')
    b = {name: getattr(bounds, name).astype(np.float32) for name in ('ilm', 'ipl_inl', 'opl_onl', 'rpe')}
    if ez is not None: b['ezc'] = ez
    if progress: progress(.52, 'Reconstructing en face · v5')
    images = {'structural': normalize(structural.mean(axis=1))}
    choroid_status = 'Not available without paired OCTA input'
    choroid_fallback = None
    if flow is not None:
        slabs = projection.build_slabs(bounds, cfg.segmentation)
        extra = v2.build_extra_slabs(bounds, structural.shape[1], v2cfg)
        ch_mode = os.environ.get('OCTAVE_CHOROID', 'proxy').lower()
        if ch_mode == 'choroidalyzer':
            from .choroid_optional import model_boundaries
            top, bottom, choroid_fallback = model_boundaries(structural,bounds.rpe,extra.choroid_top,extra.choroid_bottom)
            choroid_status = 'Choroidalyzer segmented region with per-column fallback'
        elif ch_mode == 'proxy':
            top,bottom=extra.choroid_top,extra.choroid_bottom
            choroid_status = 'RPE-offset proxy (not a choroid segmentation)'
        else:
            raise ValueError('OCTAVE_CHOROID must be proxy or choroidalyzer')
        flow_ds = projection.despeckle_flow(flow, cfg.projection.flow_median_ksize)
        bulk = projection.estimate_bulk_motion_floor(flow_ds, bounds.opl_onl, bounds.rpe, cfg.projection.bulk_motion_ref_margin)
        corrected = projection.suppress_bulk_motion(flow_ds, bulk, cfg.projection.bulk_motion_baseline_percentile)
        svc = projection.project_slab(corrected, slabs.svp_top, slabs.svp_bottom, cfg.projection)
        dvc = projection.suppress_projection_artifact_3d(corrected, slabs.svp_top, slabs.svp_bottom, slabs.dvp_top, slabs.dvp_bottom, cfg.projection.projection_artifact_suppression, cfg.projection.projection_artifact_decay_px)
        avc = v3.avascular_void(v3_fixed.median_projection(corrected, extra.avc_top, extra.avc_bottom))
        cc = projection._mean_projection(corrected, extra.cc_top, extra.cc_bottom)
        ch = projection._mean_projection(corrected, top, bottom)
        images.update({
            'svc': v2.enhance_v2(v3.row_flat_field(svc),v2cfg),
            'dvc': v2.enhance_v2(v3.row_flat_field(dvc),v2cfg),
            'avc': v5.avascular_display(avc),
            'cc': v2.enhance_v2(v3.row_flat_field(cc),v2cfg),
            'choroid': v2.enhance_v2(v3.row_flat_field(ch),v2cfg),
        })
        del flow_ds, corrected
    if progress: progress(.85, 'Saving reproducible results')
    for name, image in images.items():
        Image.fromarray(image).save(folder/f'{name}.png')
    np.savez_compressed(folder/'boundaries.npz', **b)
    np.save(folder/'structural.npy',structural)
    mm = fov_x_mm is not None and fov_y_mm is not None
    qc = bounds.qc
    warnings=[]
    if not mm: warnings.append('Lateral calibration is unknown. FAZ area remains in px²; mm² is unavailable.')
    if axial_um_per_px is None: warnings.append('Axial calibration is unknown. Thickness remains in pixels.')
    if qc.get('any_flagged'): warnings.append('Boundary tracing requires manual QC before measuring anatomy.')
    if flow is None: warnings.append('No paired OCTA flow input: vessel and FAZ analysis disabled.')
    else: warnings.append('Flow volume registration cannot be established solely from matching array shapes.')
    if mode == 'preview': warnings.append('Fast preview estimates inner layers geometrically. Do not use for final research measurements.')
    else: warnings.append('OPL/ONL in the supplied v5 segmenter remains an estimated boundary; EZC is informational only.')
    if flow is not None and choroid_fallback is not None: warnings.append(f'Choroidalyzer fallback used for {choroid_fallback:.1%} of columns.')
    if flow is not None and choroid_fallback is None: warnings.append('Choroid and CC offsets are pixel-space proxies; inspect for device/protocol mismatch.')
    return {
        'shape':list(structural.shape),'flow_available':flow is not None,'layers':list(images),
        'fovea_seed':estimate_fovea_seed(b['ilm'],b['rpe']),
        'calibration':{'fov_x_mm':fov_x_mm,'fov_y_mm':fov_y_mm,'axial_um_per_px':axial_um_per_px},
        'retinal_thickness_px':round(float(np.median(bounds.rpe-bounds.ilm)),1),
        'retinal_thickness_um':round(float(np.median(bounds.rpe-bounds.ilm)*axial_um_per_px),1) if axial_um_per_px else None,
        'qc':qc,'warnings':warnings,'choroid_boundary':choroid_status,
        'choroid_fallback_fraction':round(choroid_fallback,4) if choroid_fallback is not None else None,
        'pipeline_mode':mode,'pipeline':'Original supplied v5 projection + enhancement; v5 DP segmenter and additive v6 EZC/QC + FAZ' if mode=='v5-v6' else 'Explicit fast exploratory preview + supplied v5 projections',
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
    # Use the entire image distribution and a *strict* threshold. v5
    # midtone lift often places most background pixels at exactly 60;
    # a >= 52nd-percentile test would incorrectly mark ~100% as vessels.
    cut = np.percentile(image, threshold) if image.size else 255
    vessel = (image > cut) & (image > 0)
    vessel = ndi.binary_opening(vessel, structure=np.ones((2, 2)))
    labels, count = ndi.label(vessel)
    if count:
        sizes = np.bincount(labels.ravel())
        vessel &= (sizes[labels] >= 4) & (labels != 0)
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
    metrics.update(svc_morphometry(vessel,component) if detected else svc_morphometry(vessel,np.zeros_like(component)))
    return metrics, vessel, component


def svc_morphometry(vessel: np.ndarray, faz: np.ndarray) -> dict:
    """Pixel-space exploratory features adapted from uploaded v6 morphometry.

    Derived from the currently thresholded mask, not a pretrained vessel model.
    """
    from scipy.ndimage import convolve
    skel = skeletonize(vessel)
    nbr = convolve(skel.astype(np.uint8),np.array([[1,1,1],[1,0,1],[1,1,1]],dtype=np.uint8),mode='constant')
    h,w=vessel.shape
    a=max(int(skel.sum()),1)
    area=int(faz.sum())
    per=None
    if area:
        contours,_=cv2.findContours(faz.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        per=max((cv2.arcLength(c,True) for c in contours),default=0.0)
    out={
        'skeleton_density_per_1k_px':round(1000*float(skel.mean()),3),
        'mean_caliber_px_proxy':round(float(vessel.sum())/a,3),
        'branchpoint_density_per_10k_px':round(10000*float((skel & (nbr>=3)).mean()),3),
        'endpoint_density_per_10k_px':round(10000*float((skel & (nbr==1)).mean()),3),
        'faz_acircularity':round(float(per/(2*np.sqrt(np.pi*area))),3) if per and area else None,
    }
    # Same bounded, box-counting approximation as user-supplied v6 script.
    sizes=(2,3,4,6,8,12,16,24,32)
    entries=[]
    for size in sizes:
        if size>=min(h,w):continue
        nr,nc=h//size,w//size
        count=skel[:nr*size,:nc*size].reshape(nr,size,nc,size).any(axis=(1,3)).sum()
        if count>0:entries.append((size,count))
    out['fractal_dimension']=round(float(np.polyfit(np.log([1/i for i,_ in entries]),np.log([c for _,c in entries]),1)[0]),4) if len(entries)>=3 else None
    return out

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
