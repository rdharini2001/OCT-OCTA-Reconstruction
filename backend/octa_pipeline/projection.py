"""Slab definition and enface (flow) projection.

Given the four segmented boundary surfaces (ILM, IPL/INL, OPL/ONL, RPE),
this module defines the two OCTA slabs and projects the flow (decorrelation)
volume through each one to produce the Superficial and Deep enface images:

    SVP slab: [ILM + svp_top_guard,  IPL/INL)
    DVP slab: [IPL/INL,              OPL/ONL + dvp_bottom_extra)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ProjectionConfig, SegmentationConfig
from .segmentation import VolumeBoundaries


@dataclass
class Slabs:
    svp_top: np.ndarray
    svp_bottom: np.ndarray
    dvp_top: np.ndarray
    dvp_bottom: np.ndarray


def build_slabs(bounds: VolumeBoundaries, seg_cfg: SegmentationConfig) -> Slabs:
    svp_top = bounds.ilm + seg_cfg.svp_top_guard
    svp_bottom = bounds.ipl_inl
    dvp_top = bounds.ipl_inl
    dvp_bottom = bounds.opl_onl + seg_cfg.dvp_bottom_extra
    return Slabs(svp_top=svp_top, svp_bottom=svp_bottom, dvp_top=dvp_top, dvp_bottom=dvp_bottom)


def _mean_projection(volume: np.ndarray, top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    """Efficient mean projection over a per-column [top, bottom) depth window,
    using a cumulative-sum trick so the whole volume is only scanned once."""
    n_bscans, depth, width = volume.shape
    top_i = np.clip(np.round(top), 0, depth).astype(np.int64)
    bot_i = np.clip(np.round(bottom), 0, depth).astype(np.int64)
    bot_i = np.maximum(bot_i, top_i + 1)

    csum = volume.astype(np.float32)
    np.cumsum(csum, axis=1, out=csum)
    pad = np.zeros((n_bscans, 1, width), dtype=np.float32)
    csum = np.concatenate([pad, csum], axis=1)  # (y, depth+1, x)

    top_sum = np.take_along_axis(csum, top_i[:, None, :], axis=1)[:, 0, :]
    bot_sum = np.take_along_axis(csum, bot_i[:, None, :], axis=1)[:, 0, :]
    total = bot_sum - top_sum
    count = (bot_i - top_i).astype(np.float32)
    return total / np.maximum(count, 1)


def _gather_slab(
    volume: np.ndarray, top: np.ndarray, bottom: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gather a per-column bounded [top, bottom) depth window from a 3D
    (y, z, x) volume, without scanning the full depth. Returns
    (gathered values, valid mask, absolute z index), all shaped
    (y, maxwin, x) where maxwin is the largest slab thickness present.
    Shared by `_max_projection` and `suppress_projection_artifact_3d`.
    """
    n_bscans, depth, width = volume.shape
    top_i = np.clip(np.round(top), 0, depth - 1).astype(np.int64)
    bot_i = np.clip(np.round(bottom), 0, depth).astype(np.int64)
    bot_i = np.maximum(bot_i, top_i + 1)
    thickness = bot_i - top_i
    maxwin = int(thickness.max()) + 1
    offsets = np.arange(maxwin)
    z_idx = np.clip(top_i[:, None, :] + offsets[None, :, None], 0, depth - 1)
    gathered = np.take_along_axis(volume.astype(np.float32), z_idx, axis=1)
    valid = offsets[None, :, None] < thickness[:, None, :]
    return gathered, valid, z_idx


def _max_projection(volume: np.ndarray, top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    """Max projection over a per-column [top, bottom) depth window, via a
    bounded gather (window size = max slab thickness present, not the full
    depth) so memory stays proportional to slab thickness, not volume depth.
    """
    gathered, valid, _ = _gather_slab(volume, top, bottom)
    gathered = np.where(valid, gathered, -np.inf)
    proj = gathered.max(axis=1)
    proj[~np.isfinite(proj)] = 0
    return proj


def project_slab(
    flow_volume: np.ndarray, top: np.ndarray, bottom: np.ndarray, cfg: ProjectionConfig
) -> np.ndarray:
    if cfg.projection_mode == "max":
        return _max_projection(flow_volume, top, bottom)
    return _mean_projection(flow_volume, top, bottom)


def despeckle_flow(flow_volume: np.ndarray, ksize: int) -> np.ndarray:
    """Light per-B-scan median despeckling of the flow volume prior to
    projection (reduces shot-noise speckle without smearing vessels)."""
    if ksize <= 1:
        return flow_volume
    from scipy.ndimage import median_filter

    out = np.empty_like(flow_volume)
    for y in range(flow_volume.shape[0]):
        out[y] = median_filter(flow_volume[y], size=ksize)
    return out


def estimate_bulk_motion_floor(
    flow_volume: np.ndarray, opl_onl: np.ndarray, rpe: np.ndarray, margin: int = 10
) -> np.ndarray:
    """Estimate a per-B-scan bulk-motion "noise floor" from the ONL band
    (between OPL/ONL and RPE), which is real tissue but normally avascular.

    This targets a specific, well-known OCTA artifact, distinct from the
    speckle noise `despeckle_flow` handles: bulk transverse eye motion
    during acquisition of a B-scan makes the whole frame register spurious
    decorrelation, not just true vessels, since any tissue displacement
    between repeat scans reads as "flow" regardless of whether there was
    a vessel there. It shows up as a bright horizontal line spanning the
    full width of the enface image (visible in every patient in this
    dataset) -- confirmed directly by comparing a flagged B-scan's flow
    frame against its neighbors: the flagged frame's *entire* retinal band
    was uniformly bright, not just its vessels.

    A vitreous-only reference window does not work here (no tissue means
    no motion signal to read), so the reference has to be inside the
    retina but in a layer with no true vasculature -- the ONL fits, and is
    already segmented.
    """
    depth = flow_volume.shape[1]
    top = np.clip(opl_onl + margin, 0, depth - 1)
    bottom = np.clip(rpe - margin, 1, depth)
    bottom = np.maximum(bottom, top + 1)
    ref_mean = _mean_projection(flow_volume, top, bottom)  # (y, x)
    return np.median(ref_mean, axis=1)  # (y,)


def suppress_bulk_motion(
    flow_volume: np.ndarray,
    per_bscan_floor: np.ndarray,
    baseline_percentile: float = 50.0,
    min_scale: float = 0.35,
) -> np.ndarray:
    """Scale down each B-scan's flow signal in proportion to how far its
    motion floor exceeds the cohort baseline.

    A flat per-pixel subtraction (an earlier version of this function)
    turned out not to be strong enough: bulk motion's effect on measured
    decorrelation scales with the local signal, not just an additive
    constant, so a fixed subtraction barely dents the artifact in the
    brighter inner-retina bands even after fully explaining the excess
    seen in the (naturally dim) ONL reference band. A multiplicative
    correction -- attenuating the *whole* corrupted B-scan proportionally
    to how many multiples of baseline its reference reading is -- responds
    to the size of the corruption instead of just its absolute reference
    value, and empirically suppresses the artifact far more effectively.
    `min_scale` floors how much any single B-scan can be attenuated, so a
    catastrophically corrupted frame is suppressed rather than zeroed out
    (which would trade a bright artifact line for a dark dropout line).
    """
    baseline = np.percentile(per_bscan_floor, baseline_percentile)
    baseline = max(baseline, 1e-3)
    scale = np.clip(baseline / np.maximum(per_bscan_floor, baseline), min_scale, 1.0)
    corrected = flow_volume.astype(np.float32) * scale[:, None, None]
    return corrected


def suppress_projection_artifact_3d(
    flow_volume: np.ndarray,
    svp_top: np.ndarray,
    svp_bottom: np.ndarray,
    dvp_top: np.ndarray,
    dvp_bottom: np.ndarray,
    fraction: float,
    decay_px: float,
) -> np.ndarray:
    """Depth-resolved (3D) superficial-shadow suppression for the DVP
    projection, done on the flow volume *before* collapsing to 2D, unlike
    the enface-level slab subtraction this replaces.

    Why this is a real difference and not just moving the same math: for a
    MAX projection specifically, subtracting one constant per A-scan from
    every depth sample before taking the max is mathematically identical
    to subtracting that same constant from the max value after projecting
    (max(f(z) - c) == max(f(z)) - c for constant c). So simply moving a
    *flat* per-A-scan subtraction earlier in the pipeline would have
    changed nothing. The actual fix is making the subtraction genuinely
    depth-dependent: a superficial vessel's projection tail is strongest in
    the flow samples immediately below it and decays with distance, so a
    true independent deep vessel sitting further down the same A-scan can
    exceed the (locally weaker, at that depth) shadow estimate and survive,
    while flow samples close to the superficial peak's depth -- where a
    tail is actually expected -- get suppressed harder. A flat per-A-scan
    subtraction cannot make this distinction at all.

    For each A-scan: find the superficial slab's peak flow value and its
    depth; model the expected shadow in the deep slab as that peak value
    times `fraction`, decaying exponentially with distance (in pixels)
    below the superficial peak's depth; subtract that estimate from every
    deep-slab depth sample (floored at 0) before taking the max.

    `decay_px` is an empirically-chosen falloff distance, not a calibrated
    physical unit (this dataset's axial um/pixel scale was not supplied --
    see README/config.py) -- tuned visually against Patient 1's known
    superficial-vessel-shaped contamination in the DVP slab.

    Known limitation (Patient 3): this suppresses the contamination well on
    4 of 5 patients (visually confirmed, and DVP density rises sensibly:
    +2-6 percentage points), but Patient 3's DVP still closely resembles
    its SVP even after this fix -- tested and ruled out as the cause:
    decay length (15-80px), suppression fraction up to 0.9 (removing 90%
    of the superficial peak), and rescaling the shadow estimate to each
    patient's own DVP intensity range all left the same large-vessel
    arcade visible in Patient 3's DVP. This means it is very unlikely to
    be an under-tuned shadow model and more likely a real, patient-specific
    effect this slab-subtraction-style approach cannot fix: either
    Patient 3's superficial vessels are large/thick enough that genuine
    decorrelation signal persists across the slab boundary rather than
    just a projection tail, or this patient's segmented slab boundaries
    place the DVP window somewhere still functionally superficial for this
    eye. See README "Known limitations" -- resolving this needs either
    expert-annotated ground truth or the axial calibration this dataset
    doesn't include, not another round of parameter tuning.
    """
    if fraction <= 0:
        return _max_projection(flow_volume, dvp_top, dvp_bottom)

    svp_gathered, svp_valid, svp_z = _gather_slab(flow_volume, svp_top, svp_bottom)
    svp_masked = np.where(svp_valid, svp_gathered, -np.inf)
    svp_peak_idx = np.argmax(svp_masked, axis=1)  # (y, x)
    svp_peak_val = np.take_along_axis(svp_masked, svp_peak_idx[:, None, :], axis=1)[:, 0, :]
    svp_peak_val = np.where(np.isfinite(svp_peak_val), svp_peak_val, 0.0)

    dvp_gathered, dvp_valid, dvp_z = _gather_slab(flow_volume, dvp_top, dvp_bottom)
    # Distance from the TOP of the DVP slab (i.e. the IPL/INL boundary),
    # not from the superficial peak's own depth: the superficial slab can
    # itself be tens of px thick, so anchoring decay to wherever inside it
    # the single brightest voxel happens to sit was found (empirically, by
    # inspecting the failure below) to apply a large chunk of the decay
    # *before* the DVP slab is even reached, silently weakening suppression
    # throughout the slab's actual depth range. Anchoring to the slab
    # boundary itself is both more physically sensible (shadow starts where
    # the superficial layer ends) and avoids that silent double-decay.
    dist = np.maximum(dvp_z - dvp_top[:, None, :], 0).astype(np.float32)
    shadow = fraction * svp_peak_val[:, None, :] * np.exp(-dist / decay_px)
    corrected = np.maximum(dvp_gathered - shadow, 0.0)
    corrected = np.where(dvp_valid, corrected, -np.inf)
    proj = corrected.max(axis=1)
    proj[~np.isfinite(proj)] = 0
    return proj
