"""Retinal boundary segmentation used to define the SVP / DVP enface slabs.

Method
------
Four surfaces are traced on the **structural (denoised) OCT** volume, which
has much better contrast/SNR for layer boundaries than the flow volume:

    ILM      - vitreoretinal interface (top of retina)
    IPL/INL  - inner plexiform / inner nuclear layer junction
    OPL/ONL  - outer plexiform / outer nuclear layer junction
    RPE      - retinal pigment epithelium (brightest, thickest band)

These give the two clinically standard OCTA slabs:

    Superficial Vascular Plexus (SVP): ILM      -> IPL/INL
    Deep Vascular Plexus        (DVP): IPL/INL  -> OPL/ONL

Approach (per B-scan, fully vectorized across A-scans):

1. A coarse ILM estimate is found as the first depth index where the
   Gaussian-smoothed intensity sustainably exceeds a local noise-floor
   threshold (adaptive, estimated from the vitreous region at the top of
   the frame) *and* stays elevated for a long trailing window afterward
   (real retina stays bright all the way to the RPE; a transient artifact
   like a vitreous floater or a specular reflection lights up briefly and
   returns to baseline -- without this persistence check, a floater only a
   little wider than the sustain window satisfies it exactly like a true
   transition would, and was observed doing so). The coarse estimate is
   then refined with a dynamic-programming shortest path in a narrow band
   around it (gradient/edge-strength cost) -- see `_dp_trace`.
2. Deeper boundaries are located by re-indexing the smoothed B-scan into
   an "ILM-relative depth" representation (`gather`), i.e. row 0 of
   `gathered` is the intensity *at* the ILM for every A-scan, row d is the
   intensity d pixels below the local ILM. This normalizes out bulk
   retinal curvature/tilt so that layer bands common across all A-scans
   line up, and boundaries can then be found within fixed search windows
   calibrated from the empirical ILM-aligned intensity profile -- see
   `config.SegmentationConfig`.
3. Within each search window, IPL/INL and RPE are each found as a
   dynamic-programming shortest path (`_dp_trace`) across the whole
   B-scan, rather than an independent per-A-scan extremum search. This is
   the key robustness property: a per-column search has no way to know its
   choice disagrees with its neighbors, so it can latch onto a vessel-
   shadow dip, a speckle outlier, or a stronger-but-irrelevant pathology
   feature one A-scan at a time, producing a jagged trace. A smoothness-
   constrained shortest path is jointly optimal over the *whole* B-scan at
   once, so an isolated bad pixel is only ever cheap to visit if it also
   stays close to what the neighboring columns picked -- noise gets
   outvoted by the rest of the path instead of corrupting it locally.
4. OPL/ONL is not searched directly (see below); it is derived from the
   two DP-traced boundaries, so it inherits their smoothness for free. A
   DP-refined, image-derived version of this was implemented and tested
   but reverted after visual inspection showed it converging on the wrong
   edge -- see the detailed note at that line for what was tried and why.
5. Columns with no real inner-retina signal at all (e.g. a shadow cast by
   an overlying opacity/lesion) are detected and bridged: the coarse ILM
   detector still reports a technically-valid sustained+persistent
   crossing there, on whatever bright structure sits deepest in the
   column, because that structure genuinely is sustained and persistent --
   it just isn't the true ILM. Bridging replaces the whole boundary stack
   in those columns with an interpolation from confident neighbors rather
   than trusting a trace built on a false anchor. See
   `_detect_low_confidence_columns` for how this is told apart from a
   column that's just genuinely thin (e.g. a real foveal pit) rather than
   missing signal entirely -- that distinction was the difference between
   fixing a real problem and flattening real anatomy, gotten wrong twice
   before landing on the current check.
6. Boundary surfaces get one more light regularization pass across the
   whole volume: frame-to-frame outlier rejection along the B-scan axis
   (catches whole-B-scan mis-detections, e.g. blinks) and a light 2D
   Gaussian smoothing, since the true anatomical boundary is a smooth
   manifold in both directions and the DP above only enforces smoothness
   within a single B-scan.

This is a fast approximation of the graph-search layer segmentation used
in the OCT literature (e.g. Chiu et al. 2010) and in commercial OCTA
review software. It is **not** a validated clinical segmentation -- see
README for limitations and the recommended upgrade path (full multi-
surface graph-cut / deep-learning segmentation) for later phases.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import binary_dilation, gaussian_filter, median_filter, sobel

from .config import SegmentationConfig

logger = logging.getLogger("octa_pipeline")


@dataclass
class BScanBoundaries:
    ilm: np.ndarray
    ipl_inl: np.ndarray
    opl_onl: np.ndarray
    rpe: np.ndarray


@dataclass
class VolumeBoundaries:
    ilm: np.ndarray       # (n_bscans, width)
    ipl_inl: np.ndarray
    opl_onl: np.ndarray
    rpe: np.ndarray
    qc: dict              # see segment_volume / _check_surface_quality


def _detect_ilm(smooth: np.ndarray, cfg: SegmentationConfig) -> np.ndarray:
    depth, width = smooth.shape
    noise_region = smooth[: cfg.ilm_noise_rows, :]
    noise_floor = float(np.median(noise_region))
    noise_std = float(np.std(noise_region))
    thresh = noise_floor + cfg.ilm_threshold_k * noise_std

    above = smooth > thresh
    csum = np.cumsum(np.vstack([np.zeros((1, width)), above]), axis=0)  # (depth+1, width)

    win = cfg.ilm_sustain_rows
    sustained_roll = csum[win:, :] - csum[:-win, :]
    sustained = sustained_roll == win  # (depth-win+1, width); True where a run of `win` starts

    # Persistence check: a genuine ILM stays bright for a long trailing
    # window (real retina, all the way toward RPE); a transient artifact
    # (a floater, a specular reflection) lights up for a few rows and then
    # returns to baseline. Without this, a `win`-row-long bump satisfies
    # `sustained` exactly like a true transition would, and the detector
    # locks onto it -- this was found directly: a floater ~17 rows wide
    # fooled the (then 15-row) sustain check and threw the whole boundary
    # stack off by ~100px in the columns it touched.
    persist = cfg.ilm_persist_rows
    idx = np.arange(sustained.shape[0])
    end_idx = np.minimum(idx + persist, depth)
    persist_len = (end_idx - idx).astype(np.float64)
    persist_sum = csum[end_idx, :] - csum[idx, :]
    persist_frac = persist_sum / persist_len[:, None]
    persistent_enough = persist_frac >= cfg.ilm_persist_frac

    valid = sustained & persistent_enough

    has_hit = valid.any(axis=0)
    ilm = np.argmax(valid, axis=0)
    # fallback for columns with no valid (sustained AND persistent) crossing: mid-depth
    fallback = depth // 2
    ilm = np.where(has_hit, ilm, fallback)
    return ilm.astype(np.float64)


def _dp_trace(cost: np.ndarray, max_step: int, step_penalty: float) -> np.ndarray:
    """Dynamic-programming shortest path across a (R, width) cost surface:
    for each A-scan column, pick a row minimizing cumulative cost subject to
    a per-column smoothness constraint (adjacent columns' picks differ by at
    most `max_step`, with an extra `step_penalty` per pixel of jump).

    This replaces independent per-column extremum search with a single
    jointly-optimal path over the whole B-scan. It is the standard fix for
    "noisy"/jagged OCT layer traces: a column-by-column search can latch
    onto a local artifact (speckle, a vessel shadow, a stronger unrelated
    feature) with no way to know it disagrees with its neighbors; a
    smoothness-constrained shortest path cannot do that, since a column's
    choice is only ever cheap if it also stays close to what its neighbors
    picked. Runs in O(R * width) via a small sliding window per column
    (numpy, no python-level loop over rows).
    """
    R, width = cost.shape
    win = 2 * max_step + 1
    offsets = np.arange(-max_step, max_step + 1)
    penalty = step_penalty * np.abs(offsets)

    acc = cost[:, 0].copy()
    backptr = np.empty((R, width), dtype=np.int32)
    backptr[:, 0] = np.arange(R)

    for x in range(1, width):
        padded = np.pad(acc, (max_step, max_step), mode="constant", constant_values=np.inf)
        windows = sliding_window_view(padded, win) + penalty[None, :]  # (R, win)
        local_best = np.argmin(windows, axis=1)
        best_val = windows[np.arange(R), local_best]
        best_prev = np.clip(local_best - max_step + np.arange(R), 0, R - 1)
        acc = cost[:, x] + best_val
        backptr[:, x] = best_prev

    path = np.empty(width, dtype=np.int32)
    path[-1] = np.argmin(acc)
    for x in range(width - 1, 0, -1):
        path[x - 1] = backptr[path[x], x]
    return path.astype(np.float64)


def _flatten_edges(arr: np.ndarray, guard: int) -> np.ndarray:
    """Replace the outer `guard` columns with the nearest interior value to
    suppress signal-rolloff artifacts at the A-scan-edges of each B-scan."""
    if guard <= 0 or arr.shape[-1] <= 2 * guard:
        return arr
    out = arr.copy()
    out[..., :guard] = arr[..., guard : guard + 1]
    out[..., -guard:] = arr[..., -guard - 1 : -guard]
    return out


def _ilm_dp_refine(smooth: np.ndarray, coarse_ilm: np.ndarray, cfg: SegmentationConfig) -> np.ndarray:
    """Refine the coarse threshold-based ILM estimate with a DP shortest
    path in a narrow band around it, using a gradient (edge-strength) cost.
    The coarse estimate anchors the search so the DP cannot wander down to
    a stronger but wrong edge (e.g. the RPE, whose gradient is much
    stronger); the DP then removes single-A-scan jitter and the signal-
    rolloff artifacts that used to require a hand-rolled edge-flattening
    step.
    """
    depth, width = smooth.shape
    band = cfg.ilm_dp_band
    grad = sobel(smooth, axis=0)  # positive: darker above, brighter below (the ILM transition)

    offsets = np.arange(-band, band + 1)
    z_idx = np.clip(coarse_ilm[None, :] + offsets[:, None], 0, depth - 1).astype(np.int64)
    cost = -np.take_along_axis(grad, z_idx, axis=0)  # minimize cost == maximize positive gradient

    path_local = _dp_trace(cost, cfg.ilm_dp_max_step, cfg.ilm_dp_step_penalty)
    return coarse_ilm + (path_local - band)


def _detect_low_confidence_columns(smooth: np.ndarray, coarse_ilm: np.ndarray, cfg: SegmentationConfig) -> np.ndarray:
    """Flag columns where the whole downstream boundary stack is untrustworthy
    because there is no real inner-retina signal for the coarse ILM to have
    latched onto -- as opposed to a column where the retina is just thin.

    This was written after visually inspecting two "sudden depression"
    regions (Patients 4 and 5) that looked superficially similar (all four
    boundaries diving down together) but turned out to be different in
    kind: Patient 5's dip is a real, thin-but-present foveal pit -- a small
    genuine brightness bump is still there, just weaker. Patient 4's dip
    sits over a ~200-row span of pure noise-floor signal (no bump at all)
    before hitting a bright, RPE-like complex; the coarse ILM detector
    still reports a "valid" sustained+persistent crossing there because
    that deep complex genuinely is sustained and persistent -- it just
    isn't the true ILM, since there is no continuous elevated retinal
    tissue above it. A per-column check can't tell these apart by looking
    at where each column's own (possibly wrong) ILM landed -- that's
    circular. Instead this checks, at a FIXED depth window anchored to the
    B-scan's median ILM position (robust to a handful of bad columns), for
    any real brightness bump: comfortably above the noise floor for a
    genuine signal (even Patient 5's foveal bump clears it easily), but
    absent entirely in Patient 4's void.
    """
    depth, width = smooth.shape
    median_ilm = int(np.median(coarse_ilm))
    lo = max(0, median_ilm + cfg.confidence_window[0])
    hi = min(depth, median_ilm + cfg.confidence_window[1])
    window = smooth[lo:hi, :]

    noise_region = smooth[: cfg.ilm_noise_rows, :]
    noise_floor = float(np.median(noise_region))
    noise_std = float(np.std(noise_region))
    thresh = noise_floor + cfg.confidence_threshold_k * noise_std

    peak = window.max(axis=0)
    low_signal = peak < thresh

    # An earlier version of this also flagged columns where a sustained
    # (but not persistent) candidate was found and rejected before the
    # final coarse ILM, to catch the transition zone at the edge of a
    # low-signal region. Dropped: the same rejection pattern turned out to
    # fire on a real foveal pit too (a shallow non-ILM bump correctly
    # rejected so the detector falls through to the true, deeper foveal
    # ILM) -- and a rejection can be either correct or incorrect depending
    # on whether the deeper candidate it falls through to is itself real,
    # which isn't knowable from the rejection alone. The peak check by
    # itself was clean on both patients (no false positives on the fovea);
    # the narrower transition-zone columns it misses are left to the
    # cross-volume median-filter regularization, which is already relied
    # on elsewhere in this module for comparable-width residual noise.
    return low_signal


def _bridge_low_confidence_columns(
    low_confidence: np.ndarray, *surfaces: np.ndarray
) -> list[np.ndarray]:
    """Replace flagged columns in each surface with a linear interpolation
    from the nearest confident columns, bridging across the untrustworthy
    span instead of trusting a boundary trace built on a false anchor.
    Left alone if too little (or all) of the B-scan is confident to
    interpolate from.
    """
    width = low_confidence.shape[0]
    good_idx = np.where(~low_confidence)[0]
    if low_confidence.sum() == 0 or len(good_idx) < 2:
        return list(surfaces)
    bad_idx = np.where(low_confidence)[0]
    bridged = []
    for surface in surfaces:
        out = surface.copy()
        out[bad_idx] = np.interp(bad_idx, good_idx, surface[good_idx])
        bridged.append(out)
    return bridged


def segment_bscan(struct_bscan: np.ndarray, cfg: SegmentationConfig) -> BScanBoundaries:
    depth, width = struct_bscan.shape
    smooth = gaussian_filter(struct_bscan.astype(np.float32), sigma=cfg.gauss_sigma_bscan)

    coarse_ilm = _detect_ilm(smooth, cfg)
    coarse_ilm = _flatten_edges(coarse_ilm, cfg.edge_guard_cols)
    ilm = _ilm_dp_refine(smooth, coarse_ilm, cfg)

    offsets = np.arange(cfg.dmax)
    z_idx = np.clip(ilm[None, :] + offsets[:, None], 0, depth - 1).astype(np.int64)
    gathered = np.take_along_axis(smooth, z_idx, axis=0)  # (dmax, width)

    # IPL/INL: shortest path seeking the intensity valley just below the
    # first bright RNFL/GCL/IPL complex. The smoothness constraint is what
    # keeps this from diving into a vessel-shadow dip or a pathology
    # feature the way an independent per-column valley search would.
    #
    # A pure shortest-path search, though, has no notion of "first valley":
    # it happily settles on whichever low-cost valley is cheapest overall,
    # which was empirically the *second* dip (near the INL/OPL region) more
    # often than the true IPL/INL one -- the opposite failure mode from the
    # old per-column method, but still wrong, and it collapsed the DVP slab
    # thin in the process. `depth_bias` adds a small cost per pixel of
    # depth so the shallow, physiologically-correct valley wins unless a
    # much stronger one sits deeper -- restoring the "prefer the first
    # valley" behavior while keeping the DP's smoothness guarantee.
    lo, hi = cfg.ipl_inl_window
    depth_bias = cfg.ipl_inl_depth_bias * np.arange(hi - lo)[:, None]
    ipl_inl_cost = gathered[lo:hi, :] + depth_bias
    ipl_inl_rel = _dp_trace(ipl_inl_cost, cfg.ipl_inl_dp_max_step, cfg.ipl_inl_dp_step_penalty) + lo
    ipl_inl = ilm + ipl_inl_rel

    # RPE: shortest path seeking the brightest band after skipping past the
    # inner-retina complex; scale-invariant since it operates in ILM-
    # relative offsets, and now path-connected so a stray bright artifact
    # can't pull a single column off the true RPE surface.
    lo, hi = cfg.rpe_min_offset, cfg.dmax
    rpe_cost = -gathered[lo:hi, :]
    rpe_rel = _dp_trace(rpe_cost, cfg.rpe_dp_max_step, cfg.rpe_dp_step_penalty) + lo
    rpe = ilm + rpe_rel

    # OPL/ONL: the thin OPL band is not reliably resolvable by direct
    # valley-search across the whole volume (see README limitations), so
    # the deep-plexus slab's lower edge is instead placed at a fixed
    # fraction of the (robustly detected) IPL/INL -> RPE span, matching
    # the approximate combined INL+OPL thickness proportion. This keeps
    # the deep slab boundary well-behaved (scale-invariant, monotonic)
    # even where a direct valley search would be noisy.
    #
    # A DP-refined version of this (coarse fraction anchor + narrow-band
    # gradient search, mirroring `_ilm_dp_refine`) was implemented and
    # tuned to be numerically stable (band-width-invariant with a
    # sufficient anchor penalty -- see git history / README), but visual
    # inspection against the plain fraction estimate showed it was *wrong*,
    # not just different: on Patient 5's non-foveal B-scans it pulled
    # OPL/ONL down until it nearly touched RPE across large stretches of
    # the frame, erasing the outer-nuclear-layer gap the fraction estimate
    # correctly preserves. It had converged on a real, strong, but
    # anatomically wrong edge (most likely the ELM/IS-OS complex, which
    # sits close to RPE and produces a strong gradient of its own) rather
    # than the true OPL/ONL junction. "Numerically stable" and
    # "anatomically correct" turned out to be different things here, and
    # without expert-annotated ground truth there was no principled way to
    # fix the discriminator rather than just relocating the same ambiguity
    # -- so this was reverted rather than shipped. See README "Known
    # limitations" for what a validated fix actually needs.
    opl_onl = ipl_inl + cfg.opl_onl_fraction * (rpe - ipl_inl)

    # Bridge columns with no real inner-retina signal to have anchored the
    # ILM detection to in the first place (see _detect_low_confidence_columns)
    # -- the whole boundary stack downstream of a false ILM is untrustworthy
    # there, not just ILM itself, so all four surfaces are re-interpolated
    # across the flagged span from confident neighbors.
    low_confidence = _detect_low_confidence_columns(smooth, coarse_ilm, cfg)
    if low_confidence.any() and cfg.confidence_dilate_px > 0:
        low_confidence = binary_dilation(
            low_confidence, iterations=cfg.confidence_dilate_px
        )
    ilm, ipl_inl, opl_onl, rpe = _bridge_low_confidence_columns(
        low_confidence, ilm, ipl_inl, opl_onl, rpe
    )

    # enforce anatomical ordering ILM < IPL/INL < OPL/ONL < RPE; on violation
    # (speckle/pathology outliers) clamp to a minimal separation -- the
    # cross-volume regularization pass below cleans up remaining outliers.
    ipl_inl = np.maximum(ipl_inl, ilm + 3)
    opl_onl = np.maximum(opl_onl, ipl_inl + 3)
    rpe = np.maximum(rpe, opl_onl + 3)
    rpe = np.minimum(rpe, depth - 1)

    ipl_inl = _flatten_edges(ipl_inl, cfg.edge_guard_cols)
    opl_onl = _flatten_edges(opl_onl, cfg.edge_guard_cols)
    rpe = _flatten_edges(rpe, cfg.edge_guard_cols)

    return BScanBoundaries(ilm=ilm, ipl_inl=ipl_inl, opl_onl=opl_onl, rpe=rpe)


def _regularize_surface(surface: np.ndarray, cfg: SegmentationConfig) -> np.ndarray:
    """surface: (n_bscans, width) raw per-Bscan boundary estimates."""
    # 1) median filter along each B-scan (x) to kill single-A-scan speckle outliers
    filt = median_filter(surface, size=(1, cfg.boundary_median_width), mode="nearest")

    # 2) outlier rejection along y (B-scan axis): replace any column-wise
    #    value that deviates too far from a robust LOCAL reference (median
    #    over a window of nearby B-scans) with that reference.
    #
    #    This must not be a sequential "compare to the previous (already
    #    corrected) frame" scheme -- that construction is a single-pole
    #    cascading filter: once one B-scan trips the threshold, the frame
    #    after it is compared against the *replacement* value, not the
    #    true signal, so if the true signal has moved on (which it always
    #    has, since the trip was a real change or noise, not a stall) that
    #    frame trips too, and the correction propagates forward and
    #    freezes every subsequent B-scan at the original stale value --
    #    permanently, since the frozen reference never gets a chance to
    #    re-sync with reality. This was found by direct inspection: a
    #    volume with 7 genuine single-frame outliers on one column ended
    #    up with 121,911 "corrected" values out of ~200,000 total, because
    #    each of those 7 outliers froze the rest of the column.
    #
    #    Comparing against a windowed median of the (already x-filtered)
    #    data instead makes every frame's decision independent of any
    #    other frame's correction, so a bad frame can never propagate.
    y_window = cfg.boundary_outlier_y_window
    reference = median_filter(filt, size=(y_window, 1), mode="nearest")
    is_outlier = np.abs(filt - reference) > cfg.max_frame_to_frame_jump
    corrected = np.where(is_outlier, reference, filt)

    # 3) light 2D smoothing of the boundary surface (anatomy varies smoothly)
    smoothed = gaussian_filter(corrected, sigma=cfg.boundary_gauss_sigma_xy, mode="nearest")
    outlier_fraction = float(is_outlier.mean())
    return smoothed, outlier_fraction


def _check_surface_quality(name: str, smoothed: np.ndarray, outlier_fraction: float, cfg: SegmentationConfig) -> dict:
    """Post-hoc sanity check on a finished boundary surface, meant to catch
    *any* future regression automatically -- not just the specific bug this
    was written for -- rather than relying on someone visually inspecting
    every QC overlay for every patient. A deployment running on hundreds of
    patients needs failures to announce themselves, not wait to be noticed.
    """
    residual_jump = float(np.abs(np.diff(smoothed, axis=0)).max()) if smoothed.shape[0] > 1 else 0.0
    flagged = (
        outlier_fraction > cfg.qc_max_outlier_fraction
        or residual_jump > cfg.qc_max_residual_jump_px
    )
    if flagged:
        logger.warning(
            "segmentation QC flag on '%s': outlier_fraction=%.1f%% (limit %.1f%%), "
            "residual_frame_jump=%.1fpx (limit %.1fpx) -- inspect this volume's "
            "qc_boundary_overlay.png before trusting its enface output",
            name,
            100 * outlier_fraction,
            100 * cfg.qc_max_outlier_fraction,
            residual_jump,
            cfg.qc_max_residual_jump_px,
        )
    return {
        "outlier_fraction": round(outlier_fraction, 4),
        "residual_frame_jump_px": round(residual_jump, 1),
        "flagged": flagged,
    }


def segment_volume(structural: np.ndarray, cfg: SegmentationConfig) -> VolumeBoundaries:
    """structural: (y, z, x) uint8 volume."""
    n_bscans, depth, width = structural.shape
    ilm = np.empty((n_bscans, width))
    ipl_inl = np.empty((n_bscans, width))
    opl_onl = np.empty((n_bscans, width))
    rpe = np.empty((n_bscans, width))

    for y in range(n_bscans):
        b = segment_bscan(structural[y], cfg)
        ilm[y] = b.ilm
        ipl_inl[y] = b.ipl_inl
        opl_onl[y] = b.opl_onl
        rpe[y] = b.rpe

    ilm, ilm_outlier_frac = _regularize_surface(ilm, cfg)
    ipl_inl, ipl_inl_outlier_frac = _regularize_surface(ipl_inl, cfg)
    opl_onl, opl_onl_outlier_frac = _regularize_surface(opl_onl, cfg)
    rpe, rpe_outlier_frac = _regularize_surface(rpe, cfg)

    # re-enforce ordering after smoothing
    ipl_inl = np.maximum(ipl_inl, ilm + 3)
    opl_onl = np.maximum(opl_onl, ipl_inl + 3)
    rpe = np.maximum(rpe, opl_onl + 3)
    rpe = np.minimum(rpe, depth - 1)

    qc = {
        "ilm": _check_surface_quality("ilm", ilm, ilm_outlier_frac, cfg),
        "ipl_inl": _check_surface_quality("ipl_inl", ipl_inl, ipl_inl_outlier_frac, cfg),
        "opl_onl": _check_surface_quality("opl_onl", opl_onl, opl_onl_outlier_frac, cfg),
        "rpe": _check_surface_quality("rpe", rpe, rpe_outlier_frac, cfg),
    }
    qc["any_flagged"] = any(v["flagged"] for v in qc.values())

    # SVP/DVP slab thickness plausibility: catches a wrong-but-internally-
    # consistent segmentation (e.g. a boundary collapsed onto another) that
    # the per-surface checks above wouldn't see, since each surface can look
    # individually smooth while the *pair* is anatomically implausible.
    svp_thickness = ipl_inl - ilm
    dvp_thickness = opl_onl - ipl_inl
    thin = cfg.qc_min_slab_thickness_px
    thick = cfg.qc_max_slab_thickness_px
    svp_bad_frac = float(((svp_thickness < thin) | (svp_thickness > thick)).mean())
    dvp_bad_frac = float(((dvp_thickness < thin) | (dvp_thickness > thick)).mean())
    qc["svp_thickness_implausible_fraction"] = round(svp_bad_frac, 4)
    qc["dvp_thickness_implausible_fraction"] = round(dvp_bad_frac, 4)
    if max(svp_bad_frac, dvp_bad_frac) > cfg.qc_max_outlier_fraction:
        qc["any_flagged"] = True
        logger.warning(
            "segmentation QC flag on slab thickness: SVP implausible=%.1f%%, "
            "DVP implausible=%.1f%% (limit %.1f%%) -- inspect this volume's "
            "qc_boundary_overlay.png before trusting its enface output",
            100 * svp_bad_frac,
            100 * dvp_bad_frac,
            100 * cfg.qc_max_outlier_fraction,
        )

    return VolumeBoundaries(ilm=ilm, ipl_inl=ipl_inl, opl_onl=opl_onl, rpe=rpe, qc=qc)
