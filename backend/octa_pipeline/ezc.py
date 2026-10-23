"""Ellipsoid-zone complex (EZC) boundary: the photoreceptor IS/OS bright band,
detected as a new, additive boundary on top of the already-validated ILM/IPL-INL/
OPL-ONL/RPE from segmentation.py. Written as a separate module rather than
modifying segment_volume() directly, since v1-v5 all depend on that function's
existing output shape and this is new, less-tested territory.

Why it exists: an attempt to directly re-detect OPL/ONL from the image (see
README/METHOD.md "OPL/ONL" sections) kept being pulled toward this same bright
band instead of the true OPL/ONL transition, across three prior attempts and a
fourth one here that explicitly excluded it as a hard boundary. EZC itself,
though, is reliably detectable -- unlike OPL/ONL, it has no competing candidate
of similar strength nearby, so a straightforward "find the brightest band
strictly between IPL/INL and RPE" DP search is stable (max 3-10px frame-to-frame
jump on each of the 5 patients' central B-scan, vs. the 100+px jumps the OPL/ONL
attempt produced even with the same exclusion logic and a strong anchor penalty).

Used in v6 to give the avascular complex (AVC) slab a more anatomically correct
lower boundary: AVC is the outer nuclear layer + photoreceptors, which should
stop at the photoreceptor IS/OS junction (EZC), not extend down to the RPE
itself (the old RPE-1 boundary). AVC is a void reference regardless of exactly
where its lower edge sits, so this is a low-risk anatomical tightening, not a
repeat of the OPL/ONL attempt.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from . import segmentation
from .config import SegmentationConfig


def detect_ezc_bscan(struct_bscan: np.ndarray, ilm: np.ndarray, ipl_inl: np.ndarray,
                      rpe: np.ndarray, cfg: SegmentationConfig,
                      margin_above_ipl: int = 40, margin_below_rpe: int = 6,
                      max_step: int = 3) -> np.ndarray:
    """Brightest band strictly between IPL/INL+margin_above_ipl and RPE-margin_below_rpe,
    found by DP shortest path (same machinery as RPE detection) so it inherits
    smoothness across A-scans. ilm/ipl_inl/rpe here are this B-scan's own
    already-detected (and, for ilm, already bridged) boundaries.
    """
    smooth = gaussian_filter(struct_bscan.astype(np.float32), sigma=cfg.gauss_sigma_bscan)
    depth, width = smooth.shape
    offsets = np.arange(cfg.dmax)
    z_idx = np.clip(ilm[None, :] + offsets[:, None], 0, depth - 1).astype(np.int64)
    gathered = np.take_along_axis(smooth, z_idx, axis=0)

    ipl_rel = ipl_inl - ilm
    rpe_rel = rpe - ilm
    rows = np.arange(cfg.dmax)[:, None]
    cost = -gathered.copy()
    valid = (rows >= (ipl_rel[None, :] + margin_above_ipl)) & (rows <= (rpe_rel[None, :] - margin_below_rpe))
    cost[~valid] = 1e6
    ezc_rel = segmentation._dp_trace(cost, max_step=max_step, step_penalty=0.0)
    return ilm + ezc_rel


def detect_ezc_volume(structural: np.ndarray, bounds: segmentation.VolumeBoundaries,
                       cfg: SegmentationConfig) -> np.ndarray:
    """structural: (n_bscans, depth, width). Returns ezc: (n_bscans, width),
    cross-volume regularised the same way every other boundary in segmentation.py is."""
    n_bscans, width = bounds.ilm.shape
    ezc = np.empty((n_bscans, width))
    for y in range(n_bscans):
        ezc[y] = detect_ezc_bscan(structural[y], bounds.ilm[y], bounds.ipl_inl[y], bounds.rpe[y], cfg)
    ezc, _ = segmentation._regularize_surface(ezc, cfg)
    # keep ordering: IPL/INL < EZC < RPE after smoothing
    ezc = np.maximum(ezc, bounds.ipl_inl + 10)
    ezc = np.minimum(ezc, bounds.rpe - 2)
    return ezc
