"""Replace the pixel-offset choroid lower boundary with Choroidalyzer's segmented
choroid region (RPE/Bruch's membrane to chorio-scleral junction), run per B-scan
across the whole volume.

Choroidalyzer (justinengelmann/Choroidalyzer, MIT-licensed, pretrained U-Net on
5600 B-scans across 3 device types) segments the choroid region directly from the
structural B-scan -- real anatomy, not a fixed offset below RPE.

Fallback: columns where Choroidalyzer's region mask is empty keep the old
RPE+16..+136 px heuristic, so a failed column never produces a missing boundary.
"""
from __future__ import annotations

import sys
import os
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

# Optional external Choroidalyzer checkout; configure rather than hard-code local user paths.
_external = os.environ.get('CHOROIDALYZER_SOURCE')
if _external:
    sys.path.insert(0, str(Path(_external)))
from choroidalyze import Choroidalyzer  # noqa: E402

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = Choroidalyzer(device="cuda")
    return _model


def choroid_region_boundaries(structural: np.ndarray, rpe: np.ndarray,
                               fallback_top: np.ndarray, fallback_bottom: np.ndarray,
                               region_thresh: float = 0.5) -> tuple[np.ndarray, np.ndarray, float]:
    """structural: (n_bscans, depth, width) uint8. Returns (top, bottom, fallback_fraction)."""
    model = _get_model()
    n, depth, width = structural.shape
    pad_h = (-depth) % 128
    pad_w = (-width) % 128

    top = np.empty((n, width))
    bottom = np.empty((n, width))
    n_fallback = 0

    for y in range(n):
        img = structural[y]
        padded = np.pad(img, ((0, pad_h), (0, pad_w)), mode="reflect")
        preds = model.predict(padded[None, :, :]).numpy()
        region = preds[0, :depth, :width] > region_thresh
        region = ndi.binary_opening(region, structure=np.ones((3, 3)))

        # restrict to below RPE, since the region mask can pick up noise above it
        rpe_y = rpe[y].astype(int)
        t = np.full(width, -1.0)
        b = np.full(width, -1.0)
        ys_idx = np.where(region)[0]
        if len(ys_idx) == 0:
            top[y] = fallback_top[y]
            bottom[y] = fallback_bottom[y]
            n_fallback += width
            continue
        for x in range(width):
            col = np.where(region[:, x])[0]
            col = col[col > rpe_y[x] - 5]  # choroid must be at/below RPE
            if len(col) == 0:
                t[x] = -1
                b[x] = -1
            else:
                t[x] = col.min()
                b[x] = col.max()
        bad = t < 0
        n_fallback += int(bad.sum())
        t[bad] = fallback_top[y][bad]
        b[bad] = fallback_bottom[y][bad]
        top[y] = t
        bottom[y] = b

    # light regularisation: median filter across columns to drop single-column noise
    from scipy.ndimage import median_filter
    top = median_filter(top, size=(1, 9), mode="nearest")
    bottom = median_filter(bottom, size=(1, 9), mode="nearest")
    bottom = np.maximum(bottom, top + 5)
    fallback_fraction = n_fallback / (n * width)
    return top, bottom, fallback_fraction
