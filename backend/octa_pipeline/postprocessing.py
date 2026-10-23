"""Enface post-processing: contrast normalization, enhancement and simple
quantitative metrics (vessel density) used for the clinical-style outputs.
"""

from __future__ import annotations

import cv2
import numpy as np
from skimage.filters import frangi, threshold_otsu

from .config import VisualizationConfig


def normalize_to_uint8(img: np.ndarray, low_pct: float = 1.0, high_pct: float = 99.5) -> np.ndarray:
    """Percentile-clip and rescale a float enface image to 8-bit for display."""
    lo, hi = np.percentile(img, [low_pct, high_pct])
    if hi <= lo:
        hi = lo + 1.0
    clipped = np.clip(img, lo, hi)
    scaled = (clipped - lo) / (hi - lo) * 255.0
    return scaled.astype(np.uint8)


def enhance_clahe(img_uint8: np.ndarray, cfg: VisualizationConfig) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=cfg.clahe_clip_limit, tileGridSize=cfg.clahe_tile_grid)
    return clahe.apply(img_uint8)


def enhance_vesselness(img_uint8: np.ndarray, cfg: VisualizationConfig) -> np.ndarray:
    """Blend the CLAHE-enhanced enface image with a multiscale vesselness
    (Frangi) response computed on it.

    Motivation: patients with a lower raw signal baseline in their Flow
    data present with a cleaner, higher-contrast image (a smaller fraction
    of the enface is bright) than a patient whose Flow data happens to have
    a much higher overall background level -- confirmed directly by
    measuring each patient's background/peak signal percentiles, not just
    eyeballing the images. A uniform contrast stretch (e.g. log-compressing
    everyone to look equally "full") was tried and rejected: it mostly
    amplified background speckle, not real capillaries, on the
    already-higher-contrast patients.

    Frangi vesselness responds specifically to elongated, tubular
    structures rather than uniformly boosting brightness, so blending it
    in reveals fine capillaries that a flat contrast adjustment could not,
    without indiscriminately amplifying noise. Applied identically to every
    patient (not selectively to the "worse-looking" ones), so real
    differences between patients remain visible rather than papered over.
    """
    normed = img_uint8.astype(np.float64) / 255.0
    vesselness = frangi(normed, sigmas=cfg.vesselness_sigmas, black_ridges=False)
    vesselness_u8 = normalize_to_uint8(vesselness)
    blend = (
        img_uint8.astype(np.float64) * (1 - cfg.vesselness_blend)
        + vesselness_u8.astype(np.float64) * cfg.vesselness_blend
    )
    return np.clip(blend, 0, 255).astype(np.uint8)


def apply_display_gamma(img_uint8: np.ndarray, cfg: VisualizationConfig) -> np.ndarray:
    """Brighten a dim enface image toward a target median brightness with a
    per-image-calibrated gamma curve, for presentation only.

    Motivation: compared against a reference high-quality wide-field OCTA
    image, this pipeline's enface images sit at roughly half the reference's
    median brightness (and the gap holds across the whole distribution, not
    just the extremes) -- confirmed by direct percentile comparison, not
    just eyeballing. A *fixed* gamma was tried first and rejected: patients
    already close to the target (e.g. Patient 3, whose Flow data has a
    naturally higher baseline -- see `enhance_vesselness`) got visibly
    washed out by the same correction that helped a dimmer patient. Solving
    for the gamma that maps each image's own current median to the target
    median, per image, brightens exactly the patients that need it and
    leaves an already-bright one alone (gamma clipped to <=1.0, i.e. this
    only ever brightens, never darkens).

    This is a display-only tone curve -- monotonic, so it reorders nothing
    -- applied to a *separate* output used only in the composite/cohort
    figures. It is deliberately not applied to the file used for vessel-
    density thresholding or fed to the external validation model, so
    neither of those (already cross-checked) numbers change.
    """
    img = img_uint8.astype(np.float64)
    current_median = max(float(np.median(img)), 1.0)
    gamma = np.log(cfg.display_gamma_target_median / 255.0) / np.log(current_median / 255.0)
    gamma = float(np.clip(gamma, cfg.display_gamma_min, 1.0))
    adjusted = 255.0 * (img / 255.0) ** gamma
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def vessel_density(img_uint8: np.ndarray, cfg: VisualizationConfig) -> tuple[float, np.ndarray]:
    """Binarize the enface image (Otsu, scaled by a configurable fraction)
    and return (vessel area fraction in %, binary mask)."""
    try:
        otsu = threshold_otsu(img_uint8)
    except ValueError:
        otsu = 0
    thresh = otsu * cfg.vessel_density_threshold
    mask = img_uint8 > thresh
    density_pct = 100.0 * mask.mean()
    return density_pct, mask
