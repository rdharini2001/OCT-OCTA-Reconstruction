"""FAZ v2: the avascular hole in the already-computed, thresholded SVC vessel mask,
near the foveal seed -- not a separate edge-detection pipeline on the grayscale image.

Method: morphologically close the vessel mask (merges nearby capillaries into a
solid mesh without permanently growing vessels) so the background splits into the
true enclosed FAZ plus the large exterior background, then take the connected
non-vessel component at the foveal seed. Tries closing radius 5px first, falls back
to 3px, and is marked not-detected if neither gives a component that (a) exists at
the seed, (b) doesn't touch the image border (a border touch means the perifoveal
arcade wasn't actually closed), and (c) is a plausible size (0.3%-12% of the image).

Seed: same foveal-pit-or-centre logic as seg_v5_final.py.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi

from octa_pipeline import io_utils, segmentation
from octa_pipeline.config import PipelineConfig

OUT = Path("results_v6")
PIDS = [1, 2, 3, 4, 5]
MAX_SEED_OFFSET = 60
MIN_FRAC, MAX_FRAC = 0.003, 0.12
RADII = [5, 3]


def fovea_seed(pid, cfg, h, w):
    p = io_utils.load_patient(".", pid)
    b = segmentation.segment_volume(p.structural, cfg.segmentation)
    sm = ndi.gaussian_filter(b.rpe - b.ilm, 15)
    r0, r1, c0, c1 = int(h * 0.2), int(h * 0.8), int(w * 0.2), int(w * 0.8)
    iy, ix = np.unravel_index(np.argmin(sm[r0:r1, c0:c1]), (r1 - r0, c1 - c0))
    fy, fx = iy + r0, ix + c0
    if np.hypot(fy - h / 2, fx - w / 2) <= MAX_SEED_OFFSET:
        return int(fy), int(fx), "thickness minimum"
    return h // 2, w // 2, "image centre (thickness minimum too far)"


def faz_from_vessel_mask(vessel_mask: np.ndarray, sy: int, sx: int):
    h, w = vessel_mask.shape
    for r in RADII:
        closed = ndi.binary_closing(vessel_mask, structure=np.ones((2 * r + 1, 2 * r + 1)))
        non_vessel = ~closed
        lab, n = ndi.label(non_vessel)
        seed_lab = lab[sy, sx]
        if seed_lab == 0:
            continue
        comp = lab == seed_lab
        touches_border = comp[0, :].any() or comp[-1, :].any() or comp[:, 0].any() or comp[:, -1].any()
        frac = comp.mean()
        if touches_border or not (MIN_FRAC <= frac <= MAX_FRAC):
            continue
        return comp, r
    return np.zeros((h, w), dtype=bool), None


def main():
    rows = []
    for pid in PIDS:
        d = OUT / f"patient_{pid}"
        svc = cv2.imread(str(d / "svc_enface_v5.png"), 0)
        h, w = svc.shape
        vessel_mask = cv2.imread(str(d / "svc_vessels_model.png"), 0) > 127
        sy, sx, src = fovea_seed(pid, PipelineConfig(), h, w)
        comp, radius_used = faz_from_vessel_mask(vessel_mask, sy, sx)
        detected = radius_used is not None

        mask_u8 = (comp.astype(np.uint8)) * 255
        cv2.imwrite(str(d / "faz_mask_v2.png"), mask_u8)

        ov = cv2.cvtColor(svc, cv2.COLOR_GRAY2BGR)
        if detected:
            cnts, _ = cv2.findContours(comp.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(ov, cnts, -1, (255, 200, 0), 2)
        cv2.circle(ov, (sx, sy), 4, (0, 0, 255), -1)
        cv2.imwrite(str(d / "faz_overlay_v2.png"), ov)

        area_px = int(comp.sum())
        area_mm2 = round(area_px * (3 * 3) / (h * w), 3)  # still assumes unconfirmed 3x3mm FOV
        rows.append({"patient": pid, "seed_source": src, "seed_x": sx, "seed_y": sy,
                     "closing_radius_px": radius_used, "faz_detected": detected,
                     "faz_area_px": area_px, "faz_area_mm2_assuming_3mm_fov": area_mm2})
        print(rows[-1])

    with open(OUT / "faz_summary_v2.csv", "w", newline="") as fh:
        w_ = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w_.writeheader()
        w_.writerows(rows)


if __name__ == "__main__":
    main()
