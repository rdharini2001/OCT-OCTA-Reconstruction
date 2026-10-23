"""Vessel morphometric feature extraction, SVC only.

Scoped to SVC deliberately: it's the only slab with an externally-validated
vessel mask (aiforvision/OCTA-autosegmentation, cross-checked against this
pipeline's own density metric at Pearson r=0.996 -- see results_v5/METHOD.md).
DVC was excluded from that same external validation earlier in this project
because the model's training domain is superficial-only; computing vessel
morphometrics from an un-validated DVC mask would be inconsistent with that
decision, so this script does not attempt it.

Features, all computed from the binary SVC vessel mask already in
results_v6/patient_<N>/svc_vessels_model.png:

  vessel_density_pct       vessel pixels / image area
  skeleton_density          skeleton pixels / image area (vessel length proxy)
  mean_caliber_px           vessel area / skeleton length (an average-width proxy,
                             standard in OCTA morphometry papers)
  branchpoint_density       skeleton junctions (>=3 neighbours) per 10,000 px^2
  endpoint_density          skeleton free ends (1 neighbour) per 10,000 px^2,
                             a proxy for vessel dropout/fragmentation
  fractal_dimension         box-counting dimension of the skeleton (network
                             space-filling complexity)
  mean_tortuosity           mean(path length / straight-line distance) over
                             skeleton segments between junctions/endpoints,
                             length-weighted, segments <5px excluded as noise
  n_segments                number of skeleton segments contributing to tortuosity

FAZ features, from faz_mask_v2.png (see results_v6/METHOD.md for that method):
  faz_area_px, faz_perimeter_px, faz_circularity (4*pi*area/perimeter^2, 1.0 = perfect
  circle), faz_acircularity (perimeter / perimeter of an equal-area circle, 1.0 = circle)

No physical (mm) units are reported for anything except where results_v6 already
established the 3x3mm-FOV caveat for FAZ area -- these are pixel-space, relative
metrics for comparing the 5 patients against each other, not calibrated absolutes.
"""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

OUT = Path("results_v6")
PIDS = [1, 2, 3, 4, 5]


def box_count_fractal_dim(skel: np.ndarray, sizes=(2, 3, 4, 6, 8, 12, 16, 24, 32)) -> float:
    h, w = skel.shape
    counts = []
    used_sizes = []
    for s in sizes:
        if s >= min(h, w):
            continue
        nh, nw = h // s, w // s
        cropped = skel[: nh * s, : nw * s]
        blocks = cropped.reshape(nh, s, nw, s)
        occupied = blocks.any(axis=(1, 3))
        c = int(occupied.sum())
        if c > 0:
            counts.append(c)
            used_sizes.append(s)
    if len(counts) < 3:
        return float("nan")
    log_inv_s = np.log(1.0 / np.array(used_sizes, dtype=float))
    log_n = np.log(np.array(counts, dtype=float))
    slope, _ = np.polyfit(log_inv_s, log_n, 1)
    return float(slope)


def neighbor_count(skel: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3))
    kernel[1, 1] = 0
    return ndi.convolve(skel.astype(np.uint8), kernel.astype(np.uint8), mode="constant")


def segment_tortuosity(skel: np.ndarray, branch_mask: np.ndarray, min_len: int = 5):
    """Remove branch points, label remaining skeleton into segments, and for each
    compute path_length (pixel count) / straight_line distance between its two
    farthest-apart pixels. Returns (lengths, tortuosities)."""
    trimmed = skel & ~branch_mask
    lab, n = ndi.label(trimmed, structure=np.ones((3, 3)))
    lengths, tort = [], []
    for i in range(1, n + 1):
        ys, xs = np.nonzero(lab == i)
        plen = len(ys)
        if plen < min_len:
            continue
        pts = np.stack([ys, xs], axis=1).astype(float)
        # farthest pair via convex-hull-free approximate: use pairwise on a subsample if large
        if len(pts) > 60:
            idx = np.linspace(0, len(pts) - 1, 60).astype(int)
            sub = pts[idx]
        else:
            sub = pts
        d = np.sqrt(((sub[:, None, :] - sub[None, :, :]) ** 2).sum(-1))
        straight = d.max()
        if straight < 1e-6:
            continue
        lengths.append(plen)
        tort.append(plen / straight)
    return np.array(lengths), np.array(tort)


def faz_shape_features(faz_mask: np.ndarray) -> dict:
    area = int(faz_mask.sum())
    if area == 0:
        return {"faz_area_px": 0, "faz_perimeter_px": 0.0, "faz_circularity": float("nan"),
                "faz_acircularity": float("nan")}
    cnts, _ = cv2.findContours(faz_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    perim = float(cv2.arcLength(cnts[0], True)) if cnts else float("nan")
    circularity = (4 * np.pi * area) / (perim ** 2) if perim > 0 else float("nan")
    equal_area_circle_perim = 2 * np.sqrt(np.pi * area)
    acircularity = perim / equal_area_circle_perim if perim > 0 else float("nan")
    return {"faz_area_px": area, "faz_perimeter_px": round(perim, 1),
            "faz_circularity": round(float(circularity), 3),
            "faz_acircularity": round(float(acircularity), 3)}


def extract_patient(pid: int) -> dict:
    d = OUT / f"patient_{pid}"
    vessel = cv2.imread(str(d / "svc_vessels_model.png"), 0) > 127
    faz = cv2.imread(str(d / "faz_mask_v2.png"), 0) > 0

    h, w = vessel.shape
    area_px = h * w
    skel = skeletonize(vessel)
    nbrs = neighbor_count(skel)
    branch = skel & (nbrs >= 3)
    endpoints = skel & (nbrs == 1)

    skel_len = int(skel.sum())
    vessel_area = int(vessel.sum())
    mean_caliber = vessel_area / skel_len if skel_len > 0 else float("nan")

    lengths, tort = segment_tortuosity(skel, branch)
    mean_tort = float(np.average(tort, weights=lengths)) if len(lengths) else float("nan")

    feats = {
        "patient": pid,
        "vessel_density_pct": round(100 * vessel_area / area_px, 2),
        "skeleton_density_per_1k_px": round(1000 * skel_len / area_px, 3),
        "mean_caliber_px": round(mean_caliber, 2),
        "branchpoint_density_per_10k_px": round(10000 * int(branch.sum()) / area_px, 3),
        "endpoint_density_per_10k_px": round(10000 * int(endpoints.sum()) / area_px, 3),
        "fractal_dimension": round(box_count_fractal_dim(skel), 4),
        "mean_tortuosity": round(mean_tort, 4),
        "n_segments": int(len(lengths)),
    }
    feats.update(faz_shape_features(faz))
    return feats


def main():
    rows = [extract_patient(pid) for pid in PIDS]
    with open(OUT / "vessel_morphometrics.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)
    print("wrote results_v6/vessel_morphometrics.csv")


if __name__ == "__main__":
    main()
