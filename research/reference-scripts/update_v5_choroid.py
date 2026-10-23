"""Replace the v5 choroid slab (pixel-offset heuristic) with Choroidalyzer-derived
real anatomical boundaries. Old output is kept as choroid_enface_v5_pixeloffset.png
for comparison; the new one becomes the primary choroid_enface_v5.png.
Updates results_v5/patient_<N>/metrics_v5.json and writes results_v5/choroid_upgrade_summary.csv.
"""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from octa_pipeline import io_utils, projection, segmentation, v2, v3
from octa_pipeline.config import PipelineConfig
import choroidalyzer_boundaries as cb

OUT = Path("results_v5")
cfg = PipelineConfig()
v2cfg = v2.V2Config()
rows = []

for pid in range(1, 6):
    d = OUT / f"patient_{pid}"
    old = d / "choroid_enface_v5.png"
    backup = d / "choroid_enface_v5_pixeloffset.png"
    if old.exists() and not backup.exists():
        shutil.copy(old, backup)

    pdata = io_utils.load_patient(".", pid)
    b = segmentation.segment_volume(pdata.structural, cfg.segmentation)
    extra = v2.build_extra_slabs(b, pdata.flow.shape[1], v2cfg)

    top, bottom, fallback_frac = cb.choroid_region_boundaries(
        pdata.structural, b.rpe, extra.choroid_top, extra.choroid_bottom
    )

    flow_ds = projection.despeckle_flow(pdata.flow, cfg.projection.flow_median_ksize)
    bulk = projection.estimate_bulk_motion_floor(flow_ds, b.opl_onl, b.rpe, cfg.projection.bulk_motion_ref_margin)
    flow_c = projection.suppress_bulk_motion(flow_ds, bulk, cfg.projection.bulk_motion_baseline_percentile)
    ch_f = projection._mean_projection(flow_c, top, bottom)
    new_img = v2.enhance_v2(v3.row_flat_field(ch_f), v2cfg)
    Image.fromarray(new_img).save(old)

    old_img = np.array(Image.open(backup))
    rows.append({
        "patient": pid,
        "fallback_fraction": round(fallback_frac, 4),
        "mean_thickness_px": round(float((bottom - top).mean()), 1),
        "banding_old": round(v3.banding_index(old_img), 3),
        "banding_new": round(v3.banding_index(new_img), 3),
    })
    print(rows[-1])

    mpath = d / "metrics_v5.json"
    m = json.loads(mpath.read_text())
    m["enface"]["choroid"]["boundary_source"] = "Choroidalyzer (justinengelmann/Choroidalyzer)"
    m["enface"]["choroid"]["fallback_to_pixel_offset_fraction"] = round(fallback_frac, 4)
    m["enface"]["choroid"]["mean_thickness_px"] = round(float((bottom - top).mean()), 1)
    m["enface"]["choroid"]["median"] = int(np.median(new_img))
    m["enface"]["choroid"]["banding_index"] = round(v3.banding_index(new_img), 3)
    mpath.write_text(json.dumps(m, indent=2))

with open(OUT / "choroid_upgrade_summary.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print("done")
