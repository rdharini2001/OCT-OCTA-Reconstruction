"""Loading OCT/OCTA volumes from the delivered per-patient zip archives.

Each zip contains a stack of 8-bit PNG B-scans named ``image.NNNN.png``. The
stack index (file order) is the slow (B-scan / elevational, "y") axis; each
PNG itself is (depth "z" x width "x"). The loaded volume is returned with
shape (y, z, x) i.e. (n_bscans, depth, width).
"""

from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

_NUM_RE = re.compile(r"(\d+)")


def _sort_key(name: str) -> int:
    m = _NUM_RE.search(name)
    return int(m.group(1)) if m else -1


@dataclass
class Volume:
    data: np.ndarray  # (y, z, x) uint8
    names: list

    @property
    def shape(self):
        return self.data.shape


def load_volume_from_zip(zip_path: str | Path) -> Volume:
    """Read every PNG in the zip (in slice order) into a single 3D uint8 array."""
    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        names = sorted(
            [n for n in zf.namelist() if n.lower().endswith(".png")], key=_sort_key
        )
        if not names:
            raise ValueError(f"No PNG slices found in {zip_path}")

        first = np.array(Image.open(zf.open(names[0])).convert("L"))
        depth, width = first.shape
        vol = np.empty((len(names), depth, width), dtype=np.uint8)
        vol[0] = first
        for i, name in enumerate(names[1:], start=1):
            with zf.open(name) as fh:
                im = Image.open(fh).convert("L")
                vol[i] = np.array(im)
    return Volume(data=vol, names=names)


@dataclass
class PatientData:
    patient_id: int
    flow: np.ndarray       # (y, z, x) uint8 -- decorrelation / flow signal
    structural: np.ndarray  # (y, z, x) uint8 -- denoised structural OCT


def load_patient(data_dir: str | Path, patient_id: int) -> PatientData:
    data_dir = Path(data_dir)
    flow_zip = data_dir / f"Flow-{patient_id}.zip"
    struct_zip = data_dir / f"OctDenoise-{patient_id}.zip"
    if not flow_zip.exists():
        raise FileNotFoundError(flow_zip)
    if not struct_zip.exists():
        raise FileNotFoundError(struct_zip)

    flow_vol = load_volume_from_zip(flow_zip)
    struct_vol = load_volume_from_zip(struct_zip)

    if flow_vol.shape != struct_vol.shape:
        raise ValueError(
            f"Patient {patient_id}: Flow shape {flow_vol.shape} != "
            f"OctDenoise shape {struct_vol.shape}"
        )

    return PatientData(patient_id=patient_id, flow=flow_vol.data, structural=struct_vol.data)
