"""Untrusted OCT volume ingestion. Native axes are (B-scan, depth, A-scan)."""
from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

MAX_ZIP_BYTES = 1_250_000_000
MAX_FILES = 2500
MAX_VOXELS = 650_000_000
ALLOWED = {'.zip', '.npy', '.npz', '.tif', '.tiff'}


class DataError(ValueError):
    pass


def _validate(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 3:
        raise DataError(f'Expected a 3D volume (B-scans, depth, width), got shape {arr.shape}.')
    n, depth, width = arr.shape
    if n < 2 or depth < 48 or width < 48:
        raise DataError('Volume needs at least 2 B-scans and minimum depth/width of 48 pixels.')
    if n * depth * width > MAX_VOXELS:
        raise DataError('Volume exceeds configured voxel safety limit.')
    if not np.issubdtype(arr.dtype, np.number):
        raise DataError('Volume must contain numeric intensities.')
    if not np.all(np.isfinite(arr)):
        raise DataError('Volume contains NaN or infinite intensities.')
    if arr.dtype == np.uint8:
        return np.ascontiguousarray(arr)
    lo, hi = np.percentile(arr, [0.1, 99.9])
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    return np.ascontiguousarray((np.clip((arr.astype(np.float32) - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8))


def _natural_key(text: str):
    return [(1, int(c)) if c.isdigit() else (0, c.lower()) for c in re.split(r'(\d+)', Path(text).name)]


def _load_zip(path: Path, slice_limit: int = 0) -> tuple[np.ndarray, int]:
    with zipfile.ZipFile(path) as z:
        members = [x for x in z.infolist() if not x.is_dir() and x.filename.lower().endswith(('.png', '.tif', '.tiff')) and not Path(x.filename).name.startswith('.')]
        if not members:
            raise DataError('ZIP must contain a numbered stack of PNG/TIFF B-scan images.')
        if len(members) > MAX_FILES or sum(x.file_size for x in members) > MAX_ZIP_BYTES:
            raise DataError('ZIP exceeds frame count or uncompressed size limit.')
        base_names = [Path(x.filename).name.lower() for x in members]
        if len(set(base_names)) != len(base_names):
            raise DataError('ZIP has repeated frame filenames across directories; provide one unambiguous stack.')
        members.sort(key=lambda x: _natural_key(x.filename))
        original_slices = len(members)
        # Decode *only* the selected preview frames. A 448-slice scan should not
        # require keeping 448 full-resolution frames in memory just to show 128.
        if slice_limit and original_slices > slice_limit:
            selected = np.linspace(0, original_slices - 1, slice_limit, dtype=np.int64)
            members = [members[int(i)] for i in selected]
        frames = []
        expected = None
        for entry in members:
            try:
                with Image.open(io.BytesIO(z.read(entry))) as src:
                    frame = np.array(src.convert('L'))
            except (UnidentifiedImageError, OSError, ValueError) as exc:
                raise DataError(f'Could not read frame {entry.filename}') from exc
            if expected is None:
                expected = frame.shape
                if len(members) * expected[0] * expected[1] > MAX_VOXELS:
                    raise DataError('Decompressed volume exceeds voxel safety limit.')
            if frame.shape != expected:
                raise DataError(f'Frame size mismatch: {entry.filename} is {frame.shape}, expected {expected}.')
            frames.append(frame)
        return _validate(np.stack(frames)), original_slices


def load_volume(
    path: str | Path, key: str | None = None, *,
    slice_limit: int = 0, return_count: bool = False,
) -> np.ndarray | tuple[np.ndarray, int]:
    """Read a 3D OCT volume, optionally subsampling for an exploratory preview.

    With return_count=True, also return the original frame count. Sampling is
    deterministic and aligned across separately loaded paired volumes. The
    full volume must still be used for quantitative scan-level comparisons.
    """
    if slice_limit < 0 or (0 < slice_limit < 2):
        raise DataError('slice_limit must be zero or at least two B-scans')
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in ALLOWED:
        raise DataError(f'Unsupported input {suffix}. Use ZIP image stack, NPY, NPZ or TIFF stack.')
    if suffix == '.zip':
        arr, original_slices = _load_zip(path, slice_limit)
        return (arr, original_slices) if return_count else arr
    if suffix == '.npy':
        # mmap protects memory when opening a large, uncompressed NPY source.
        arr = np.load(path, allow_pickle=False, mmap_mode='r')
    elif suffix == '.npz':
        with np.load(path, allow_pickle=False) as archive:
            names = list(archive.files)
            selected = key if key in names else next((x for x in names if x.lower() in ('structural', 'flow', 'volume', 'data')), None)
            selected = selected or (names[0] if len(names) == 1 else None)
            if selected is None:
                raise DataError(f'NPZ has several arrays ({names}); name desired array "structural" or "flow".')
            arr = archive[selected]
    else:
        import tifffile
        arr = tifffile.imread(path)
    original_slices = int(arr.shape[0]) if arr.ndim > 0 else 0
    if slice_limit and arr.ndim == 3 and original_slices > slice_limit:
        ids = np.linspace(0, original_slices - 1, slice_limit, dtype=np.int64)
        arr = arr[ids]
    arr = _validate(arr)
    return (arr, original_slices) if return_count else arr


def select_slices(arr: np.ndarray, limit: int) -> tuple[np.ndarray, int]:
    """Explicit preview decimation; preserve full volume when limit=0."""
    if limit <= 0 or arr.shape[0] <= limit:
        return arr, 1
    ids = np.linspace(0, arr.shape[0] - 1, limit, dtype=int)
    return arr[ids], arr.shape[0]
