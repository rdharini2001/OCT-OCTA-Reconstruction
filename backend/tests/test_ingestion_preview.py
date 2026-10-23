"""Volume import is deliberately usable across scanner-export ZIP naming patterns."""
from __future__ import annotations

import io
from pathlib import Path
import zipfile

import numpy as np
import pytest
from PIL import Image
from app.io import DataError, load_volume


def test_preview_zip_only_reads_selected_frames(tmp_path: Path):
    path = tmp_path / 'retina.zip'
    with zipfile.ZipFile(path, 'w') as z:
        for i in range(12):
            frame = np.full((48, 60), i * 20, dtype=np.uint8)
            buf = io.BytesIO()
            Image.fromarray(frame).save(buf, format='PNG')
            z.writestr(f'frame.{i + 1:04}.png', buf.getvalue())
    volume, count = load_volume(path, 'structural', slice_limit=4, return_count=True)
    assert count == 12
    assert volume.shape == (4, 48, 60)
    np.testing.assert_array_equal(volume[:, 0, 0], [0, 60, 140, 220])
    whole, count = load_volume(path, 'structural', return_count=True)
    assert whole.shape == (12, 48, 60)
    assert count == 12


def test_numeric_npy_preview_and_source_count(tmp_path: Path):
    path = tmp_path / 'volume.npy'
    np.save(path, np.arange(9 * 48 * 50, dtype=np.uint16).reshape(9, 48, 50))
    preview, n = load_volume(path, slice_limit=3, return_count=True)
    assert preview.shape == (3, 48, 50)
    assert n == 9


def test_repeated_frame_names_are_rejected(tmp_path: Path):
    path = tmp_path / 'ambiguous.zip'
    frame = io.BytesIO()
    Image.fromarray(np.zeros((48, 48), dtype=np.uint8)).save(frame, 'PNG')
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr('scan-a/frame.0001.png', frame.getvalue())
        z.writestr('scan-b/frame.0001.png', frame.getvalue())
    with pytest.raises(DataError, match='repeated frame filenames'):
        load_volume(path)
