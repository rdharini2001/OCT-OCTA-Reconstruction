"""CLI analysis of a paired research volume without the website.

Usage: python scripts/run_local.py --structural scan.zip [--flow flow.zip]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'backend'))
from app.io import load_volume
from app.analysis import segment_and_project


def main():
    p=argparse.ArgumentParser(description='OCTAVE research CLI (no diagnosis)')
    p.add_argument('--structural',required=True,type=Path)
    p.add_argument('--flow',type=Path)
    p.add_argument('--out',type=Path,default=Path('octave-analysis'))
    p.add_argument('--fov-x-mm',type=float)
    p.add_argument('--fov-y-mm',type=float)
    p.add_argument('--axial-um-per-px',type=float)
    p.add_argument('--preview-slices',type=int,default=0)
    a=p.parse_args()
    if (a.fov_x_mm is None)!=(a.fov_y_mm is None):p.error('Specify both field-of-view dimensions or neither')
    if a.preview_slices != 0 and a.preview_slices < 2:p.error('Preview must contain at least two B-scans')
    struct,original=load_volume(a.structural,'structural',slice_limit=a.preview_slices,return_count=True)
    flow,flow_orig=load_volume(a.flow,'flow',slice_limit=a.preview_slices,return_count=True) if a.flow else (None,None)
    if flow is not None and (flow.shape!=struct.shape or flow_orig!=original):p.error('Flow and structural source dimensions do not match')
    m=segment_and_project(struct,flow,a.out,a.fov_x_mm,a.fov_y_mm,a.axial_um_per_px)
    m.update({'original_slices':original,'sampled_slices':struct.shape[0],'preview_sampling':original!=struct.shape[0]})
    (a.out/'cli_metadata.json').write_text(json.dumps(m,indent=2))
    print(json.dumps({'out':str(a.out),'layers':m['layers'],'qc':m['qc']['any_flagged']},indent=2))

if __name__=='__main__':main()
