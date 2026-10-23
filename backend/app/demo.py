"""Deterministic *synthetic* OCT/OCTA phantom, not patient or ground-truth data."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from scipy.ndimage import gaussian_filter
from .analysis import normalize


def create_demo(root: Path) -> None:
    folder = root / 'synthetic-demo'
    if (folder / 'metadata.json').exists():
        return
    folder.mkdir(exist_ok=True, parents=True)
    rng = np.random.default_rng(1776)
    n, depth, width = 100, 260, 352
    xx = np.arange(width, dtype=np.float32)[None, :]
    yy = np.arange(n, dtype=np.float32)[:, None]
    pit = 22 * np.exp(-((xx-width/2)/(width*.21))**2 - ((yy-n/2)/(n*.28))**2)
    roll = 7 * np.sin(xx / width * np.pi*2) + 3 * np.cos(yy/n * np.pi*2)
    ilm = 55 + roll + pit
    ipl = ilm + 47 - 14 * np.exp(-((xx-width/2)/72)**2 - ((yy-n/2)/22)**2)
    opl = ipl + 36 + 2 * np.sin(xx / 27)
    rpe = 166 + roll*0.45 + 3*np.sin(xx/45)
    ezc = rpe - 18
    vol = np.zeros((n,depth,width),dtype='uint8')
    z = np.arange(depth, dtype=np.float32)[:,None]
    for j in range(n):
        a = (np.exp(-((z-ilm[j:j+1])/3.5)**2)*85
             + np.exp(-((z-(ilm[j:j+1]+13))/12)**2)*45
             + np.exp(-((z-ipl[j:j+1])/4.0)**2)*61
             + np.exp(-((z-opl[j:j+1])/4.0)**2)*72
             + np.exp(-((z-ezc[j:j+1])/3.0)**2)*79
             + np.exp(-((z-rpe[j:j+1])/6.0)**2)*165)
        tissue = (z >= ilm[j:j+1]) & (z <= rpe[j:j+1]+24)
        a += tissue.astype(np.float32)*32 + rng.normal(0,13,size=(depth,width))
        vol[j] = np.clip(a,0,255).astype('uint8')
    # 2D synthetic capillary phantom with a central circular vessel-free zone.
    mask = np.zeros((n,width),dtype='uint8')
    for side in (-1,1):
        for i in range(19):
            xs = np.arange(width)
            center = n/2 + side*(18+i*1.58) + 4.5*np.sin(xs/(15+i%5*4)+i)
            for x in range(0,width,2):
                y = int(center[x]); cv2.circle(mask,(x,y),1 if i%3 else 2,255,-1)
    for i in range(45):
        x0 = int(rng.integers(4,width-5))
        y0 = int(rng.integers(4,n-5))
        x1 = int(np.clip(x0+rng.normal(0,24),0,width-1)); y1 = int(np.clip(y0+rng.normal(0,11),0,n-1))
        cv2.line(mask,(x0,y0),(x1,y1),int(rng.integers(105,230)),1)
    yy0,xx0=np.ogrid[:n,:width]
    fovea=((xx0-width*.5)/(width*.095))**2+((yy0-n*.5)/(n*.26))**2 < 1
    mask[fovea]=0
    svc = normalize(gaussian_filter(mask.astype(float),.7)+rng.normal(7,5,size=mask.shape))
    # A deliberately enclosed synthetic FAZ ensures the demo exercises contour logic.
    cv2.ellipse(svc,(width//2,n//2),(40,16),0,0,360,255,5)
    cv2.ellipse(svc,(width//2,n//2),(33,10),0,0,360,0,-1)
    dvc = normalize(gaussian_filter(np.roll(mask,5,axis=0),1.4)*.72+rng.normal(13,12,size=mask.shape))
    avc = normalize(rng.normal(12,4,size=mask.shape))
    cc = normalize(gaussian_filter(rng.gamma(2.2,20,mask.shape),.85))
    ch = normalize(gaussian_filter(rng.gamma(2,17,mask.shape),1.6))
    images = {'structural': normalize(vol.mean(axis=1)), 'svc':svc,'dvc':dvc,'avc':avc,'cc':cc,'choroid':ch}
    np.save(folder/'structural.npy',vol)
    np.savez_compressed(folder/'boundaries.npz',ilm=ilm.astype('float32'),ipl_inl=ipl.astype('float32'),opl_onl=opl.astype('float32'),rpe=rpe.astype('float32'),ezc=ezc.astype('float32'))
    for key,img in images.items(): Image.fromarray(img).save(folder/f'{key}.png')
    metadata = {
        'id':'synthetic-demo','name':'Synthetic retinal phantom','created_at':'2026-01-01T00:00:00Z',
        'source':'Synthetic demonstration (not a human patient)', 'synthetic':True, 'status':'ready',
        'shape':[n,depth,width], 'original_slices':n, 'sampled_slices':n,'preview_sampling':False,
        'flow_available':True,'layers':list(images),
        'calibration':{'fov_x_mm':None,'fov_y_mm':None,'axial_um_per_px':None},
        'retinal_thickness_px':round(float(np.median(rpe-ilm)),1),'retinal_thickness_um':None,
        'fovea_seed':{'x':width//2,'y':n//2,'source':'synthetic centre'},
        'qc':{'any_flagged':False,'ilm':{'flagged':False},'ipl_inl':{'flagged':False},'opl_onl':{'flagged':False},'rpe':{'flagged':False}},
        'warnings':['SYNTHETIC PHANTOM — for UI exploration and software tests; NOT patient data, measured vasculature or diagnostic reference.', 'Physical scales intentionally absent; only pixel measurements supported.'],
        'pipeline':'Deterministic synthetic phantom generator (no segmentation inference)',
    }
    (folder/'metadata.json').write_text(json.dumps(metadata,indent=2))
