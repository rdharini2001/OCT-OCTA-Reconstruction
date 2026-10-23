"""Optional Choroidalyzer adapter: use real model weights if installed."""
from __future__ import annotations
from functools import lru_cache
import numpy as np
from scipy import ndimage as ndi

@lru_cache(maxsize=1)
def _model():
    try:
        import torch
        from choroidalyze import Choroidalyzer
    except ImportError as exc:
        raise RuntimeError('Choroidalyzer extra not installed. Install its package and pretrained weights, or set OCTAVE_CHOROID=proxy.') from exc
    return Choroidalyzer(device='cuda' if torch.cuda.is_available() else 'cpu')

def model_boundaries(structural:np.ndarray,rpe:np.ndarray,fallback_top:np.ndarray,fallback_bottom:np.ndarray,threshold:float=.5):
    model=_model()
    n,depth,width=structural.shape
    top=np.empty((n,width),np.float32);bottom=np.empty((n,width),np.float32)
    fallback=0
    for i in range(n):
        pad_h=(-depth)%128;pad_w=(-width)%128
        pad=np.pad(structural[i],((0,pad_h),(0,pad_w)),mode='reflect')
        prediction=model.predict(pad[None,:,:])
        if hasattr(prediction,'detach'):prediction=prediction.detach()
        if hasattr(prediction,'cpu'):prediction=prediction.cpu()
        mask=np.asarray(prediction)[0,:depth,:width]>threshold
        mask=ndi.binary_opening(mask,structure=np.ones((3,3),dtype=bool))
        for x in range(width):
            depths=np.flatnonzero(mask[:,x] & (np.arange(depth)>=int(rpe[i,x])-5))
            if len(depths):top[i,x]=depths.min();bottom[i,x]=depths.max()
            else:
                top[i,x]=fallback_top[i,x];bottom[i,x]=fallback_bottom[i,x];fallback+=1
    top=ndi.median_filter(top,size=(1,9),mode='nearest')
    bottom=np.maximum(ndi.median_filter(bottom,size=(1,9),mode='nearest'),top+5)
    return top,bottom,fallback/(n*width)
