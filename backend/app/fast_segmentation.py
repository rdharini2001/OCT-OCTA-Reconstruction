"""Fast vectorized heuristic retinal boundaries for *exploratory* volumes.

Device-general preview path; not a clinically validated retinal-layer model.
Outputs the same boundary dataclass as the user-supplied OCTA pipeline.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from octa_pipeline.segmentation import VolumeBoundaries


def _as_smooth(surface: np.ndarray, d: int) -> np.ndarray:
    a=ndi.median_filter(surface,size=(1,11),mode='nearest')
    return np.clip(ndi.gaussian_filter(a,sigma=(1.0,2.8),mode='nearest'),0,d-1)


def segment_fast(structural: np.ndarray) -> VolumeBoundaries:
    n,d,w=structural.shape
    ilm=np.empty((n,w),np.float32);rpe=np.empty((n,w),np.float32)
    xs=np.arange(w)
    for i in range(n):
        img=ndi.gaussian_filter(structural[i].astype(np.float32),sigma=(2.1,2.3))
        top=img[:max(12,min(d//9,45))]
        noise=np.median(top,axis=0)
        # Adaptive signal transition: require signal over multiple depth samples.
        cutoff=np.maximum(noise+14.0,14.0)
        candidate=ndi.uniform_filter1d((img>cutoff[None,:]).astype(np.float32),size=9,axis=0)>0.72
        candidate[int(d*.63):]=False
        if not np.any(candidate):
            raise ValueError('No detectable retinal intensity: verify scan orientation and modality.')
        first=np.argmax(candidate,axis=0)
        bad=~candidate.any(axis=0)
        # Guard against isolated floater/shadows using cross-column continuity.
        med=ndi.median_filter(first,size=31,mode='nearest')
        bad|=np.abs(first-med)>max(16,int(d*.09))
        first[bad]=med[bad]
        first=ndi.median_filter(first.astype(np.float32),size=11,mode='nearest')
        ilm[i]=first
        # Detect brightest deep band (RPE), more than 20% of depth below ILM.
        z=np.arange(d)[:,None]
        lower=first[None,:]+np.maximum(28,d*.18)
        allowed=(z>lower)&(z<d-6)
        deep=np.where(allowed,img,-np.inf)
        found=np.argmax(deep,axis=0)
        bad_r=~np.isfinite(deep[found,xs])
        if np.any(bad_r): found[bad_r]=np.minimum(d-7,first[bad_r]+max(35,int(d*.32)))
        rpe[i]=ndi.median_filter(found.astype(np.float32),size=15,mode='nearest')
    ilm=_as_smooth(ilm,d)
    rpe=_as_smooth(rpe,d)
    # Physically plausible layer ordering; depth-specific boundaries are proxies.
    thickness=np.maximum(rpe-ilm,16)
    ipl=_as_smooth(ilm+.34*thickness,d)
    opl=_as_smooth(ipl+.39*np.maximum(rpe-ipl,8),d)
    ipl=np.maximum(ipl,ilm+3);opl=np.maximum(opl,ipl+3);rpe=np.maximum(rpe,opl+3)
    rpe=np.clip(rpe,0,d-1)
    qc={}
    for k,s in (('ilm',ilm),('ipl_inl',ipl),('opl_onl',opl),('rpe',rpe)):
        jumps=np.abs(np.diff(s,axis=0)) if n>1 else np.zeros((1,w))
        max_jump=float(np.max(jumps))
        flagged=max_jump>max(22,d*.12)
        qc[k]={'outlier_fraction':0.0,'residual_frame_jump_px':round(max_jump,1),'flagged':bool(flagged)}
    very_thin=np.mean((rpe-ilm)<max(15,d*.07))
    qc['svp_thickness_implausible_fraction']=float(np.mean((ipl-ilm)<4))
    qc['dvp_thickness_implausible_fraction']=float(np.mean((opl-ipl)<4))
    qc['any_flagged']=bool(any(qc[k]['flagged'] for k in ('ilm','ipl_inl','opl_onl','rpe')) or very_thin>.03)
    qc['segmentation_method']='Fast intensity-derived boundaries; IPL/INL and OPL/ONL are geometric approximations, not validated tissue-class segmentations.'
    return VolumeBoundaries(ilm=ilm,ipl_inl=ipl,opl_onl=opl,rpe=rpe,qc=qc)
