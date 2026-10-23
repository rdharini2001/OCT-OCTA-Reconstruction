"""OCTAVE FastAPI research server. Local files never leave this server."""
from __future__ import annotations

import csv
import hashlib
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4
import zipfile

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
from PIL import Image
from pydantic import BaseModel

from .io import load_volume, DataError
from .analysis import segment_and_project, vessel_and_faz, metrics_overlay, png_bytes, png_bytes_rgb, LAYERS
from .demo import create_demo

DATA_ROOT = Path(os.environ.get('OCTAVE_DATA_DIR',str(Path(__file__).resolve().parents[2]/'runtime-data'))).resolve()
DATA_ROOT.mkdir(parents=True,exist_ok=True)
MAX_UPLOAD_MB = int(os.environ.get('OCTAVE_MAX_UPLOAD_MB','200'))
MAX_SLICES = int(os.environ.get('OCTAVE_PREVIEW_SLICES','128'))
executor=ThreadPoolExecutor(max_workers=1)
jobs: dict[str,dict] = {}
job_lock=threading.Lock()

app=FastAPI(title='OCTAVE Research API',version='0.1.0',description='Local, research-only OCT/OCTA processing and visualization. Not a medical device.')
app.add_middleware(CORSMiddleware,allow_origins=['http://localhost:5173','http://127.0.0.1:5173'],allow_credentials=False,allow_methods=['*'],allow_headers=['*'])


@app.on_event('startup')
def startup():
    create_demo(DATA_ROOT)


def dataset_dir(dataset_id: str) -> Path:
    if not dataset_id or any(x not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for x in dataset_id.lower()) or len(dataset_id)>80:
        raise HTTPException(400,'Invalid dataset identifier')
    d=DATA_ROOT/dataset_id
    if not (d/'metadata.json').is_file():
        raise HTTPException(404,'Dataset not found')
    return d


def read_dataset(dataset_id: str) -> dict:
    return json.loads((dataset_dir(dataset_id)/'metadata.json').read_text())


def update_job(job_id: str, **kw):
    with job_lock:
        jobs[job_id].update(kw)


def work(job_id: str, dataset_id: str, source: str, structural_path: Path, flow_path: Path | None, x_mm: float | None, y_mm: float | None, z_um: float | None, sample_limit: int):
    folder=DATA_ROOT/dataset_id
    def progress(value: float,msg: str): update_job(job_id,progress=round(value,3),stage=msg)
    try:
        input_hashes={'structural_sha256':hashlib.file_digest(structural_path.open('rb'),'sha256').hexdigest()}
        if flow_path: input_hashes['flow_sha256']=hashlib.file_digest(flow_path.open('rb'),'sha256').hexdigest()
        progress(.02,'Decoding structural volume')
        s,orig_n=load_volume(structural_path,'structural',slice_limit=sample_limit,return_count=True)
        progress(.07,'Decoding paired flow volume')
        f,flow_n=load_volume(flow_path,'flow',slice_limit=sample_limit,return_count=True) if flow_path else (None,None)
        if f is not None and (s.shape != f.shape or orig_n != flow_n):
            raise DataError(f'Structural {s.shape} ({orig_n} source frames) and flow {f.shape} ({flow_n} source frames) differ. Use a registered, paired volume.')
        progress(.12,'Running reconstruction pipeline')
        result=segment_and_project(s,f,folder,x_mm,y_mm,z_um,progress)
        meta={
            'id':dataset_id,'name':source[:120], 'created_at':datetime.now(timezone.utc).isoformat(),
            'source':'User-uploaded local OCT volume','synthetic':False,'status':'ready',
            'original_slices':int(orig_n),'sampled_slices':int(s.shape[0]), 'preview_sampling':orig_n!=s.shape[0],
            'provenance':{'input_hashes':input_hashes,'sampling_max_slices':sample_limit,'version':'0.1.0','app':'OCTAVE'},
            **result,
        }
        if meta['preview_sampling']:
            meta['warnings'].insert(0,f'Preview mode: only {s.shape[0]} / {orig_n} B-scans processed. Run full volume (sampling=0) for research quantification.')
        (folder/'metadata.json').write_text(json.dumps(meta,indent=2))
        update_job(job_id,status='complete',progress=1,stage='Analysis ready',dataset_id=dataset_id)
    except Exception as exc:
        update_job(job_id,status='failed',progress=1,stage='Processing failed',error=f'{type(exc).__name__}: {exc}')
    finally:
        for p in [structural_path,flow_path]:
            if p is not None:
                p.unlink(missing_ok=True)


async def save_upload(f: UploadFile, out: Path):
    suffix=Path(f.filename or '').suffix.lower()
    if suffix not in ('.zip','.npy','.npz','.tif','.tiff'):
        raise HTTPException(422,'Input must be a zip image stack, NPY, NPZ or TIFF stack')
    total=0
    try:
        with out.open('wb') as target:
            while chunk:=await f.read(1024*1024):
                total+=len(chunk)
                if total>MAX_UPLOAD_MB*1024*1024:
                    raise HTTPException(413,f'File exceeds {MAX_UPLOAD_MB} MB upload limit')
                target.write(chunk)
    except Exception:
        out.unlink(missing_ok=True)
        raise
    finally:
        await f.close()


@app.get('/api/health')
def health(): return {'status':'ok','service':'OCTAVE','version':'0.1.0','disclaimer':'Research use only'}


@app.get('/api/datasets')
def get_datasets():
    results=[]
    for f in DATA_ROOT.glob('*/metadata.json'):
        try: results.append(json.loads(f.read_text()))
        except (ValueError,OSError): continue
    return sorted(results,key=lambda d:d.get('created_at',''),reverse=True)


@app.post('/api/datasets',status_code=202)
async def create_dataset(
    structural:UploadFile=File(...),flow:UploadFile|None=File(None),
    name:str=Form('New OCT volume'),fov_x_mm:float|None=Form(None),fov_y_mm:float|None=Form(None),
    axial_um_per_px:float|None=Form(None),max_slices:int=Form(MAX_SLICES)
):
    if (fov_x_mm is None)!=(fov_y_mm is None):
        raise HTTPException(422,'Provide both lateral scan widths or neither')
    if any(v is not None and not (0<v<=60) for v in (fov_x_mm,fov_y_mm)):
        raise HTTPException(422,'Lateral FOV values must be in (0,60] mm')
    if axial_um_per_px is not None and not (0<axial_um_per_px<=50):
        raise HTTPException(422,'Axial scale must be in (0,50] micrometers/pixel')
    if max_slices<0 or (max_slices!=0 and not (16<=max_slices<=2500)):
        raise HTTPException(422,'max_slices must be 0 (full) or 16..2500')
    id=uuid4().hex[:16]
    folder=DATA_ROOT/id;folder.mkdir(parents=True)
    sp=folder/f'structural_upload{Path(structural.filename or ".zip").suffix.lower()}'
    fp=folder/f'flow_upload{Path(flow.filename or ".zip").suffix.lower()}' if flow else None
    try:
        await save_upload(structural,sp)
        if flow is not None and fp is not None: await save_upload(flow,fp)
    except Exception:
        for f in folder.glob('*'): f.unlink(missing_ok=True)
        folder.rmdir()
        raise
    job_id=uuid4().hex
    with job_lock:
        jobs[job_id]={'id':job_id,'dataset_id':id,'status':'processing','stage':'Queued','progress':0}
    executor.submit(work,job_id,id,name,sp,fp,fov_x_mm,fov_y_mm,axial_um_per_px,max_slices)
    return jobs[job_id]


@app.get('/api/jobs/{job_id}')
def get_job(job_id:str):
    if job_id not in jobs: raise HTTPException(404,'Job not found or server restarted')
    return jobs[job_id]


@app.get('/api/datasets/{dataset_id}')
def get_dataset(dataset_id:str): return read_dataset(dataset_id)


@app.get('/api/datasets/{dataset_id}/bscan/{index}')
def bscan(dataset_id:str,index:int,overlay:bool=Query(True)):
    d=dataset_dir(dataset_id)
    volume=np.load(d/'structural.npy',mmap_mode='r')
    if index<0 or index>=volume.shape[0]: raise HTTPException(404,'B-scan index out of range')
    raw=volume[index]
    if not overlay: return Response(content=png_bytes(raw),media_type='image/png')
    rgb=np.stack([raw,raw,raw],axis=2).copy()
    colors={'ilm':(255,125,134),'ipl_inl':(255,206,120),'opl_onl':(67,229,201),'rpe':(148,143,255),'ezc':(117,165,250)}
    with np.load(d/'boundaries.npz') as bd:
        for key,color in colors.items():
            if key not in bd: continue
            vals=np.clip(np.rint(bd[key][index]),0,raw.shape[0]-1).astype('int32')
            pts=np.stack([np.arange(raw.shape[1],dtype='int32'),vals],axis=1).reshape(-1,1,2)
            import cv2
            cv2.polylines(rgb,[pts],False,color,1,lineType=cv2.LINE_AA)
    return Response(content=png_bytes_rgb(rgb),media_type='image/png')


@app.get('/api/datasets/{dataset_id}/enface/{layer}')
def enface(dataset_id:str,layer:str):
    d=dataset_dir(dataset_id)
    if layer not in LAYERS or not (d/f'{layer}.png').is_file():raise HTTPException(404,'Layer unavailable for this acquisition')
    return FileResponse(d/f'{layer}.png',media_type='image/png')


@app.get('/api/datasets/{dataset_id}/analyze/{layer}')
def analyze(dataset_id:str,layer:str='svc',threshold:int=52,seed_x:int|None=None,seed_y:int|None=None):
    meta=read_dataset(dataset_id);d=dataset_dir(dataset_id)
    if not meta['flow_available']: raise HTTPException(422,'Structural-only dataset has no OCTA vascular analysis')
    if layer not in ('svc','dvc','cc'):raise HTTPException(422,'Vessel analysis is available for SVC, DVC and CC; FAZ is only reported for SVC')
    img=np.array(Image.open(d/f'{layer}.png').convert('L'))
    default_seed=meta.get('fovea_seed',{})
    if seed_x is None: seed_x=default_seed.get('x')
    if seed_y is None: seed_y=default_seed.get('y')
    try: metrics,vessel,faz=vessel_and_faz(img,threshold,seed_x,seed_y,meta['calibration']['fov_x_mm'],meta['calibration']['fov_y_mm'])
    except ValueError as exc:raise HTTPException(422,str(exc)) from exc
    if layer!='svc':
        for k in list(metrics):
            if k.startswith('faz_') or k=='closing_radius_px': metrics.pop(k,None)
        faz[:]=False
    metrics['layer']=layer
    return metrics


@app.get('/api/datasets/{dataset_id}/mask/{layer}')
def mask(dataset_id:str,layer:str='svc',threshold:int=52,seed_x:int|None=None,seed_y:int|None=None):
    meta=read_dataset(dataset_id);d=dataset_dir(dataset_id)
    if not meta['flow_available'] or layer not in ('svc','dvc','cc'):raise HTTPException(422,'Unavailable vessel mask')
    img=np.array(Image.open(d/f'{layer}.png').convert('L'))
    default_seed=meta.get('fovea_seed',{})
    if seed_x is None: seed_x=default_seed.get('x')
    if seed_y is None: seed_y=default_seed.get('y')
    try:metrics,vessel,faz=vessel_and_faz(img,threshold,seed_x,seed_y,meta['calibration']['fov_x_mm'],meta['calibration']['fov_y_mm'])
    except ValueError as exc:raise HTTPException(422,str(exc)) from exc
    if layer!='svc':faz[:]=False
    return Response(content=metrics_overlay(img,vessel,faz,metrics['faz_seed']['x'],metrics['faz_seed']['y']),media_type='image/png')


@app.get('/api/datasets/{dataset_id}/surface')
def surface(dataset_id:str,step:int=5):
    d=dataset_dir(dataset_id)
    if not 1<=step<=24:raise HTTPException(422,'step must be in 1..24')
    with np.load(d/'boundaries.npz') as b:
        return {'step':step,'layers':{key:np.rint(b[key][::step,::step]).astype(int).tolist() for key in ('ilm','ipl_inl','opl_onl','rpe')},'dimensions':read_dataset(dataset_id)['shape']}


@app.get('/api/datasets/{dataset_id}/export')
def export(dataset_id:str,threshold:int=52,seed_x:int|None=None,seed_y:int|None=None):
    d=dataset_dir(dataset_id);meta=read_dataset(dataset_id)
    if not 1<=threshold<=99:raise HTTPException(422,'Invalid threshold percentile')
    defaults=meta.get('fovea_seed',{})
    if seed_x is None:seed_x=defaults.get('x')
    if seed_y is None:seed_y=defaults.get('y')
    output=io.BytesIO()
    with zipfile.ZipFile(output,mode='w',compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr('metadata.json',json.dumps(meta,indent=2))
        z.writestr('analysis_settings.json',json.dumps({'threshold_percentile':threshold,'faz_seed_x':seed_x,'faz_seed_y':seed_y},indent=2))
        rows=[]
        if meta['flow_available']:
            for layer in ('svc','dvc','cc'):
                img=np.array(Image.open(d/f'{layer}.png').convert('L'))
                metric,vessel,faz=vessel_and_faz(img,threshold,seed_x,seed_y,fov_x_mm=meta['calibration']['fov_x_mm'],fov_y_mm=meta['calibration']['fov_y_mm'])
                if layer=='svc':z.writestr('images/svc_vessel_faz_overlay.png',metrics_overlay(img,vessel,faz,metric['faz_seed']['x'],metric['faz_seed']['y']))
                rows.append({'layer':layer,'vessel_density_pct':metric['vessel_density_pct'],'faz_detected':metric['faz_detected'] if layer=='svc' else '', 'faz_area_px':metric['faz_area_px'] if layer=='svc' else '', 'faz_area_mm2':metric['faz_area_mm2'] if layer=='svc' else ''})
            csv_io=io.StringIO();writer=csv.DictWriter(csv_io,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            z.writestr('vascular_metrics.csv',csv_io.getvalue())
        for layer in meta['layers']:z.write(d/f'{layer}.png',f'images/{layer}.png')
        z.write(d/'boundaries.npz','boundaries.npz')
        z.writestr('README.txt','OCTAVE RESEARCH EXPORT. NOT FOR CLINICAL DIAGNOSIS.\nFAZ area requires supplied lateral calibration and successful detection.\nVessel metrics depend on enface normalization and threshold.\nRead metadata.json for QC flags and sampling limitations.\n')
    output.seek(0)
    return StreamingResponse(output,media_type='application/zip',headers={'Content-Disposition':f'attachment; filename="octave-{dataset_id}-export.zip"'})


frontend_dist=Path(__file__).resolve().parents[2]/'frontend'/'dist'
if frontend_dist.is_dir():
    app.mount('/',StaticFiles(directory=str(frontend_dist),html=True),name='frontend')
