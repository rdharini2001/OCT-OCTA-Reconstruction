import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { ReactNode, MouseEvent } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import {
  Activity, ArrowDownToLine, ArrowRight, ArrowUpRight, BookOpen, Boxes, Check, ChevronDown,
  ChevronLeft, ChevronRight, CircleAlert, ClipboardCheck, CloudUpload,
  Crosshair, Database, Eye, EyeOff, FileImage, Fingerprint, FlaskConical, Focus,
  Info, Layers, Maximize2, Microscope, MoreHorizontal,
  Plus, ScanEye, Settings2, ShieldCheck, SlidersHorizontal,
  Sparkles, UploadCloud, Waves, X,
} from 'lucide-react'
import { api, type Dataset, type Job, type Metrics } from './lib/api'
import VolumeExplorer from './components/VolumeExplorer'

const layerNames:Record<string,string> = {structural:'Structural',svc:'Superficial',dvc:'Deep vascular',avc:'Avascular',cc:'Choriocapillaris',choroid:'Choroid'}
const layerAbbrev:Record<string,string> = {structural:'OCT',svc:'SVC',dvc:'DVC',avc:'AVC',cc:'CC',choroid:'CH'}
const navigation = [
  {name:'Overview',key:'overview',icon:Boxes},
  {name:'Reconstruction',key:'reconstruction',icon:Layers},
  {name:'Vascular analysis',key:'vascular',icon:Activity},
  {name:'3D explorer',key:'volume',icon:ScanEye},
  {name:'Dataset library',key:'library',icon:Database},
  {name:'Methods & QC',key:'methods',icon:ClipboardCheck},
] as const

type Page = typeof navigation[number]['key']
const num=(x:number|null|undefined, digits=1)=>x===null||x===undefined?'—':x.toLocaleString('en-US',{maximumFractionDigits:digits})
function getError(e:unknown){return e instanceof Error?e.message:String(e)}

function SmallLabel({children}:{children:ReactNode}){return <span className="overline">{children}</span>}
function StatCard({icon:Icon,label,value,unit,caption,accent}:{icon:typeof Activity;label:string;value:string|number;unit?:string;caption:string;accent:string}){
  return <div className="stat-card"><div className="stat-card-top"><span className="stat-label">{label}</span><span className="stat-icon" style={{color:accent,background:`${accent}13`}}><Icon size={18}/></span></div><div className="stat-value">{value}<span>{unit}</span></div><div className="stat-caption"><span className="stat-dash" style={{background:accent}}/>{caption}</div></div>
}
function SideBar({active,setActive,onUpload,dataset}:{active:Page;setActive:(v:Page)=>void;onUpload:()=>void;dataset:Dataset|null}){
  return <aside className="sidebar"><div className="brand"><div className="brand-symbol"><div className="brand-pupil"/></div><div><div className="brand-name">OCTAVE<span className="brand-period">.</span></div><div className="brand-subline">RETINAL RESEARCH STUDIO</div></div></div>
    <div className="nav-block"><SmallLabel>WORKSPACE</SmallLabel><nav>{navigation.slice(0,4).map(({name,key,icon:Icon})=><button type="button" key={key} className={`nav-link ${active===key?'active':''}`} onClick={()=>setActive(key)}><Icon size={18} strokeWidth={1.85}/><span>{name}</span>{active===key&&<span className="nav-indicator"/>}</button>)}</nav></div>
    <div className="nav-block secondary"><SmallLabel>MANAGE</SmallLabel><nav>{navigation.slice(4).map(({name,key,icon:Icon})=><button type="button" key={key} className={`nav-link ${active===key?'active':''}`} onClick={()=>setActive(key)}><Icon size={18} strokeWidth={1.85}/><span>{name}</span>{active===key&&<span className="nav-indicator"/>}</button>)}</nav></div>
    <div className="sidebar-bottom"><div className="side-info"><div className="info-icon"><FlaskConical size={16}/></div><b>Built for discovery.</b><p>Open research tools. Transparent methods. Reproducible analysis.</p><button onClick={()=>setActive('methods')}>Read methodology <ArrowUpRight size={13}/></button></div><button className="sidebar-upload" onClick={onUpload}><Plus size={18}/> New analysis <ArrowRight size={16}/></button><div className="sidebar-footer"><div className="status-dot"/> SYSTEM ONLINE <span>·</span> v0.1.0</div></div>
  </aside>
}
function UploadModal({onClose,onSuccess}:{onClose:()=>void;onSuccess:(id:string)=>void}){
  const [structural,setStructural]=useState<File|null>(null)
  const [flow,setFlow]=useState<File|null>(null)
  const [name,setName]=useState('Untitled OCT study')
  const [x,setX]=useState('');const [y,setY]=useState('');const [z,setZ]=useState('')
  const [sampling,setSampling]=useState('128')
  const [job,setJob]=useState<Job|null>(null);const [error,setError]=useState('')
  const [busy,setBusy]=useState(false)
  const dropRef=useRef<HTMLInputElement>(null)
  useEffect(()=>{
    if(!job||job.status!=='processing')return
    const interval=window.setInterval(async()=>{
      try{const next=await api.job(job.id);setJob(next);if(next.status==='complete'){window.clearInterval(interval);onSuccess(next.dataset_id)}}catch(e){setError(getError(e));window.clearInterval(interval)}
    },1200)
    return ()=>window.clearInterval(interval)
  },[job?.id,job?.status,onSuccess])
  async function submit(){
    if(!structural)return
    if((x&&!y)||(!x&&y)){setError('Enter both scan width and height to calibrate area.');return}
    setBusy(true);setError('')
    try{
      const fd=new FormData();fd.append('structural',structural);if(flow)fd.append('flow',flow)
      fd.append('name',name);if(x)fd.append('fov_x_mm',x);if(y)fd.append('fov_y_mm',y);if(z)fd.append('axial_um_per_px',z)
      fd.append('max_slices',sampling)
      setJob(await api.upload(fd))
    }catch(e){setError(getError(e))}finally{setBusy(false)}
  }
  return <motion.div className="modal-backdrop" initial={{opacity:0}} animate={{opacity:1}} exit={{opacity:0}} onClick={onClose} role="presentation"><motion.div className="upload-modal" initial={{scale:.97,y:15,opacity:0}} animate={{scale:1,y:0,opacity:1}} exit={{scale:.97,opacity:0}} onClick={e=>e.stopPropagation()} role="dialog" aria-modal="true" aria-label="New OCT analysis">
    <div className="modal-heading"><div><SmallLabel>NEW WORKSPACE SESSION</SmallLabel><h2>Import your OCT volume</h2><p>Bring in your own data, reconstruct retinal slabs, and explore every layer.</p></div><button className="icon-button" aria-label="Close" onClick={onClose}><X size={20}/></button></div>
    {job?<div className="job-progress"><div className="job-spinner"><Waves size={34}/></div><h3>{job.status==='failed'?'Analysis did not complete':job.status==='complete'?'Analysis complete':'Processing your volume'}</h3><p>{job.error||job.stage}</p><div className="progress-track"><div style={{width:`${job.progress*100}%`}}/></div><span>{Math.round(job.progress*100)}% · {job.status}</span>{job.status==='failed'&&<button className="btn secondary" onClick={()=>{setJob(null);setError(job.error||'Processing failed')}}>Back to import</button>}</div>:<>
    <div className="modal-fields"><label className="field-group"><span>STUDY NAME</span><input value={name} onChange={e=>setName(e.target.value)} placeholder="e.g. Macular OCT study" maxLength={120}/></label>
    <div className="file-grid"><div className={`dropzone ${structural?'selected':''}`} onDragOver={e=>e.preventDefault()} onDrop={e=>{e.preventDefault();setStructural(e.dataTransfer.files[0]||null)}} onClick={()=>dropRef.current?.click()}><input ref={dropRef} type="file" accept=".zip,.npy,.npz,.tif,.tiff" hidden onChange={e=>setStructural(e.target.files?.[0]||null)}/><div className="drop-icon"><UploadCloud size={21}/></div><b>{structural?structural.name:'Structural OCT'}</b><span>{structural?`${(structural.size/1e6).toFixed(1)} MB · ready`:'Required · drop or browse'}</span><small>ZIP · NPY · NPZ · TIFF</small></div>
    <div className={`dropzone ${flow?'selected':''}`} onDragOver={e=>e.preventDefault()} onDrop={e=>{e.preventDefault();setFlow(e.dataTransfer.files[0]||null)}}><input id="flow-input" type="file" accept=".zip,.npy,.npz,.tif,.tiff" hidden onChange={e=>setFlow(e.target.files?.[0]||null)}/><label htmlFor="flow-input" className="drop-label"><div className="drop-icon"><Waves size={21}/></div><b>{flow?flow.name:'Paired flow / OCTA'}</b><span>{flow?`${(flow.size/1e6).toFixed(1)} MB · ready`:'Optional · enables vascular analysis'}</span><small>Must match structural dimensions</small></label></div></div>
    <div className="modal-section-head"><b>Spatial calibration</b><span>Recommended for research measurements</span></div><div className="calibration-grid"><label className="field-group"><span>SCAN WIDTH (mm)</span><input type="number" min="0" max="60" step="any" placeholder="e.g. 3" value={x} onChange={e=>setX(e.target.value)}/></label><label className="field-group"><span>SCAN HEIGHT (mm)</span><input type="number" min="0" max="60" step="any" placeholder="e.g. 3" value={y} onChange={e=>setY(e.target.value)}/></label><label className="field-group"><span>AXIAL (µm/px)</span><input type="number" min="0" step="any" placeholder="Optional" value={z} onChange={e=>setZ(e.target.value)}/></label></div>
    <label className="field-group"><span>PROCESSING RESOLUTION</span><select value={sampling} onChange={e=>setSampling(e.target.value)}><option value="128">Preview · up to 128 B-scans (faster, not for quantification)</option><option value="256">High detail · up to 256 B-scans (preview if sampled)</option><option value="0">Full resolution · process every B-scan</option></select></label></div>
    <div className="upload-warning"><Info size={17}/><span>Only upload de-identified research data. No data is sent to a cloud service by this application; files stay on your server.</span></div>
    {error&&<div className="form-error"><CircleAlert size={16}/>{error}</div>}
    <div className="modal-actions"><button className="btn ghost" onClick={onClose}>Cancel</button><button className="btn primary" disabled={!structural||busy} onClick={submit}><Sparkles size={17}/>{busy?'Uploading…':'Start reconstruction'}<ArrowRight size={16}/></button></div>
    </>}
  </motion.div></motion.div>
}

function LayerSelector({dataset,layer,onChange}:{dataset:Dataset;layer:string;onChange:(v:string)=>void}){
  return <div className="layer-selector" role="tablist" aria-label="OCT image layers">{dataset.layers.map(key=><button type="button" role="tab" aria-selected={layer===key} onClick={()=>onChange(key)} className={`layer-tab ${layer===key?'selected':''}`} key={key}><span>{layerAbbrev[key]}</span>{layerNames[key]}</button>)}</div>
}
function Viewer({dataset,layer,setLayer,scanIndex,setScanIndex,showLayers,setShowLayers,threshold,setThreshold,metrics,seed,setSeed}:{dataset:Dataset;layer:string;setLayer:(v:string)=>void;scanIndex:number;setScanIndex:(v:number)=>void;showLayers:boolean;setShowLayers:(v:boolean)=>void;threshold:number;setThreshold:(v:number)=>void;metrics:Metrics|null;seed:{x:number;y:number}|undefined;setSeed:(v:{x:number;y:number})=>void}){
  const [mode,setMode]=useState<'image'|'vessels'>('image');const [zoom,setZoom]=useState(1)
  const [brightness,setBrightness]=useState(100)
  const imgRef=useRef<HTMLImageElement>(null)
  const vascular=dataset.flow_available&&['svc','dvc','cc'].includes(layer)
  const masked=mode==='vessels'&&vascular
  useEffect(()=>{if(!vascular)setMode('image')},[vascular])
  function clickImage(e:MouseEvent<HTMLImageElement>){
    if(!vascular||layer!=='svc'||!imgRef.current)return
    const r=imgRef.current.getBoundingClientRect();const x=Math.round((e.clientX-r.left)/r.width*(dataset.shape[2]-1));const y=Math.round((e.clientY-r.top)/r.height*(dataset.shape[0]-1));setSeed({x:Math.max(0,Math.min(x,dataset.shape[2]-1)),y:Math.max(0,Math.min(y,dataset.shape[0]-1))});setMode('vessels')
  }
  return <div className="viewer-shell"><div className="viewer-heading"><div><div className="panel-eyebrow"><span className="live-dot"/> LIVE WORKSPACE <span className="eyebrow-divider">/</span> MULTIMODAL VIEWER</div><h3>Image reconstruction</h3></div><div className="viewer-heading-right"><span className="sample-label">{dataset.shape[2]} × {dataset.shape[0]} px</span><button className="icon-button" onClick={()=>setZoom(z=>z===1?1.45:1)} title="Toggle zoom"><Maximize2 size={17}/></button></div></div>
  <LayerSelector dataset={dataset} layer={layer} onChange={setLayer}/>
  <div className="viewer-panes"><div className="image-pane"><div className="pane-top"><div className="pane-name">EN FACE <span>·</span> {layerAbbrev[layer]}</div><div className="pane-status"><span className="dot-cyan"/> {dataset.flow_available&&layer!=='structural'?'FLOW PROJECTION':'REFLECTANCE'}</div></div>
      <div className="image-stage enface-stage"><div className="scan-corner top-left"/><div className="scan-corner top-right"/><div className="scan-corner bottom-left"/><div className="scan-corner bottom-right"/>
        <img ref={imgRef} onClick={clickImage} title={layer==='svc'?'Click to relocate FAZ seed':''} style={{filter:`brightness(${brightness}%)`,transform:`scale(${zoom})`,cursor:vascular&&layer==='svc'?'crosshair':'default'}} src={masked?api.mask(dataset.id,layer,threshold,seed):api.enface(dataset.id,layer)} alt={`${layerNames[layer]} en face retinal image`} draggable={false}/>
        <div className="enface-scanline" style={{top:`${(scanIndex+.5)/dataset.shape[0]*100}%`}}/><span className="image-axis axis-x">TEMPORAL <ArrowRight size={11}/> NASAL</span><span className="image-axis axis-y">SUPERIOR / INFERIOR</span>
      </div><div className="pane-bottom"><div className="viewer-toolset"><button className={mode==='image'?'tool-active':''} onClick={()=>setMode('image')}><Eye size={14}/> Original</button><button disabled={!vascular} className={mode==='vessels'?'tool-active':''} onClick={()=>setMode('vessels')}><Focus size={14}/> Vessel overlay</button></div><span className="pane-small">{layer==='svc'&&masked?'CLICK TO PLACE FAZ SEED':'EN FACE PROJECTION'}</span></div>
    </div><div className="image-pane"><div className="pane-top"><div className="pane-name">B-SCAN <span>·</span> CROSS-SECTION</div><div className="pane-status">{String(scanIndex+1).padStart(3,'0')} / {dataset.shape[0]}</div></div><div className="image-stage bscan-stage"><div className="scan-corner top-left"/><div className="scan-corner top-right"/><div className="scan-corner bottom-left"/><div className="scan-corner bottom-right"/><img src={api.bscan(dataset.id,scanIndex,showLayers)} alt={`Structural OCT B-scan ${scanIndex+1} ${showLayers?'with segmented retinal layer boundaries':''}`} style={{filter:`brightness(${brightness}%)`,transform:`scale(${zoom})`}} draggable={false}/><span className="bscan-tag">{showLayers?'BOUNDARIES ON':'RAW B-SCAN'}</span></div><div className="pane-bottom"><div className="viewer-toolset"><button className={showLayers?'tool-active':''} onClick={()=>setShowLayers(!showLayers)}>{showLayers?<Eye size={14}/>:<EyeOff size={14}/>} Segmentation</button><button onClick={()=>{setZoom(1);setBrightness(100)}}><Settings2 size={14}/> Reset view</button></div><span className="pane-small">STRUCTURAL OCT</span></div></div></div>
  <div className="slice-bar"><div className="slice-header"><span><SlidersHorizontal size={15}/> B-SCAN POSITION</span><b>Slice {scanIndex+1} <small>/ {dataset.shape[0]}</small></b></div><div className="slider-row"><button aria-label="Previous B-scan" onClick={()=>setScanIndex(Math.max(0,scanIndex-1))}><ChevronLeft size={18}/></button><input className="slice-slider" type="range" min={0} max={dataset.shape[0]-1} value={scanIndex} onChange={e=>setScanIndex(+e.target.value)} aria-label="B-scan position"/><button aria-label="Next B-scan" onClick={()=>setScanIndex(Math.min(dataset.shape[0]-1,scanIndex+1))}><ChevronRight size={18}/></button></div><div className="slice-markers"><span>SUPERIOR</span><span>FOVEAL CENTER</span><span>INFERIOR</span></div></div>
  <div className="viewer-footer"><div className="color-legend"><span><i style={{background:'#ff7d86'}}/>ILM</span><span><i style={{background:'#ffce78'}}/>IPL / INL</span><span><i style={{background:'#43e5c9'}}/>OPL / ONL</span><span><i style={{background:'#948fff'}}/>RPE</span></div><label className="brightness-control"><span><SlidersHorizontal size={14}/> Display</span><input type="range" min="65" max="165" value={brightness} onChange={e=>setBrightness(+e.target.value)}/></label></div>
  {masked&&<div className="threshold-control"><span><SlidersHorizontal size={15}/> Vessel detection percentile</span><input min="5" max="95" step="1" type="range" value={threshold} onChange={e=>setThreshold(+e.target.value)}/><strong>{threshold}th</strong><small>Image-derived threshold; not calibrated flow.</small></div>}
  </div>
}

function AnalysisInspector({dataset,metrics,layer}:{dataset:Dataset;metrics:Metrics|null;layer:string}){
  const svc=layer==='svc';const flow=dataset.flow_available;const qcOk=!dataset.qc.any_flagged
  return <div className="analysis-inspector"><div className="inspector-title"><SmallLabel>ANALYSIS INSIGHTS</SmallLabel><MoreHorizontal size={20}/></div><div className="inspector-intro"><div className="inspector-orb"><Fingerprint size={25}/></div><h3>Vascular metrics</h3><p>Image-derived quantitative markers from the current segmentation.</p></div>
  <div className="inspector-metric hero-metric"><span>VESSEL AREA DENSITY</span><div>{flow&&metrics?num(metrics.vessel_density_pct,2):'—'} <small>{flow?'%':''}</small></div><p>{flow?'Threshold-derived area fraction':'Requires paired OCTA flow data'}</p></div>
  <div className="inspector-split"><div><span>SKELETON LENGTH</span><b>{metrics?num(metrics.vessel_length_px_proxy,0):'—'}</b><small>pixel-path proxy</small></div><div><span>FAZ CIRCULARITY</span><b>{svc&&metrics?.faz_detected?num(metrics.faz_circularity,3):'—'}</b><small>{svc&&metrics?!metrics.faz_detected?'Not detected':'0–1 range':'SVC only'}</small></div></div>
  <div className="divider-line"/><div className="inspector-faz"><div className="section-title-sm"><Crosshair size={16}/> FOVEAL AVASCULAR ZONE</div><div className="faz-display"><div><span>FAZ AREA</span><strong>{svc&&metrics?.faz_detected?num(metrics.faz_area_mm2??metrics.faz_area_px,metrics.faz_area_mm2!=null?3:0):'—'}</strong><small>{svc&&metrics?.faz_detected?metrics.faz_area_mm2!=null?'mm²':'px² (uncalibrated)':'Not detected / unavailable'}</small></div><div className={`faz-chip ${svc&&metrics?.faz_detected?'detected':''}`}>{svc&&metrics?.faz_detected?<Check size={12}/>:<Info size={12}/>} {svc&&metrics?.faz_detected?'ENCLOSED':'UNAVAILABLE'}</div></div><p>{svc&&metrics?.faz_detected?`Morphological closing: ${metrics.closing_radius_px}px. Inspect contour before using in research.`:'FAZ is reported only when a plausible, enclosed seed-region is detected in the SVC mask.'}</p></div>
  <div className="divider-line"/><div className="quality-title"><span><ShieldCheck size={16}/> QUALITY CONTROL</span><span className={`qc-pill ${qcOk?'good':'warning'}`}>{qcOk?'CHECKS PASSED':'REVIEW'}</span></div><div className="quality-grid"><div><span>Scan mode</span><b>{dataset.preview_sampling?'Sampled preview':'Full data'}</b></div><div><span>Spatial scales</span><b>{dataset.calibration.fov_x_mm?'Calibrated':'Not provided'}</b></div><div><span>Source</span><b>{dataset.synthetic?'Synthetic phantom':'Imported volume'}</b></div></div>
  <div className="inspector-note"><Info size={15}/><p>Research-only measurements. Anatomical boundaries and thresholds require expert review; no diagnosis or clinical decision support.</p></div></div>
}
function DensityChart({metrics,dataset,scanIndex,setScanIndex}:{metrics:Metrics|null;dataset:Dataset;scanIndex:number;setScanIndex:(x:number)=>void}){
  const points=useMemo(()=>metrics?.row_density_pct.map((v,i)=>({slice:i+1,density:v}))||[],[metrics])
  return <div className="chart-card"><div className="chart-heading"><div><SmallLabel>SPATIAL PROFILE</SmallLabel><h3>Vessel density across B-scans</h3><p>Area fraction by en face row · exploratory view</p></div><div className="chart-chip"><span className="chart-chip-dot"/> SVC / SELECTED PLEXUS</div></div><div className="chart-body">{metrics?<ResponsiveContainer width="100%" height="100%"><AreaChart data={points} margin={{top:14,right:5,bottom:0,left:-23}} onClick={state=>{const p=state?.activeLabel;if(p!==undefined)setScanIndex(Math.max(0,Math.min(dataset.shape[0]-1,Number(p)-1)))}}><defs><linearGradient id="area-gradient" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#40d8c0" stopOpacity={.35}/><stop offset="95%" stopColor="#40d8c0" stopOpacity={.01}/></linearGradient></defs><CartesianGrid vertical={false} stroke="#dfece9" strokeDasharray="3 6"/><XAxis dataKey="slice" tickLine={false} axisLine={false} fontSize={10} tick={{fill:'#7c9692'}} interval="preserveStartEnd"/><YAxis tickLine={false} axisLine={false} fontSize={10} tick={{fill:'#7c9692'}} unit="%"/><Tooltip contentStyle={{border:'1px solid #c9e3dc',borderRadius:12,background:'#fff',fontSize:12}} formatter={v=>[`${v}%`,'Vessel density']} labelFormatter={v=>`B-scan ${v}`}/><Area type="monotone" dataKey="density" stroke="#189c89" strokeWidth={2.4} fill="url(#area-gradient)" activeDot={{r:5,fill:'#189c89'}}/></AreaChart></ResponsiveContainer>:<div className="chart-empty">{dataset.flow_available?'Computing density profile…':'Pair a flow/OCTA volume to see the vascular density profile.'}</div>}</div><div className="chart-footer"><span>0 <em>SUPERIOR</em></span><span>SELECTED SLICE {scanIndex+1}</span><span><em>INFERIOR</em> {dataset.shape[0]}</span></div></div>
}
function Warnings({dataset}:{dataset:Dataset}){
 const [expanded,setExpanded]=useState(false)
 return <div className={`warnings-strip ${dataset.qc.any_flagged?'alert':''}`}><div className="warnings-icon"><CircleAlert size={19}/></div><div><b>{dataset.preview_sampling?'Preview sampling active':dataset.synthetic?'Synthetic demonstration active':dataset.qc.any_flagged?'Segmentation requires review':'Research-grade quality guidance'}</b><p>{dataset.warnings[0]||'Review boundary segmentation and acquisition calibration before interpreting results.'}</p>{expanded&&<ul>{dataset.warnings.slice(1).map((w,i)=><li key={i}>{w}</li>)}</ul>}</div><button onClick={()=>setExpanded(!expanded)}>{expanded?'Hide details':'View all warnings'} <ChevronDown size={15} style={{transform:expanded?'rotate(180deg)':'none'}}/></button></div>
}
function DatasetLibrary({datasets,selected,onSelect,onUpload}:{datasets:Dataset[];selected:string;onSelect:(id:string)=>void;onUpload:()=>void}){
  return <div className="library-page"><div className="section-page-heading"><SmallLabel>YOUR RESEARCH LIBRARY</SmallLabel><h2>Every dataset. One workspace.</h2><p>Analyze imported scans, revisit reconstructions, and move between studies without changing tools.</p></div><div className="library-toolbar"><div><Database size={18}/> {datasets.length} STUDIES AVAILABLE</div><button className="btn primary" onClick={onUpload}><Plus size={17}/> Import new study</button></div><div className="dataset-grid">{datasets.map(d=><button className={`dataset-card ${selected===d.id?'active':''}`} key={d.id} onClick={()=>onSelect(d.id)}><div className="dataset-thumbnail"><img src={api.enface(d.id,d.layers.includes('svc')?'svc':'structural')} alt="Study preview"/><span>{d.synthetic?'DEMO':'IMPORTED'}</span></div><div className="dataset-content"><div className="dataset-title"><h3>{d.name}</h3><ArrowUpRight size={18}/></div><span className="dataset-meta">{d.shape[0]} B-scans <i/> {d.flow_available?'OCT + OCTA':'Structural only'}</span><div className="dataset-bottom"><span>{d.preview_sampling?'SAMPLED PREVIEW':'FULL STACK'}</span><span>{selected===d.id?<><Check size={13}/> ACTIVE</>:'OPEN STUDY'}</span></div></div></button>)}<button className="dataset-add" onClick={onUpload}><div><Plus size={25}/></div><b>Import another dataset</b><p>Bring your own B-scan stack</p></button></div></div>
}
function Methods(){return <div className="methods-page"><div className="section-page-heading"><SmallLabel>REPRODUCIBLE SCIENCE</SmallLabel><h2>The method behind the image.</h2><p>Explicit assumptions, independently inspectable intermediate results and transparent quality controls.</p></div><div className="method-banner"><div><span><FlaskConical size={18}/> OPEN RESEARCH ENGINE</span><h3>Every segmentation is a hypothesis to verify.</h3><p>Inspect the original B-scan next to the segmented layers and follow the derivation of each metric. The application prioritizes traceability over invented precision.</p></div><div className="method-banner-ring"><Layers size={56}/></div></div><div className="methods-grid">
  {[
    {step:'01',title:'Volume ingestion',detail:'Numbered PNG/TIFF ZIP stacks, 3D NPY/NPZ arrays and multi-page TIFF. Requires explicit (B-scan, depth, A-scan) ordering; paired structural and flow volumes must match shape.',icon:Database},
    {step:'02',title:'Retinal segmentation',detail:'The supplied heuristic OCTA pipeline traces ILM, IPL/INL, OPL/ONL and RPE with smoothness-constrained paths and reports QC flags. Boundaries may fail on unseen pathology or device protocols.',icon:Layers},
    {step:'03',title:'Slab reconstruction',detail:'En face reflectance from structural OCT; OCTA plexus projections ONLY from registered flow/decorrelation volumes. Deep projection-artifact attenuation and bulk-motion correction use the provided experimental pipeline.',icon:Waves},
    {step:'04',title:'Vessels & FAZ',detail:'Percentile-based binary vessel mask with morphological cleanup; FAZ from enclosed non-vessel component seeded near center or manually placed. Return not-detected when a plausible region is absent.',icon:Crosshair},
    {step:'05',title:'Physical calibration',detail:'Lateral field of view is required for mm² FAZ area. Axial µm/pixel is required for thickness in micrometers. Missing metadata never triggers an assumed 3×3 mm scan.',icon:SlidersHorizontal},
    {step:'06',title:'Scientific validation',detail:'The software is a research prototype, not a medically validated OCTA generator. Assess accuracy by device, pathology, and scan protocol using expert-annotated reference data before quantitative studies.',icon:ShieldCheck},
  ].map(({step,title,detail,icon:Icon})=><div className="method-card" key={step}><div className="method-num">{step}<Icon size={21}/></div><h3>{title}</h3><p>{detail}</p></div>)}
  </div><div className="references"><h3>Methods that informed this workspace</h3><p><a href="https://github.com/jaburke166/OCTolyzer" target="_blank" rel="noreferrer">OCTolyzer <ArrowUpRight size={13}/></a> for transparent retinal/choroidal research tooling; <a href="https://github.com/justinengelmann/Choroidalyzer" target="_blank" rel="noreferrer">Choroidalyzer <ArrowUpRight size={13}/></a> for optional trained choroid segmentation; and the user-provided OCTA pipeline / FAZ extraction experiments. Model weights are not bundled. Code reuse is documented in this repository.</p></div></div>}

function App(){
  const [datasets,setDatasets]=useState<Dataset[]>([]);const [datasetId,setDatasetId]=useState('synthetic-demo');const [page,setPage]=useState<Page>('overview');const [showUpload,setShowUpload]=useState(false)
  const [initialLoading,setInitialLoading]=useState(true);const [loadError,setLoadError]=useState('')
  const [layer,setLayer]=useState('svc');const [scanIndex,setScanIndex]=useState(50)
  const [showLayers,setShowLayers]=useState(true);const [threshold,setThreshold]=useState(52)
  const [metrics,setMetrics]=useState<Metrics|null>(null);const [seed,setSeed]=useState<{x:number;y:number}|undefined>()
  const dataset=datasets.find(d=>d.id===datasetId)||datasets[0]||null
  const activeDatasetId=dataset?.id
  const refresh=useCallback(async()=>{const list=await api.list();setDatasets(list);return list},[])
  useEffect(()=>{refresh().then(()=>setInitialLoading(false)).catch(e=>{setInitialLoading(false);setLoadError(getError(e))})},[refresh])
  useEffect(()=>{if(!dataset)return;setScanIndex(Math.floor(dataset.shape[0]/2));setSeed(undefined);setLayer(dataset.flow_available?'svc':'structural');setThreshold(52)},[activeDatasetId])
  useEffect(()=>{
    if(!dataset?.flow_available||!['svc','dvc','cc'].includes(layer)){setMetrics(null);return}
    let alive=true;const t=window.setTimeout(()=>api.metrics(dataset.id,layer,threshold,seed).then(m=>{if(alive)setMetrics(m)}).catch(()=>{if(alive)setMetrics(null)}),90)
    return()=>{alive=false;window.clearTimeout(t)}
  },[activeDatasetId,layer,threshold,seed?.x,seed?.y,dataset?.flow_available])
  const handleImported=useCallback(async(id:string)=>{try{await refresh();setDatasetId(id);setPage('overview');setShowUpload(false)}catch(e){setLoadError(getError(e))}},[refresh])
  const selectDataset=(id:string)=>{setDatasetId(id);setPage('overview')}
  if(initialLoading)return <div className="boot-screen"><div className="brand-symbol big"><div className="brand-pupil"/></div><strong>OCTAVE</strong><p>Opening retinal research workspace…</p></div>
  if(loadError&&!dataset)return <div className="boot-screen"><CircleAlert size={28}/><h2>Analysis server not available</h2><p>{loadError}</p><code>cd backend && uvicorn app.main:app --reload</code><button className="btn primary" onClick={()=>window.location.reload()}>Try again</button></div>
  return <div className="app-shell"><SideBar active={page} setActive={setPage} onUpload={()=>setShowUpload(true)} dataset={dataset}/><div className="main-layout"><header className="topbar"><div className="breadcrumb"><span>Research workspace</span><ChevronRight size={14}/><strong>{navigation.find(n=>n.key===page)?.name}</strong></div><div className="topbar-actions"><span className="research-pill"><span/> RESEARCH MODE</span><div className="header-sep"/><button className="topbar-docs" onClick={()=>setPage('methods')}><BookOpen size={17}/> Documentation</button><div className="avatar">O</div></div></header>
    <main className="main-content">{dataset&&<>
      {page==='library'?<DatasetLibrary datasets={datasets} selected={dataset.id} onSelect={selectDataset} onUpload={()=>setShowUpload(true)}/>:page==='methods'?<Methods/>:page==='volume'?<><div className="page-header"><div><SmallLabel>3D RETINAL ANATOMY</SmallLabel><h1>Explore every contour<span className="highlight-dot">.</span></h1><p>Rotate, isolate and inspect retinal surfaces derived from your OCT volume.</p></div><div className="page-header-actions"><button className="btn secondary" onClick={()=>setPage('reconstruction')}><Layers size={16}/> 2D workspace</button></div></div><VolumeExplorer datasetId={dataset.id}/><Warnings dataset={dataset}/></>:<>
      <div className="page-header"><div><div className="page-header-kicker"><span className="kicker-line"/> ADVANCED OCT & OCTA ANALYSIS <span className="kicker-version">/ RESEARCH STUDIO</span></div><h1>{page==='vascular'?'The vascular landscape':page==='reconstruction'?'Reconstruct what matters':'See beneath the surface'}<span className="highlight-dot">.</span></h1><p>{page==='vascular'?'Anatomy-guided vessel mapping, quantitative density and transparent FAZ evaluation.':page==='reconstruction'?'Transform volumetric B-scans into inspectable, depth-resolved en face views.':'A deeper perspective on retinal structure, microvasculature and the foveal zone.'}</p></div><div className="page-header-actions"><button className="btn secondary" onClick={()=>window.open(api.export(dataset.id,threshold,seed),'_blank')}><ArrowDownToLine size={16}/> Export results</button><button className="btn primary" onClick={()=>setShowUpload(true)}><Plus size={17}/> New analysis</button></div></div>
      <div className="study-bar"><div className="study-bar-left"><div className="study-icon"><ScanEye size={19}/></div><div><span>ACTIVE STUDY</span><div className="select-wrapper"><select value={dataset.id} onChange={e=>setDatasetId(e.target.value)} aria-label="Active study">{datasets.map(d=><option key={d.id} value={d.id}>{d.name}</option>)}</select><ChevronDown size={15}/></div></div></div><div className="study-details"><span><Database size={15}/>{dataset.shape[0]} B-scans</span><span><FileImage size={15}/>{dataset.shape[2]} × {dataset.shape[1]} px</span><span><Waves size={15}/>{dataset.flow_available?'Paired flow input':'Structural only'}</span></div><div className={`study-validity ${dataset.synthetic?'demo':dataset.preview_sampling?'preview':''}`}><span/>{dataset.synthetic?'SYNTHETIC DATA':dataset.preview_sampling?'PREVIEW / SAMPLED':'FULL RESOLUTION'}</div></div>
      <div className="stats-grid"><StatCard icon={Activity} label="VESSEL DENSITY" value={metrics?num(metrics.vessel_density_pct,2):'—'} unit={metrics?'%':''} caption={metrics?'Selected plexus · area fraction':'Flow data required'} accent="#11b89e"/><StatCard icon={Crosshair} label="FAZ AREA" value={layer==='svc'&&metrics?.faz_detected?num(metrics.faz_area_mm2??metrics.faz_area_px,metrics.faz_area_mm2!=null?3:0):'—'} unit={layer==='svc'&&metrics?.faz_detected?(metrics.faz_area_mm2!=null?'mm²':'px²'):''} caption={metrics?.faz_detected&&layer==='svc'?'Morphological segmentation':'Not detected / SVC only'} accent="#eeae68"/><StatCard icon={Layers} label="RETINAL THICKNESS" value={num(dataset.retinal_thickness_um??dataset.retinal_thickness_px,1)} unit={dataset.retinal_thickness_um!=null?'µm':'px'} caption="Median ILM to RPE · estimated" accent="#8189f3"/><StatCard icon={ShieldCheck} label="SEGMENTATION QC" value={dataset.qc.any_flagged?'Review':'Passed'} caption={dataset.preview_sampling?'Preview acquisition':'Automated QC · inspect overlays'} accent={dataset.qc.any_flagged?'#d99a4d':'#16bda2'}/></div>
      <div className={`workspace-grid ${page==='reconstruction'?'expanded-workspace':''}`}><Viewer dataset={dataset} layer={layer} setLayer={setLayer} scanIndex={scanIndex} setScanIndex={setScanIndex} showLayers={showLayers} setShowLayers={setShowLayers} threshold={threshold} setThreshold={setThreshold} metrics={metrics} seed={seed} setSeed={setSeed}/><AnalysisInspector dataset={dataset} metrics={metrics} layer={layer}/></div>
      <div className="below-grid"><DensityChart metrics={metrics} dataset={dataset} scanIndex={scanIndex} setScanIndex={setScanIndex}/><div className="quick-actions"><SmallLabel>RESEARCH TOOLS</SmallLabel><h3>Take the analysis further</h3><p>Inspect volumetric anatomy or share reproducible results with your team.</p><button onClick={()=>setPage('volume')}><div className="quick-action-icon"><ScanEye size={18}/></div><span><b>3D surface explorer</b><small>Explore retinal geometry</small></span><ArrowUpRight size={17}/></button><button onClick={()=>window.open(api.export(dataset.id,threshold,seed),'_blank')}><div className="quick-action-icon amber"><ArrowDownToLine size={18}/></div><span><b>Export research package</b><small>Metadata, metrics, masks & images</small></span><ArrowUpRight size={17}/></button></div></div>
      <Warnings dataset={dataset}/>
    </>}</>}
    <footer className="page-footer"><div className="footer-lock"><Microscope size={15}/> OCTAVE RESEARCH STUDIO <span>·</span> NOT FOR CLINICAL DIAGNOSIS</div><div>OCT / OCTA <span>·</span> v0.1.0 <span>·</span> Open research prototype</div></footer></main></div>
    <AnimatePresence>{showUpload&&<UploadModal onClose={()=>setShowUpload(false)} onSuccess={handleImported}/>}</AnimatePresence></div>
}
export default App
