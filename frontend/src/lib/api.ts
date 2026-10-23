export interface Calibration { fov_x_mm: number | null; fov_y_mm: number | null; axial_um_per_px: number | null }
export interface QC { any_flagged: boolean; [key: string]: unknown }
export interface Dataset {
  id: string; name: string; created_at: string; source: string; synthetic: boolean; status: string;
  shape: [number, number, number]; original_slices: number; sampled_slices: number; preview_sampling: boolean;
  flow_available: boolean; layers: string[]; calibration: Calibration;
  retinal_thickness_px: number; retinal_thickness_um: number | null;
  qc: QC; warnings: string[]; pipeline: string
}
export interface Metrics {
  layer: string; threshold_percentile: number; threshold_intensity: number; vessel_density_pct: number;
  vessel_length_px_proxy: number; vessel_length_mm_proxy: number | null;
  faz_detected?: boolean; faz_area_px?: number | null; faz_area_mm2?: number | null;
  faz_perimeter_px?: number | null; faz_circularity?: number | null;
  faz_seed?: {x:number;y:number}; closing_radius_px?: number|null;
  row_density_pct: number[]; calibrated: boolean; method_note: string
}
export interface Job { id:string; dataset_id:string; status:'processing'|'failed'|'complete'; stage:string; progress:number; error?:string }
export interface Surface { dimensions:[number,number,number]; step:number; layers:Record<string,number[][]> }
async function read<T>(path:string, init?:RequestInit):Promise<T> {
  const r=await fetch(path,init)
  if(!r.ok){const t=await r.text();try{throw new Error(JSON.parse(t).detail || t)}catch(e){if(e instanceof Error && !e.message.startsWith('{'))throw e;throw new Error(t)}}
  return r.json() as Promise<T>
}
export const api = {
  list:()=>read<Dataset[]>('/api/datasets'),
  metrics:(id:string,layer:string,threshold:number,seed?:{x:number;y:number})=>read<Metrics>(`/api/datasets/${id}/analyze/${layer}?threshold=${threshold}${seed?`&seed_x=${seed.x}&seed_y=${seed.y}`:''}`),
  surface:(id:string)=>read<Surface>(`/api/datasets/${id}/surface?step=6`),
  upload:(fd:FormData)=>read<Job>('/api/datasets',{method:'POST',body:fd}),
  job:(id:string)=>read<Job>(`/api/jobs/${id}`),
  bscan:(id:string,i:number,overlay:boolean)=>`/api/datasets/${id}/bscan/${i}?overlay=${overlay}`,
  enface:(id:string,layer:string)=>`/api/datasets/${id}/enface/${layer}`,
  mask:(id:string,layer:string,threshold:number,seed?:{x:number;y:number})=>`/api/datasets/${id}/mask/${layer}?threshold=${threshold}${seed?`&seed_x=${seed.x}&seed_y=${seed.y}`:''}`,
  export:(id:string,threshold=52,seed?:{x:number;y:number})=>`/api/datasets/${id}/export?threshold=${threshold}${seed?`&seed_x=${seed.x}&seed_y=${seed.y}`:''}`,
}
