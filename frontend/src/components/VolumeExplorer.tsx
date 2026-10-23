import { useEffect, useMemo, useState } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls, Grid } from '@react-three/drei'
import * as THREE from 'three'
import { Layers3, Move3d, RotateCw } from 'lucide-react'
import { api, type Surface } from '../lib/api'

const colors:Record<string,string> = {ilm:'#fb827e',ipl_inl:'#ffca80',opl_onl:'#37dcbf',rpe:'#8f8cf4'}
const labels:Record<string,string> = {ilm:'ILM',ipl_inl:'IPL / INL',opl_onl:'OPL / ONL',rpe:'RPE'}
function Sheet({points,color,level}:{points:number[][];color:string;level:number}){
  const geom=useMemo(()=>{
    const rows=points.length, cols=points[0]?.length||0
    const g=new THREE.BufferGeometry()
    const verts:number[]=[];const triangles:number[]=[]
    const stepX=9/(cols-1||1),stepZ=6/(rows-1||1)
    for(let y=0;y<rows;y++) for(let x=0;x<cols;x++) verts.push(x*stepX-4.5, 1.65-points[y][x]*.023+level*.0, y*stepZ-3)
    for(let y=0;y<rows-1;y++)for(let x=0;x<cols-1;x++){const a=y*cols+x,b=a+1,c=a+cols,d=c+1;triangles.push(a,c,b,b,c,d)}
    g.setAttribute('position',new THREE.Float32BufferAttribute(verts,3));g.setIndex(triangles);g.computeVertexNormals();return g
  },[points,level])
  useEffect(()=>()=>geom.dispose(),[geom])
  return <mesh geometry={geom}><meshStandardMaterial color={color} transparent opacity={.68} side={THREE.DoubleSide} metalness={.16} roughness={.35} depthWrite={false}/></mesh>
}
export default function VolumeExplorer({datasetId}:{datasetId:string}){
  const [data,setData]=useState<Surface|null>(null)
  const [visible,setVisible]=useState<Record<string,boolean>>({ilm:true,ipl_inl:true,opl_onl:true,rpe:true})
  const [error,setError]=useState('')
  useEffect(()=>{setData(null);setError('');api.surface(datasetId).then(setData).catch(e=>setError(String(e)))},[datasetId])
  return <div className="volume-layout">
    <div className="volume-stage">
      <div className="stage-label"><Layers3 size={15}/> RETINAL SURFACE EXPLORER <span>INTERACTIVE 3D</span></div>
      {data?<Canvas camera={{position:[8,5,10],fov:42}} dpr={[1,1.7]} gl={{antialias:true}}>
        <color attach="background" args={['#0a171b']}/>
        <fog attach="fog" args={['#0a171b',14,27]}/>
        <ambientLight intensity={1.35}/><pointLight position={[2,8,5]} intensity={65} color="#8cedd7"/>
        <group rotation={[0,-.28,0]}>{Object.entries(data.layers).filter(([key])=>visible[key]).map(([key,pts],i)=><Sheet key={key} points={pts} color={colors[key]} level={i}/>)}</group>
        <Grid position={[0,-4,0]} args={[16,16]} sectionSize={1} sectionColor="#2a5055" cellColor="#1b3438" fadeDistance={22} fadeStrength={2} infiniteGrid/>
        <OrbitControls makeDefault enablePan enableDamping minDistance={4} maxDistance={28}/>
      </Canvas>:<div className="loading-stage">{error||'Loading surface geometry…'}</div>}
      <div className="stage-caption"><Move3d size={14}/> Drag to rotate <span>·</span> Scroll to zoom <span>·</span> Right-drag to pan</div>
    </div>
    <aside className="volume-inspector"><div className="overline">DISPLAY SETTINGS</div><h3>Layer visibility</h3><p>Explore the geometric relationship between estimated retinal boundaries. Distances are scaled for visualization; the scan is not a volumetric vessel model.</p>
    <div className="surface-list">{Object.keys(colors).map(k=><label className="surface-toggle" key={k}><span className="surface-dot" style={{background:colors[k]}}/>{labels[k]}<input aria-label={`Show ${labels[k]}`} type="checkbox" checked={visible[k]} onChange={e=>setVisible(v=>({...v,[k]:e.target.checked}))}/><span className="toggle-track"/></label>)}</div>
    <div className="fact-block"><RotateCw size={17}/><div><b>Native depth coordinates</b><span>Surfaces derive from segmentation of structural OCT. Vertical exaggeration is for clarity.</span></div></div></aside>
  </div>
}
