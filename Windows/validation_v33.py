"""End-to-end validation of the 3.3 Windows pipeline on the local RTX.

Run from source:   python validation_v33.py <output-folder> [resolution_mm]
Run packaged:      Bichito3D-Unificado-v3.3.exe --validate <output-folder>

A synthetic *general object* (box + spheres, no face assumptions) is ray-cast
from an orbiting TrueDepth-like camera with 0.5 mm depth noise and no IMU help.
Blocks go through the real v3 BlockStore (lossless codec, SHA-256, fsync),
then both reconstruction modes run on the same session. Results are measured
against the analytic surface, rendered with the solid viewer and exported.

This proves the software path (GPU stages, LOD viewer, export selection). It
does not measure TrueDepth's physical accuracy.
"""
import hashlib
import io
import json
import math
import struct
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
import numpy as np

W,H,FX,CX,CY=320,240,280.,160.,120.
SPHERE=(np.array([0.,-.022,0.]),.020)
BUMP=(np.array([.030,-.004,-.020]),.008)
BOX_CENTER=np.array([0.,.012,0.]); BOX_HALF=np.array([.035,.014,.024]); BOX_ANGLE=math.radians(20)


def ry(angle):
    c,s=math.cos(angle),math.sin(angle); return np.array([[c,0,s],[0,1,0],[-s,0,c]])


BOX_R=ry(BOX_ANGLE)


def sdf(points):
    """Signed distance to the union (metres)."""
    out=[]
    for center,radius in (SPHERE,BUMP): out.append(np.linalg.norm(points-center,axis=1)-radius)
    local=(points-BOX_CENTER)@BOX_R; q=np.abs(local)-BOX_HALF
    out.append(np.linalg.norm(np.maximum(q,0),axis=1)+np.minimum(q.max(1),0))
    return np.min(out,axis=0)


def raycast(origin,directions):
    t=np.full(len(directions),np.inf); obj=np.full(len(directions),-1)
    for index,(center,radius) in enumerate((SPHERE,BUMP)):
        oc=origin-center; b=directions@oc; c=oc@oc-radius**2; disc=b*b-c
        hit=disc>0; root=-b-np.sqrt(np.where(hit,disc,0))
        better=hit&(root>0)&(root<t); t[better]=root[better]; obj[better]=index
    o=(origin-BOX_CENTER)@BOX_R; d=directions@BOX_R
    with np.errstate(divide='ignore',invalid='ignore'):
        t1=(-BOX_HALF-o)/d; t2=(BOX_HALF-o)/d
    near=np.nanmax(np.minimum(t1,t2),axis=1); far=np.nanmin(np.maximum(t1,t2),axis=1)
    hit=(near<=far)&(near>0)&(near<t); t[hit]=near[hit]; obj[hit]=2
    return t,obj


def frames(count=30,arc=70.,noise=.0005,seed=1):
    rng=np.random.default_rng(seed)
    v,u=np.mgrid[:H,:W]
    rays=np.stack(((u-CX)/FX,(v-CY)/FX,np.ones_like(u,dtype=float)),-1).reshape(-1,3)
    for index in range(count):
        angle=math.radians(-arc/2+arc*index/(count-1)); tilt=math.radians(12*math.sin(index/4))
        R=ry(angle)@np.array([[1,0,0],[0,math.cos(tilt),-math.sin(tilt)],[0,math.sin(tilt),math.cos(tilt)]])
        position=R@np.array([0,0,-.30])
        world=rays@R.T; length=np.linalg.norm(world,axis=1)
        t,obj=raycast(position,world/length[:,None])
        depth=(t/length).reshape(H,W)   # z in camera frame
        depth=np.where(np.isfinite(depth),depth+rng.normal(0,noise,depth.shape),np.nan).astype(np.float32)
        palette=np.array([[200,90,60],[60,160,90],[70,110,210],[0,0,0]],np.uint8)
        rgb=palette[obj.reshape(H,W)]
        stripes=((u//6+v//6)%2==0)[...,None]
        rgb=np.where(stripes,rgb,(rgb*.75).astype(np.uint8))
        bgra=np.concatenate([rgb[...,::-1],np.full((H,W,1),255,np.uint8)],2)
        T=np.eye(4); T[:3,:3]=R; T[:3,3]=position
        yield index,depth,bgra,T


def raw_packet(index,depth,bgra):
    meta=json.dumps({'source':'validation_v33 synthetic ray-cast','depthOriginalFormat':'fdep','depthConverted':False}).encode()
    header=struct.pack('<IId9f3f4fHHI',0xB1C01702,index,1+index/15,1,0,0,0,1,0,0,0,1,0,0,0,FX,FX,CX,CY,W,H,depth.nbytes)
    return header+depth.astype('<f4').tobytes()+struct.pack('<HHI',W,H,bgra.nbytes)+bgra.tobytes()+struct.pack('<I',len(meta))+meta


def encode(packet):
    """Same lossless block format as the iPhone (byte-shuffled Zstd depth + PNG)."""
    import zstandard
    from PIL import Image
    size=struct.unpack_from('<I',packet,84)[0]; offset=88+size
    w,h,rgb_size=struct.unpack_from('<HHI',packet,offset)
    shuffled=np.frombuffer(packet[88:offset],np.uint8).reshape(-1,4).T.copy().tobytes()
    depth=zstandard.ZstdCompressor(level=1).compress(shuffled)
    bgra=np.frombuffer(packet,np.uint8,count=rgb_size,offset=offset+8).reshape(h,w,4)
    output=io.BytesIO(); Image.fromarray(np.ascontiguousarray(bgra[:,:,[2,1,0,3]])).save(output,format='PNG')
    png=output.getvalue()
    header=bytearray(packet[:88]); struct.pack_into('<I',header,0,0xB1C01703); struct.pack_into('<I',header,84,len(depth))
    return bytes(header)+depth+struct.pack('<HHI',w,h,len(png))+png+packet[offset+8+rgb_size:]


def gpu_sampler(stop,samples):
    while not stop.is_set():
        try:
            out=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader,nounits'],
                               capture_output=True,text=True,timeout=5,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)).stdout
            util,mem=[float(x) for x in out.strip().splitlines()[0].split(',')]
            samples.append((time.time(),util,mem))
        except Exception: pass
        stop.wait(.5)


def hardware():
    info={}
    try:
        out=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total,temperature.gpu','--format=csv,noheader'],
                           capture_output=True,text=True,timeout=10,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)).stdout.strip()
        info['nvidia_smi']=out
    except Exception as error: info['nvidia_smi']=f'no disponible: {error}'
    from gpu_backend import cuda_module
    cp=cuda_module(); props=cp.cuda.runtime.getDeviceProperties(0)
    info.update(cuda_device=props['name'].decode(),compute_capability=f"{props['major']}.{props['minor']}",
                vram_total_mb=round(props['totalGlobalMem']/1024**2),cuda_driver=cp.cuda.runtime.driverGetVersion(),
                cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),cupy=cp.__version__)
    import torch
    info.update(torch=torch.__version__,torch_cuda=torch.version.cuda,torch_sees_cuda=torch.cuda.is_available())
    return info


def run(destination,resolution_mm=.2):
    out=Path(destination); out.mkdir(parents=True,exist_ok=True)
    report={'started':time.strftime('%Y-%m-%d %H:%M:%S'),'frozen':bool(getattr(sys,'frozen',False)),'resolution_mm':resolution_mm}
    try:
        report['hardware']=hardware()
        from chunk_protocol import BlockStore
        from offline_pipeline import Settings,reconstruct,export_selected
        from mesh_view import Camera,GpuRenderer
        from PIL import Image
        store=BlockStore(out/'sessions'); session=str(uuid.uuid4())
        poses=[]; raw_bytes=0; block_bytes=0
        for index,depth,bgra,T in frames():
            packet=raw_packet(index,depth,bgra); block=encode(packet); raw_bytes+=len(packet); block_bytes+=len(block)
            message=struct.pack('<I',index)+hashlib.sha256(block).digest()+block
            ack=store.accept(session,message); assert ack['seq']==index
            ack_again=store.accept(session,message); assert ack_again==ack   # resend after lost ACK: no duplicate
            poses.append(T)
        path,receipt=store.finish(session,len(poses),0)
        report['session']={'frames':receipt['frames'],'raw_mb':round(raw_bytes/1e6,2),'lossless_blocks_mb':round(block_bytes/1e6,2)}
        settings=Settings(resolution_mm=resolution_mm,registration_mm=1.,memory_mb=2048,device='CUDA',min_observations=2.)
        from dataclasses import replace
        results={}
        to_scene=poses[0]   # reconstruction frame = first camera
        for name,config in (('RTX sola',replace(settings,neural_denoising=False,neural_alignment=False)),
                            ('RTX + IA',replace(settings,neural_denoising=True,neural_alignment=False))):
            stop=threading.Event(); samples=[]; sampler=threading.Thread(target=gpu_sampler,args=(stop,samples),daemon=True); sampler.start()
            start=time.time()
            model,model_report=reconstruct(path,config,lambda *args:None,None)
            wall=time.time()-start
            stop.set(); sampler.join(2)
            # Geometric error against the analytic object (algorithm, not sensor).
            errors=[]
            for tile in sorted(Path(model_report['folder']).glob('tile-*.npz')):
                with np.load(tile) as data: v=data['vertices'].reshape(-1,3)[::7]
                world=v@to_scene[:3,:3].T+to_scene[:3,3]; errors.append(np.abs(sdf(world)))
            errors=np.concatenate(errors)*1000
            view=model.view_mesh(); renderer=GpuRenderer(); renderer.upload(view)
            camera=Camera(); camera.frame(view); camera.orbit(80,-60)
            if results: camera.reference=results['RTX sola']['camera_reference']
            image=renderer.render(camera,800,600,use_color=False)
            Image.fromarray(image).save(out/f"viewer-{'rtx-ia' if 'IA' in name else 'rtx'}.png")
            covered=float((np.abs(image.astype(int)-image[0,0].astype(int)).sum(2)>30).mean())
            stl=out/f"{'rtx-ia' if 'IA' in name else 'rtx'}.stl"
            count=export_selected(model,stl)
            busy=[u for _,u,_ in samples]
            results[name]={'triangles':model_report['triangles'],'accepted':model_report['accepted'],'rejected':model_report['rejected'],
                'keyframes':model_report.get('keyframes'),'stages':model_report['stages'],'view_levels':model_report['view_levels'],
                'viewer':{'renderer':renderer.name,'triangles_drawn':int(len(view.faces)),'source':view.source,'screen_coverage':round(covered,3)},
                'error_mm':{'mean':round(float(errors.mean()),4),'p95':round(float(np.percentile(errors,95)),4),'max':round(float(errors.max()),4)},
                'seconds':round(wall,1),'stage_seconds':model_report.get('timings'),
                'gpu_util_note':'nvidia-smi cada 0,5 s; mosaicos pequeños limitan la ocupación, no indican CPU',
                'gpu_util_peak_pct':max(busy) if busy else None,'gpu_util_mean_pct':round(float(np.mean(busy)),1) if busy else None,
                'gpu_mem_peak_mb':max((m for *_,m in samples),default=None),'vram_pool_peak_mb':model_report.get('vram_pool_peak_mb'),
                'stl':{'path':str(stl),'triangles':count,'sha256':hashlib.sha256(stl.read_bytes()).hexdigest()},
                'camera_reference':camera.reference,'folder':model_report['folder']}
        for value in results.values(): value.pop('camera_reference',None)
        report['results']=results
        a,b=results['RTX sola'],results['RTX + IA']
        report['checks']={
            'both_results_independent':a['folder']!=b['folder'] and a['stl']['sha256']!=b['stl']['sha256'],
            'stl_matches_selected':a['stl']['triangles']==a['triangles'] and b['stl']['triangles']==b['triangles'],
            'viewer_solid_surface':min(a['viewer']['screen_coverage'],b['viewer']['screen_coverage'])>.05,
            'all_dense_stages_cuda':all(s['dispositivo'].startswith('CUDA') for r in (a,b) for s in r['stages'] if 'ICP' in s['etapa'] or 'TSDF' in s['etapa'] or 'Marching' in s['etapa'])}
        report['ok']=all(report['checks'].values())
    except Exception:
        import traceback
        report['ok']=False; report['error']=traceback.format_exc()
    report['finished']=time.strftime('%Y-%m-%d %H:%M:%S')
    (out/'validation-report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,default=str),encoding='utf-8')
    return 0 if report.get('ok') else 1


if __name__=='__main__':
    raise SystemExit(run(sys.argv[1],float(sys.argv[2]) if len(sys.argv)>2 else .2))
