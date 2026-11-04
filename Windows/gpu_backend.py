"""CUDA tiled TSDF + Marching Cubes. VRAM is bounded independently of scene size."""
from pathlib import Path
import base64
import json
import math
import os
import struct
import sysconfig
import numpy as np

_dll_handles=[]
_cupy=None
def cuda_module():
    global _cupy
    if _cupy is not None: return _cupy
    if os.name=='nt':

        base=Path(getattr(__import__('sys'),'_MEIPASS',sysconfig.get_paths()['purelib']))/'nvidia'
        # CuPy registration uses cuBLAS/cuSOLVER shipped with the unified Torch
        # runtime even when neural inference is disabled.
        linear_algebra=base.parent/'torch'/'lib'
        if linear_algebra.is_dir():
            _dll_handles.append(os.add_dll_directory(str(linear_algebra)))
            os.environ['PATH']=str(linear_algebra)+os.pathsep+os.environ.get('PATH','')
        for folder in ('cuda_runtime','cuda_nvrtc'):
            directory=base/folder/'bin'
            if directory.is_dir():
                _dll_handles.append(os.add_dll_directory(str(directory)))
                os.environ['PATH']=str(directory)+os.pathsep+os.environ.get('PATH','')
                import ctypes
                for builtin in directory.glob('nvrtc-builtins*.dll'):
                    _dll_handles.append(ctypes.WinDLL(str(builtin)))
        runtime=base/'cuda_runtime'
        if runtime.is_dir(): os.environ['CUDA_PATH']=str(runtime)
    import cupy as cp
    if cp.cuda.runtime.getDeviceCount()<1: raise RuntimeError('No hay GPU CUDA disponible')
    _cupy=cp
    return cp


INTEGRATE=r'''
extern "C" __global__ void integrate(float* field, float* weights, float* colors,
 const float* depth, const unsigned char* rgb, const float* E, const float* K,
 int width, int height, int side, float ox, float oy, float oz, float voxel, float truncation) {
 int i=blockDim.x*blockIdx.x+threadIdx.x, total=side*side*side;
 if(i>=total) return;
 int x=i%side,y=(i/side)%side,z=i/(side*side);
 float wx=ox+x*voxel,wy=oy+y*voxel,wz=oz+z*voxel;
 float px=E[0]*wx+E[1]*wy+E[2]*wz+E[3];
 float py=E[4]*wx+E[5]*wy+E[6]*wz+E[7];
 float pz=E[8]*wx+E[9]*wy+E[10]*wz+E[11];
 if(pz<=0) return;
 int u=__float2int_rn(K[0]*px/pz+K[2]),v=__float2int_rn(K[1]*py/pz+K[3]);
 if(u<0||v<0||u>=width||v>=height) return;
 int pixel=v*width+u; float d=depth[pixel],sdf=d-pz;
 if(!(d>0)||sdf < -truncation) return;
 float value=fminf(1.f,sdf/truncation),w=weights[i];
 field[i]=(field[i]*w+value)/(w+1.f); weights[i]=w+1.f;
 for(int c=0;c<3;c++) colors[3*i+c]=(colors[3*i+c]*w+(float)rgb[3*pixel+c])/(w+1.f);
}
'''

MARCH=r'''
__device__ int corner(int cell,int c,int cells) {
 int side=cells+1,x=cell%cells,y=(cell/cells)%cells,z=cell/(cells*cells);
 const int dx[8]={0,1,1,0,0,1,1,0},dy[8]={0,0,1,1,0,0,1,1},dz[8]={0,0,0,0,1,1,1,1};
 return (x+dx[c])+side*((y+dy[c])+side*(z+dz[c]));
}
extern "C" __global__ void count_triangles(const float* field,const float* weights,const signed char* table,
 int* counts,int* cases,int cells,float min_weight) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i>=cells*cells*cells) return;
 int code=0;
 for(int c=0;c<8;c++){int index=corner(i,c,cells);if(weights[index]<min_weight){counts[i]=0;cases[i]=0;return;}if(field[index]<0)code|=1<<c;}
 int n=0;for(int j=0;j<16&&table[code*16+j]>=0;j+=3)n++;
 counts[i]=n;cases[i]=code;
}
extern "C" __global__ void emit_triangles(const float* field,const float* colors,const signed char* table,
 const int* counts,const int* cases,const long long* offsets,float* vertices,float* outcolors,
 int cells,float ox,float oy,float oz,float voxel) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=cells*cells*cells||counts[i]==0)return;
 const int edge_a[12]={0,1,2,3,4,5,6,7,0,1,2,3},edge_b[12]={1,2,3,0,5,6,7,4,4,5,6,7};
 int side=cells+1;
 for(int j=0;j<counts[i]*3;j++){
 int edge=table[cases[i]*16+j],a=corner(i,edge_a[edge],cells),b=corner(i,edge_b[edge],cells);
 float fa=field[a],fb=field[b],t=fabsf(fa-fb)>1e-12f?fa/(fa-fb):.5f;t=fminf(1.f,fmaxf(0.f,t));
 int ax=a%side,ay=(a/side)%side,az=a/(side*side),bx=b%side,by=(b/side)%side,bz=b/(side*side);
 long long out=(offsets[i]+j/3)*9+(j%3)*3;
 vertices[out]=ox+(ax+t*(bx-ax))*voxel;vertices[out+1]=oy+(ay+t*(by-ay))*voxel;vertices[out+2]=oz+(az+t*(bz-az))*voxel;
 for(int c=0;c<3;c++)outcolors[out+c]=(colors[3*a+c]+t*(colors[3*b+c]-colors[3*a+c]))/255.f;
 }
}
'''


class CudaTile:
    def __init__(self,cells,voxel,origin):
        self.cp=cp=cuda_module(); self.cells=cells; self.voxel=voxel; self.origin=origin
        side=cells+1; self.side=side
        self.field=cp.ones(side**3,dtype=cp.float32); self.weights=cp.zeros(side**3,dtype=cp.float32)
        self.colors=cp.zeros((side**3,3),dtype=cp.float32)
        self.kernel=cp.RawKernel(INTEGRATE,'integrate')

    def integrate(self,depth,rgb,intrinsics,pose):
        cp=self.cp; h,w=depth.shape
        E=cp.asarray(np.linalg.inv(pose).astype(np.float32)); K=cp.asarray(intrinsics,dtype=cp.float32)
        self.kernel(((self.side**3+255)//256,),(256,),
          (self.field,self.weights,self.colors,cp.asarray(depth,dtype=cp.float32),cp.asarray(rgb,dtype=cp.uint8),E,K,
           np.int32(w),np.int32(h),np.int32(self.side),*[np.float32(v) for v in self.origin],np.float32(self.voxel),np.float32(max(self.voxel*8,.002))))

    def mesh(self,min_weight=1.,stride=1):
        """Marching Cubes in CUDA. stride>1 samples the same TSDF every `stride`
        voxels: a coarser but coherent surface used only for the viewer."""
        cp=self.cp
        from skimage.measure import _marching_cubes_lewiner_luts as lut
        table=cp.asarray(np.frombuffer(base64.decodebytes(lut.CASESCLASSIC[1].encode()),np.int8))
        field,weights,colors,cells=self.field,self.weights,self.colors,self.cells
        if stride>1:
            if cells%stride: raise ValueError('Paso de LOD incompatible con el mosaico')
            side=self.side
            pick=lambda a,c=1: cp.ascontiguousarray(a.reshape((side,side,side)+((c,) if c>1 else ()))[::stride,::stride,::stride]).reshape((-1,c) if c>1 else (-1,))
            field,weights,colors=pick(field),pick(weights),pick(colors,3); cells//=stride
        n=cells**3; counts=cp.empty(n,dtype=cp.int32); cases=cp.empty(n,dtype=cp.int32)
        cp.RawKernel(MARCH,'count_triangles')(((n+255)//256,),(256,),
          (field,weights,table,counts,cases,np.int32(cells),np.float32(min_weight)))
        offsets=cp.cumsum(counts,dtype=cp.int64)-counts
        total=int(cp.sum(counts).get())
        if not total: return np.empty((0,3,3),np.float32),np.empty((0,3,3),np.float32)
        vertices=cp.empty((total,3,3),cp.float32); out_colors=cp.empty_like(vertices)
        cp.RawKernel(MARCH,'emit_triangles')(((n+255)//256,),(256,),
          (field,colors,table,counts,cases,offsets,vertices,out_colors,np.int32(cells),
           *[np.float32(v) for v in self.origin],np.float32(self.voxel*stride)))
        return vertices.get(),out_colors.get()


class TriangleCount:
    def __init__(self,count): self.count=count
    def __len__(self): return self.count


class TiledModel:
    """Full-resolution triangles live in tile-*.npz (exported unchanged).
    view-*.npz hold welded display LODs derived from the same TSDF."""
    def __init__(self,folder):
        self.folder=Path(folder); self.meta=meta=json.loads((self.folder/'model.json').read_text())
        self.triangles=TriangleCount(meta['triangles']); self.total_vertices=meta['triangles']*3

    def view_levels(self):
        levels=[]
        for path in self.folder.glob('view-*.npz'):
            with np.load(path) as data: levels.append((int(data['stride']),int(len(data['faces'])),path))
        return sorted(levels)

    def view_mesh(self,budget=None):
        from mesh_view import ViewMesh,VIEW_BUDGET
        budget=budget or VIEW_BUDGET
        levels=self.view_levels()
        if not levels: raise ValueError('Este resultado no tiene malla de visualización; reconstruye con la versión 3.3.')
        fitting=[level for level in levels if level[1]<=budget]
        stride,count,path=fitting[0] if fitting else levels[-1]
        with np.load(path) as data:
            colors=data['colors'] if data['colors'].size else None
            mesh=ViewMesh(data['vertices'],data['faces'],data['normals'].astype(np.float32),colors,
                'malla completa' if stride==1 else f'LOD 1/{stride} del mismo TSDF (vóxel {self.meta["settings"]["resolution_mm"]*stride:.3g} mm)',
                len(self.triangles))
        return mesh

    def export_mm(self,path):
        path=Path(path); parts=sorted(self.folder.glob('tile-*.npz'))
        if path.suffix.lower()=='.stl':
            if len(self.triangles)>0xffffffff: raise ValueError('Demasiados triángulos para STL binario; usa PLY')
            with open(path,'wb') as file:
                file.write(b'Bichito CUDA tiles; units=mm'.ljust(80,b'\0')); file.write(struct.pack('<I',len(self.triangles)))
                dtype=np.dtype([('normal','<f4',(3,)),('vertices','<f4',(3,3)),('attribute','<u2')])
                for part in parts:
                    with np.load(part) as data: vertices=data['vertices']*1000
                    normal=np.cross(vertices[:,1]-vertices[:,0],vertices[:,2]-vertices[:,0])
                    normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-20)
                    records=np.zeros(len(vertices),dtype=dtype);records['normal']=normal;records['vertices']=vertices
                    file.write(records.tobytes())
        elif path.suffix.lower()=='.ply':
            vertex_count=len(self.triangles)*3
            if vertex_count>0x7fffffff: raise ValueError('Exporta por mosaicos: el modelo supera el índice PLY de 32 bits')
            header=f'ply\nformat binary_little_endian 1.0\ncomment units mm\nelement vertex {vertex_count}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nelement face {len(self.triangles)}\nproperty list uchar int vertex_indices\nend_header\n'
            with open(path,'wb') as file:
                file.write(header.encode())
                dtype=np.dtype([('xyz','<f4',(3,)),('rgb','u1',(3,))])
                for part in parts:
                    with np.load(part) as data:
                        vertices=data['vertices'].reshape(-1,3)*1000; colors=data['colors'].reshape(-1,3)
                    out=np.empty(len(vertices),dtype=dtype);out['xyz']=vertices;out['rgb']=np.clip(colors*255,0,255).astype(np.uint8);file.write(out.tobytes())
                faces=np.dtype([('n','u1'),('indices','<i4',(3,))])
                for start in range(0,len(self.triangles),100000):
                    count=min(100000,len(self.triangles)-start);out=np.empty(count,dtype=faces);out['n']=3
                    out['indices']=np.arange(start*3,(start+count)*3,dtype=np.int32).reshape(-1,3);file.write(out.tobytes())
        else: raise ValueError('Formato de exportación desconocido')
        return True


LOD_STRIDES=(1,2,4,8,16)


def rotation_angle(transform):
    return math.degrees(math.acos(float(np.clip((np.trace(transform[:3,:3])-1)/2,-1,1))))


def reconstruct_tiled(path,settings,progress,cancel):
    """All dense 3D stages on the RTX. CPU: decoding, rectification, file I/O
    and per-tile frustum culling of a handful of scalars."""
    from capture_format import validate_session,records
    from offline_pipeline import prepare_frame,relative_rotation
    from cuda_registration import register,surface_tile_keys,unique_tiles
    from mesh_view import weld,VIEW_BUDGET
    metadata=validate_session(path)
    clock=__import__('time').perf_counter; t0=clock(); timings={}
    folder=Path(path).parent/(Path(path).stem+'-cuda-'+__import__('uuid').uuid4().hex[:8])
    folder.mkdir(); inputs=folder/'inputs'; inputs.mkdir()
    cp=cuda_module()
    stages=[]
    cells=settings.tile_cells; voxel=settings.resolution_mm/1000; width=cells*voxel
    keys=cp.empty((0,3),dtype=cp.int64)
    # Keyframe registration: each frame aligns to the current keyframe (not to
    # the previous frame), so small per-frame errors do not add up. A new
    # keyframe is taken when overlap drops or the motion grows.
    key=None; last=None
    accepted=[]; rejected=0; keyframes=0; fallback=0
    for index,frame in enumerate(records(path)):
        if cancel is not None and cancel.is_set(): raise InterruptedError('Cancelado; sesión original conservada')
        if index%settings.stride: continue
        depth,rgb=prepare_frame(frame,settings); fx,fy,cx,cy=frame.intrinsics
        cloud=cp.asarray(depth)
        valid,delta,fitness,rmse=int(cp.count_nonzero(cloud))>=100,np.eye(4),1.,0.
        candidate=np.eye(4); promote=key is None
        if valid and key is not None:
            imu=np.eye(4); imu[:3,:3]=relative_rotation(last['imu'],frame.rotation)
            guess=np.linalg.inv(key['pose'])@last['pose']@imu
            if settings.neural_alignment:
                from ai_alignment import neural_seed
                seed=neural_seed((depth,rgb),key['images'],frame.intrinsics,key['intrinsics'],require_cuda=True)
                if seed is not None: guess=seed
            try: valid,delta,fitness,rmse=register(cloud,key['depth'],frame.intrinsics,key['intrinsics'],guess,settings)
            except ValueError: valid=False
            candidate=key['pose']@delta
            if valid:
                promote=(fitness<min(.95,settings.min_fitness+.2) or np.linalg.norm(delta[:3,3])>.02 or rotation_angle(delta)>10)
            elif last is not key:
                # Keyframe out of reach: try the immediately previous frame, then promote.
                try: valid,delta,fitness,rmse=register(cloud,last['depth'],frame.intrinsics,last['intrinsics'],imu,settings)
                except ValueError: valid=False
                candidate=last['pose']@delta; promote=valid; fallback+=valid
        if valid:
            frame_keys=surface_tile_keys(cloud,frame.intrinsics,candidate,width)
            keys=unique_tiles(cp.concatenate((keys,frame_keys)))
            current={'pose':candidate,'depth':cloud,'intrinsics':frame.intrinsics,'imu':frame.rotation,'images':(depth,rgb)}
            last=current
            if promote: key=current; keyframes+=1
            name=f'{index:09d}'
            np.save(inputs/(name+'-depth.npy'),depth);np.save(inputs/(name+'-rgb.npy'),rgb)
            accepted.append((name,candidate.copy(),frame.intrinsics.copy(),np.linalg.inv(candidate)))
        else: rejected+=1
        progress(index+1,metadata['frames'],len(accepted),rejected,'RTX · alineación ICP y localización de superficie (CUDA)',fitness,rmse)
    timings['alineacion_s']=round(clock()-t0,2); t1=clock()
    stages.append({'etapa':'Decodificación y rectificación RGB-D','dispositivo':'CPU'})
    stages.append({'etapa':'ICP punto-plano coarse-to-fine con keyframes','dispositivo':'CUDA'})
    stages.append({'etapa':'Localización de superficie en mosaicos','dispositivo':'CUDA'})
    if settings.neural_denoising: stages.append({'etapa':'Limpieza neuronal de profundidad','dispositivo':'CUDA (PyTorch)'})
    if settings.neural_alignment: stages.append({'etapa':'Semilla de alineación LoFTR','dispositivo':'CUDA (PyTorch)'})
    if len(accepted)<2: raise ValueError(f'Pocos frames alineados para reconstruir ({len(accepted)} aceptados, {rejected} rechazados). Revisa distancia, movimiento y parámetros ICP.')
    if len(keys)==0: raise ValueError('No se obtuvo superficie global consistente')
    keys=unique_tiles(cp.concatenate([keys+cp.asarray([x,y,z]) for x in (-1,0,1) for y in (-1,0,1) for z in (-1,0,1)]))
    keys=cp.asnumpy(keys)
    total=0
    lods={k:([],[],0) for k in LOD_STRIDES if cells%k==0}
    from collections import OrderedDict
    cache=OrderedDict();cache_bytes=0
    cache_limit=min(settings.memory_mb*1024**2//2,cp.cuda.runtime.memGetInfo()[0]//4)
    def prepared(name):
        nonlocal cache_bytes
        if name in cache:
            cache.move_to_end(name);return cache[name]
        depth=np.load(inputs/(name+'-depth.npy'));rgb=np.load(inputs/(name+'-rgb.npy'))
        size=depth.nbytes+rgb.nbytes
        while cache and cache_bytes+size>cache_limit:
            _,old=cache.popitem(last=False);cache_bytes-=old[0].nbytes+old[1].nbytes
        value=(depth,rgb,cp.asarray(depth),cp.asarray(rgb))
        cache[name]=value;cache_bytes+=size
        return value
    peak=0
    for tile_index,tile_key in enumerate(keys):
        if cancel is not None and cancel.is_set(): raise InterruptedError('Cancelado; mosaicos terminados y originales conservados')
        origin=tile_key.astype(np.float64)*width; center=origin+width/2; radius=math.sqrt(3)*width/2+.004
        tile=CudaTile(cells,voxel,origin)
        for name,pose,intrinsics,inverse in accepted:
            point=inverse@np.r_[center,1.]
            if point[2]<=0: continue
            depth,rgb,gpu_depth,gpu_rgb=prepared(name); h,w=depth.shape; fx,fy,cx,cy=intrinsics
            u,v=fx*point[0]/point[2]+cx,fy*point[1]/point[2]+cy
            margin=max(fx,fy)*radius/point[2]
            if u+margin<0 or v+margin<0 or u-margin>=w or v-margin>=h: continue
            crop=depth[max(0,int(v-margin)):min(h,int(v+margin)+2),max(0,int(u-margin)):min(w,int(u+margin)+2)]
            valid=crop[crop>0]
            if len(valid)==0 or valid.min()>point[2]+radius or valid.max()<point[2]-radius: continue
            tile.integrate(gpu_depth,gpu_rgb,intrinsics,pose)
        minimum=max(1.,settings.min_observations)
        vertices,colors=tile.mesh(minimum)
        if len(vertices):
            np.savez(folder/f'tile-{tile_index:08d}.npz',vertices=vertices,colors=colors)
            total+=len(vertices)
        for k in list(lods):
            v,c=(vertices,colors) if k==1 else tile.mesh(minimum,k)
            parts,cparts,count=lods[k]
            if len(v):
                parts.append(v); cparts.append(np.clip(c*255+.5,0,255).astype(np.uint8)); count+=len(v)
            if count>VIEW_BUDGET: del lods[k]   # too dense for display; coarser levels remain
            else: lods[k]=(parts,cparts,count)
        peak=max(peak,cp.get_default_memory_pool().total_bytes())
        del tile
        progress(tile_index+1,len(keys),len(accepted),rejected,f'RTX · TSDF + Marching Cubes CUDA, mosaico {tile_index+1}/{len(keys)}',1.,0.)
    if total==0: raise ValueError('No se obtuvo malla fina consistente')
    timings['tsdf_marching_cubes_s']=round(clock()-t1,2); t2=clock()
    stages.append({'etapa':'TSDF por mosaicos','dispositivo':'CUDA'})
    stages.append({'etapa':'Marching Cubes (malla completa y LOD de visor)','dispositivo':'CUDA'})
    saved=[]
    for k,(parts,cparts,count) in sorted(lods.items()):
        if not count: continue
        V,F,N,C=weld(np.concatenate(parts),np.concatenate(cparts),quantum=voxel*k/64,xp=cp)
        np.savez(folder/f'view-{k:02d}.npz',stride=k,vertices=V,faces=F,normals=N.astype(np.float16),colors=C)
        saved.append({'paso':k,'triangulos':int(len(F))})
    timings['malla_visor_s']=round(clock()-t2,2)
    stages.append({'etapa':'Soldadura de la malla de visualización','dispositivo':'CUDA'})
    stages.append({'etapa':'Escritura de mosaicos y exportación','dispositivo':'CPU (E/S)'})
    report={'settings':__import__('dataclasses').asdict(settings),'captured':metadata,'accepted':len(accepted),'rejected':rejected,
            'keyframes':keyframes,'keyframe_fallbacks':int(fallback),
            'device':'CUDA: ICP + localización + TSDF + Marching Cubes',
            'gpu_name':cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
            'stages':stages,'cpu_tasks':['decodificación y rectificación RGB-D','selección de vistas por mosaico (escalares)','E/S de disco'],
            'vram_pool_peak_mb':round(peak/1024**2,1),'timings':timings,
            'registration':'CUDA projective point-to-plane coarse-to-fine, alineación a keyframe',
            'resolution_note':'resolution_mm es el tamaño de vóxel de la reconstrucción, no la precisión física del sensor TrueDepth.',
            'vertices':total*3,'triangles':total,'tiles':len(keys),'view_levels':saved,'folder':str(folder)}
    (folder/'model.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    # Prepared images are a reconstructible PC cache, never the source session.
    import shutil
    if inputs.resolve().parent!=folder.resolve() or inputs.name!='inputs': raise ValueError('Ruta de caché inválida')
    shutil.rmtree(inputs)
    return TiledModel(folder),report
