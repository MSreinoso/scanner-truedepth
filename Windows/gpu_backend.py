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

INTEGRATE_MANY=r'''
extern "C" __global__ void integrate_many(float* field, float* weights, float* colors,
 const unsigned long long* depth_ptrs, const unsigned long long* rgb_ptrs, const float* E, const float* K,
 const int* sizes, int count, int side, float ox, float oy, float oz, float voxel, float truncation) {
 // Same arithmetic and frame order as `integrate`, one launch for all visible frames.
 int i=blockDim.x*blockIdx.x+threadIdx.x, total=side*side*side;
 if(i>=total) return;
 int x=i%side,y=(i/side)%side,z=i/(side*side);
 float wx=ox+x*voxel,wy=oy+y*voxel,wz=oz+z*voxel;
 float f=field[i],w=weights[i],c0=colors[3*i],c1=colors[3*i+1],c2=colors[3*i+2];
 for(int k=0;k<count;k++){
  const float* e=E+12*k; const float* q=K+4*k;
  float px=e[0]*wx+e[1]*wy+e[2]*wz+e[3];
  float py=e[4]*wx+e[5]*wy+e[6]*wz+e[7];
  float pz=e[8]*wx+e[9]*wy+e[10]*wz+e[11];
  if(pz<=0) continue;
  int width=sizes[2*k],height=sizes[2*k+1];
  int u=__float2int_rn(q[0]*px/pz+q[2]),v=__float2int_rn(q[1]*py/pz+q[3]);
  if(u<0||v<0||u>=width||v>=height) continue;
  int pixel=v*width+u;
  float d=((const float*)depth_ptrs[k])[pixel],sdf=d-pz;
  if(!(d>0)||sdf < -truncation) continue;
  const unsigned char* rgb=(const unsigned char*)rgb_ptrs[k];
  float value=fminf(1.f,sdf/truncation);
  f=(f*w+value)/(w+1.f);
  c0=(c0*w+(float)rgb[3*pixel])/(w+1.f); c1=(c1*w+(float)rgb[3*pixel+1])/(w+1.f); c2=(c2*w+(float)rgb[3*pixel+2])/(w+1.f);
  w=w+1.f;
 }
 field[i]=f; weights[i]=w; colors[3*i]=c0; colors[3*i+1]=c1; colors[3*i+2]=c2;
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


_KERNELS={}
def _kernel(code,name):
    if (code,name) not in _KERNELS: _KERNELS[(code,name)]=cuda_module().RawKernel(code,name)
    return _KERNELS[(code,name)]


class CudaTile:
    def __init__(self,cells,voxel,origin):
        self.cp=cp=cuda_module(); self.cells=cells; self.voxel=voxel; self.origin=origin
        side=cells+1; self.side=side
        self.field=cp.ones(side**3,dtype=cp.float32); self.weights=cp.zeros(side**3,dtype=cp.float32)
        self.colors=cp.zeros((side**3,3),dtype=cp.float32)
        self.kernel=cp.RawKernel(INTEGRATE,'integrate')

    def integrate_many(self,frames):
        """frames: list of (gpu_depth, gpu_rgb, extrinsic 3x4 float32, intrinsics, width, height)."""
        cp=self.cp
        if not frames: return
        kernel=_kernel(INTEGRATE_MANY,'integrate_many')
        depth_ptrs=cp.asarray([f[0].data.ptr for f in frames],dtype=cp.uint64)
        rgb_ptrs=cp.asarray([f[1].data.ptr for f in frames],dtype=cp.uint64)
        E=cp.asarray(np.stack([f[2] for f in frames]).astype(np.float32))
        K=cp.asarray(np.array([f[3] for f in frames],dtype=np.float32))
        sizes=cp.asarray(np.array([(f[4],f[5]) for f in frames],dtype=np.int32))
        kernel(((self.side**3+255)//256,),(256,),(self.field,self.weights,self.colors,depth_ptrs,rgb_ptrs,E,K,sizes,np.int32(len(frames)),
            np.int32(self.side),*[np.float32(v) for v in self.origin],np.float32(self.voxel),np.float32(max(self.voxel*8,.002))))

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
        per_triangle=50 if path.suffix.lower()=='.stl' else 3*15+13
        need=84+per_triangle*len(self.triangles)
        if free_bytes(path.parent)<need+256*1024**2:
            raise DiskSpaceError(f'No hay espacio para exportar: el archivo ocupará ~{gb(need)} GB y hay {gb(free_bytes(path.parent))} GB libres en {path.parent}.')
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
                    rgb=colors if colors.dtype==np.uint8 else np.clip(colors*255,0,255).astype(np.uint8)
                    out=np.empty(len(vertices),dtype=dtype);out['xyz']=vertices;out['rgb']=rgb;file.write(out.tobytes())
                faces=np.dtype([('n','u1'),('indices','<i4',(3,))])
                for start in range(0,len(self.triangles),100000):
                    count=min(100000,len(self.triangles)-start);out=np.empty(count,dtype=faces);out['n']=3
                    out['indices']=np.arange(start*3,(start+count)*3,dtype=np.int32).reshape(-1,3);file.write(out.tobytes())
        else: raise ValueError('Formato de exportación desconocido')
        return True


LOD_STRIDES=(1,2,4,8,16)


def rotation_angle(transform):
    return math.degrees(math.acos(float(np.clip((np.trace(transform[:3,:3])-1)/2,-1,1))))


BYTES_PER_TRIANGLE=9*4+9      # float32 vertices + uint8 RGB per triangle in tile files
FRAME_INPUT_BYTES=640*480*(4+3)
DISK_MARGIN=2*1024**3


def free_bytes(folder):
    import shutil
    return shutil.disk_usage(folder).free


def estimate_mesh_bytes(surface_tiles,cells):
    """Upper-range estimate: a surface crossing a tile yields ~2*cells^2 triangles."""
    return int(surface_tiles*2*cells*cells*BYTES_PER_TRIANGLE*1.15)


class DiskSpaceError(OSError):
    pass


def gb(value): return f'{value/1024**3:.1f}'.replace('.',',')


def reconstruct_tiled(path,settings,progress,cancel):
    """Creates the result folder and removes it if reconstruction does not finish:
    an incomplete tile set cannot be viewed or exported and only fills the disk.
    The original session is never touched."""
    import shutil
    folder=Path(path).parent/(Path(path).stem+'-cuda-'+__import__('uuid').uuid4().hex[:8])
    folder.mkdir()
    try:
        return _reconstruct_tiled(path,settings,progress,cancel,folder)
    except BaseException:
        shutil.rmtree(folder,ignore_errors=True)
        raise


def _reconstruct_tiled(path,settings,progress,cancel,folder):
    """All dense 3D stages on the RTX. CPU: decoding, rectification, file I/O
    and per-tile frustum culling of a handful of scalars."""
    from capture_format import validate_session,records
    from offline_pipeline import prepare_frame,relative_rotation
    from cuda_registration import register,surface_tile_keys,unique_tiles
    from mesh_view import weld,VIEW_BUDGET
    metadata=validate_session(path)
    clock=__import__('time').perf_counter; t0=clock(); timings={}
    inputs=folder/'inputs'; inputs.mkdir()
    needed=metadata['frames']//max(1,settings.stride)*FRAME_INPUT_BYTES+DISK_MARGIN
    if free_bytes(folder)<needed:
        raise DiskSpaceError(f'Espacio insuficiente para preparar la reconstrucción: hacen falta ~{gb(needed)} GB y hay {gb(free_bytes(folder))} GB libres en el disco. Libera espacio; la sesión original se conserva.')
    cp=cuda_module()
    stages=[]
    cells=settings.tile_cells; voxel=settings.resolution_mm/1000; width=cells*voxel
    keys=cp.empty((0,3),dtype=cp.int64)
    # Keyframe registration: each frame aligns to the current keyframe (not to
    # the previous frame), so small per-frame errors do not add up. A new
    # keyframe is taken when overlap drops or the motion grows.
    from collections import deque
    key=None; last=None; recent=deque(maxlen=6)
    accepted=[]; frame_info=[]; rejected=0; keyframes=0; fallback=0; tracked=0
    for index,frame in enumerate(records(path)):
        if cancel is not None and cancel.is_set(): raise InterruptedError('Cancelado; sesión original conservada')
        if index%settings.stride: continue
        depth,rgb=prepare_frame(frame,settings); fx,fy,cx,cy=frame.intrinsics
        cloud=cp.asarray(depth)
        valid,delta,fitness,rmse=int(cp.count_nonzero(cloud))>=100,np.eye(4),1.,0.
        candidate=np.eye(4); promote=key is None
        if valid and key is not None:
            imu=np.eye(4); imu[:3,:3]=relative_rotation(last['imu'],frame.rotation)
            predicted=last['pose']@imu      # constant position + IMU rotation
            seed=None
            if settings.neural_alignment:
                from ai_alignment import neural_seed
                seed=neural_seed((depth,rgb),key['images'],frame.intrinsics,key['intrinsics'],require_cuda=True)
            # Current keyframe first; then the previous frame and recent
            # keyframes, so a single bad frame or fast motion does not end tracking.
            references=[key]+([last] if last is not key else [])+[k for k in reversed(recent) if k is not key][:3]
            valid=False; track=None
            for attempt,reference in enumerate(references):
                guess=seed if (attempt==0 and seed is not None) else np.linalg.inv(reference['pose'])@predicted
                try: ok,delta,fit,err=register(cloud,reference['depth'],frame.intrinsics,reference['intrinsics'],guess,settings)
                except ValueError: continue
                if ok:
                    valid=True; candidate=reference['pose']@delta; fitness,rmse=fit,err
                    if attempt==0:
                        promote=(fitness<min(.95,settings.min_fitness+.2) or np.linalg.norm(delta[:3,3])>.02 or rotation_angle(delta)>10)
                    else:
                        promote=True; fallback+=1
                    break
                # Looser tracking test: keeps the pose chain alive without
                # letting this frame into the fused geometry.
                if (track is None and np.isfinite(delta).all() and fit>=.6*settings.min_fitness and err<=2*settings.max_rmse_mm/1000
                        and np.linalg.norm(delta[:3,3])<=.1 and rotation_angle(delta)<=45):
                    track=reference['pose']@delta; fitness,rmse=fit,err
            if not valid and track is not None:
                tracked+=1
                last={'pose':track,'depth':cloud,'intrinsics':frame.intrinsics,'imu':frame.rotation,'images':(depth,rgb)}
                recent.append(last)
        if valid:
            frame_keys=surface_tile_keys(cloud,frame.intrinsics,candidate,width)
            keys=unique_tiles(cp.concatenate((keys,frame_keys)))
            current={'pose':candidate,'depth':cloud,'intrinsics':frame.intrinsics,'imu':frame.rotation,'images':(depth,rgb)}
            last=current
            if promote:
                key=current; keyframes+=1; recent.append(current)
            name=f'{index:09d}'
            np.save(inputs/(name+'-depth.npy'),depth);np.save(inputs/(name+'-rgb.npy'),rgb)
            accepted.append((name,candidate.copy(),frame.intrinsics.copy(),np.linalg.inv(candidate)))
            positive=depth[depth>0]
            frame_info.append(((float(positive.min()),float(positive.max())) if positive.size else (np.inf,-np.inf),depth.shape,depth.nbytes+rgb.nbytes))
        else: rejected+=1
        progress(index+1,metadata['frames'],len(accepted),rejected,'RTX · alineación ICP y localización de superficie (CUDA)',fitness,rmse)
    timings['alineacion_s']=round(clock()-t0,2); t1=clock()
    stages.append({'etapa':'Decodificación y rectificación RGB-D','dispositivo':'CPU'})
    stages.append({'etapa':'ICP punto-plano coarse-to-fine con keyframes','dispositivo':'CUDA'})
    stages.append({'etapa':'Localización de superficie en mosaicos','dispositivo':'CUDA'})
    if settings.rgb_guidance: stages.append({'etapa':'Filtro bilateral guiado por RGB','dispositivo':'CUDA'})
    if settings.neural_denoising: stages.append({'etapa':'Limpieza neuronal de profundidad','dispositivo':'CUDA (PyTorch)'})
    if settings.neural_alignment: stages.append({'etapa':'Semilla de alineación LoFTR','dispositivo':'CUDA (PyTorch)'})
    if len(accepted)<2: raise ValueError(f'Pocos frames alineados para reconstruir ({len(accepted)} aceptados, {rejected} rechazados). Revisa distancia, movimiento y parámetros ICP.')
    if len(keys)==0: raise ValueError('No se obtuvo superficie global consistente')
    estimate=estimate_mesh_bytes(int(len(keys)),cells)
    available=free_bytes(folder)-DISK_MARGIN
    if estimate>available:
        coarser=settings.resolution_mm*math.sqrt(estimate/max(available,1))*1.1
        raise DiskSpaceError(f'La malla a {settings.resolution_mm:g} mm se estima en ~{gb(estimate)} GB y hay {gb(available)} GB disponibles. '
                             f'No se reduce la resolución automáticamente: libera espacio, elige ≥{coarser:.2f} mm o reduce la distancia máxima en Avanzado. '
                             'La sesión original se conserva.')
    keys=unique_tiles(cp.concatenate([keys+cp.asarray([x,y,z]) for x in (-1,0,1) for y in (-1,0,1) for z in (-1,0,1)]))
    keys=cp.asnumpy(keys)
    total=0
    lods={k:([],[],0) for k in LOD_STRIDES if cells%k==0}
    from collections import OrderedDict
    cache=OrderedDict();cache_bytes=0
    cache_limit=max(64*1024**2,min(settings.memory_mb*1024**2,cp.cuda.runtime.memGetInfo()[0]//3))
    def prepared(name):
        nonlocal cache_bytes
        if name in cache:
            cache.move_to_end(name);return cache[name]
        depth=np.load(inputs/(name+'-depth.npy'));rgb=np.load(inputs/(name+'-rgb.npy'))
        size=depth.nbytes+rgb.nbytes
        while cache and cache_bytes+size>cache_limit:
            _,old=cache.popitem(last=False);cache_bytes-=old[2].nbytes+old[3].nbytes
        value=(None,None,cp.asarray(depth),cp.asarray(rgb))
        cache[name]=value;cache_bytes+=size
        return value
    # Vectorized visibility: one matrix product per tile for all frames.
    inverses=np.stack([entry[3] for entry in accepted])                 # (N,4,4) world->camera
    intrinsics_all=np.array([entry[2] for entry in accepted],dtype=np.float64)
    extrinsics=inverses[:,:3,:].astype(np.float32)
    depth_range=np.array([info[0] for info in frame_info]); frame_bytes=[info[2] for info in frame_info]
    shapes=np.array([info[1] for info in frame_info]); heights,widths=shapes[:,0],shapes[:,1]
    peak=0; launches=0
    for tile_index,tile_key in enumerate(keys):
        if cancel is not None and cancel.is_set(): raise InterruptedError('Cancelado; resultado incompleto descartado y sesión original conservada')
        origin=tile_key.astype(np.float64)*width; center=origin+width/2; radius=math.sqrt(3)*width/2+.004
        tile=CudaTile(cells,voxel,origin)
        point=inverses[:,:3,:3]@center+inverses[:,:3,3]
        z=point[:,2]; safe=np.maximum(z,1e-9)
        u=intrinsics_all[:,0]*point[:,0]/safe+intrinsics_all[:,2]; v=intrinsics_all[:,1]*point[:,1]/safe+intrinsics_all[:,3]
        margin=np.maximum(intrinsics_all[:,0],intrinsics_all[:,1])*radius/safe
        visible=np.flatnonzero((z>0)&(u+margin>=0)&(v+margin>=0)&(u-margin<widths)&(v-margin<heights)
                               &(depth_range[:,0]<=z+radius)&(depth_range[:,1]>=z-radius))
        # Chunks bounded by the GPU image cache; order preserved (deterministic).
        start=0
        while start<len(visible):
            chunk=[]; used=0
            while start<len(visible) and (not chunk or used+frame_bytes[visible[start]]<=cache_limit):
                chunk.append(visible[start]); used+=frame_bytes[visible[start]]; start+=1
            frames=[]
            for k in chunk:
                _,_,gpu_depth,gpu_rgb=prepared(accepted[k][0])
                frames.append((gpu_depth,gpu_rgb,extrinsics[k],accepted[k][2],int(widths[k]),int(heights[k])))
            tile.integrate_many(frames); launches+=1
        minimum=max(1.,settings.min_observations)
        vertices,colors=tile.mesh(minimum)
        if len(vertices):
            if tile_index%25==0 and free_bytes(folder)<DISK_MARGIN//2:
                raise DiskSpaceError(f'Disco casi lleno durante la fusión ({gb(free_bytes(folder))} GB libres). Resultado incompleto descartado; la sesión original se conserva.')
            np.savez(folder/f'tile-{tile_index:08d}.npz',vertices=vertices,colors=np.clip(colors*255+.5,0,255).astype(np.uint8))
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
            'keyframes':keyframes,'keyframe_fallbacks':int(fallback),'tracking_only_frames':int(tracked),
            'device':'CUDA: ICP + localización + TSDF + Marching Cubes',
            'gpu_name':cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
            'stages':stages,'cpu_tasks':['decodificación y rectificación RGB-D','selección de vistas por mosaico (escalares)','E/S de disco'],
            'disk_estimate_gb':round(estimate/1024**3,2),'vram_pool_peak_mb':round(peak/1024**2,1),'timings':timings,'fusion_kernel_launches':launches,
            'registration':'CUDA projective point-to-plane coarse-to-fine, alineación a keyframe',
            'resolution_note':'resolution_mm es el tamaño de vóxel de la reconstrucción, no la precisión física del sensor TrueDepth.',
            'vertices':total*3,'triangles':total,'tiles':len(keys),'view_levels':saved,'folder':str(folder)}
    (folder/'model.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    # Prepared images are a reconstructible PC cache, never the source session.
    import shutil
    if inputs.resolve().parent!=folder.resolve() or inputs.name!='inputs': raise ValueError('Ruta de caché inválida')
    shutil.rmtree(inputs)
    return TiledModel(folder),report
