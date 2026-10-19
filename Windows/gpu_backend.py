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
def cuda_module():
    if os.name=='nt':
        base=Path(sysconfig.get_paths()['purelib'])/'nvidia'
        for folder in ('cuda_runtime','cuda_nvrtc'):
            directory=base/folder/'bin'
            if directory.is_dir():
                _dll_handles.append(os.add_dll_directory(str(directory)))
                os.environ['PATH']=str(directory)+os.pathsep+os.environ.get('PATH','')
                import ctypes
                for builtin in directory.glob('nvrtc-builtins*.dll'):
                    _dll_handles.append(ctypes.WinDLL(str(builtin)))
        runtime=base/'cuda_runtime'
        if runtime.is_dir(): os.environ.setdefault('CUDA_PATH',str(runtime))
    import cupy as cp
    if cp.cuda.runtime.getDeviceCount()<1: raise RuntimeError('No hay GPU CUDA disponible')
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

    def mesh(self,min_weight=1.):
        cp=self.cp
        from skimage.measure import _marching_cubes_lewiner_luts as lut
        table=cp.asarray(np.frombuffer(base64.decodebytes(lut.CASESCLASSIC[1].encode()),np.int8))
        n=self.cells**3; counts=cp.empty(n,dtype=cp.int32); cases=cp.empty(n,dtype=cp.int32)
        cp.RawKernel(MARCH,'count_triangles')(((n+255)//256,),(256,),
          (self.field,self.weights,table,counts,cases,np.int32(self.cells),np.float32(min_weight)))
        offsets=cp.cumsum(counts,dtype=cp.int64)-counts
        total=int(cp.sum(counts).get())
        if not total: return np.empty((0,3,3),np.float32),np.empty((0,3,3),np.float32)
        vertices=cp.empty((total,3,3),cp.float32); colors=cp.empty_like(vertices)
        cp.RawKernel(MARCH,'emit_triangles')(((n+255)//256,),(256,),
          (self.field,self.colors,table,counts,cases,offsets,vertices,colors,np.int32(self.cells),
           *[np.float32(v) for v in self.origin],np.float32(self.voxel)))
        return vertices.get(),colors.get()


class TriangleCount:
    def __init__(self,count): self.count=count
    def __len__(self): return self.count


class TiledModel:
    def __init__(self,folder):
        self.folder=Path(folder); meta=json.loads((self.folder/'model.json').read_text())
        self.triangles=TriangleCount(meta['triangles']); self.total_vertices=meta['triangles']*3
        with np.load(self.folder/'preview.npz') as preview:
            self.vertices=preview['vertices']; self.vertex_colors=preview['colors']

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


def reconstruct_tiled(path,settings,progress,cancel):
    from dataclasses import replace
    import open3d as o3d
    from offline_pipeline import validate_session,records,prepare_frame,align_cloud,TensorFusion
    metadata=validate_session(path)
    folder=Path(path).parent/(Path(path).stem+'-cuda-'+__import__('uuid').uuid4().hex[:8])
    folder.mkdir(); inputs=folder/'inputs'; inputs.mkdir()
    coarse_settings=replace(settings,resolution_mm=max(.8,settings.resolution_mm*4),device='CPU',memory_mb=min(512,settings.memory_mb))
    coarse=TensorFusion(coarse_settings)
    previous,previous_imu,pose=None,None,np.eye(4)
    previous_images,previous_intrinsics=None,None
    accepted=[]; rejected=0
    for index,frame in enumerate(records(path)):
        if cancel is not None and cancel.is_set(): raise InterruptedError('Cancelado; sesión original conservada')
        if index%settings.stride: continue
        depth,rgb=prepare_frame(frame,settings); h,w=depth.shape; fx,fy,cx,cy=frame.intrinsics
        rgbd=o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(rgb),o3d.geometry.Image(depth),depth_scale=1.,depth_trunc=settings.max_depth_m,convert_rgb_to_intensity=False)
        intrinsic=o3d.camera.PinholeCameraIntrinsic(w,h,fx,fy,cx,cy)
        cloud=o3d.geometry.PointCloud.create_from_rgbd_image(rgbd,intrinsic).voxel_down_sample(settings.registration_mm/1000)
        valid,delta,fitness,rmse=len(cloud.points)>=100,np.eye(4),1.,0.
        if valid and previous is not None:
            initial=None
            if settings.neural_alignment:
                from ai_alignment import neural_seed
                initial=neural_seed((depth,rgb),previous_images,frame.intrinsics,previous_intrinsics)
            try: valid,delta,fitness,rmse=align_cloud(cloud,previous,previous_imu,frame.rotation,settings,initial)
            except ValueError: valid=False
        if valid:
            candidate=pose@delta
            coarse.integrate(depth,rgb,frame.intrinsics,candidate)
            pose,previous,previous_imu=candidate,cloud,frame.rotation
            previous_images,previous_intrinsics=(depth,rgb),frame.intrinsics
            name=f'{index:09d}'
            np.save(inputs/(name+'-depth.npy'),depth);np.save(inputs/(name+'-rgb.npy'),rgb)
            accepted.append((name,pose.copy(),frame.intrinsics.copy()))
        else: rejected+=1
        progress(index+1,metadata['frames'],len(accepted),rejected,'CPU: alineación global',fitness,rmse)
    if len(accepted)<2: raise ValueError('Pocos frames alineados para reconstruir')
    points=np.asarray(coarse.grid.cpu().extract_point_cloud(weight_threshold=max(1.,settings.min_observations)).to_legacy().points)
    if len(points)==0: raise ValueError('No se obtuvo superficie global consistente')
    cells=settings.tile_cells; voxel=settings.resolution_mm/1000; width=cells*voxel
    keys=np.unique(np.floor(points/width).astype(np.int64),axis=0)
    # Cover the coarse localization error without changing final voxel spacing.
    keys=np.unique(np.concatenate([keys+[x,y,z] for x in (-1,0,1) for y in (-1,0,1) for z in (-1,0,1)]),axis=0)
    del coarse,points
    cp=cuda_module(); total=0; samples=[]; sample_colors=[]
    for tile_index,key in enumerate(keys):
        if cancel is not None and cancel.is_set(): raise InterruptedError('Cancelado; mosaicos terminados y originales conservados')
        origin=key.astype(np.float64)*width; center=origin+width/2; radius=math.sqrt(3)*width/2+.004
        tile=CudaTile(cells,voxel,origin)
        for name,pose,intrinsics in accepted:
            point=np.linalg.inv(pose)@np.r_[center,1.]
            if point[2]<=0: continue
            depth=np.load(inputs/(name+'-depth.npy')); h,w=depth.shape; fx,fy,cx,cy=intrinsics
            u,v=fx*point[0]/point[2]+cx,fy*point[1]/point[2]+cy
            margin=max(fx,fy)*radius/point[2]
            if u+margin<0 or v+margin<0 or u-margin>=w or v-margin>=h: continue
            crop=depth[max(0,int(v-margin)):min(h,int(v+margin)+2),max(0,int(u-margin)):min(w,int(u+margin)+2)]
            valid=crop[crop>0]
            if len(valid)==0 or valid.min()>point[2]+radius or valid.max()<point[2]-radius: continue
            rgb=np.load(inputs/(name+'-rgb.npy'))
            tile.integrate(depth,rgb,intrinsics,pose)
        vertices,colors=tile.mesh(max(1.,settings.min_observations))
        if len(vertices):
            np.savez_compressed(folder/f'tile-{tile_index:08d}.npz',vertices=vertices,colors=colors)
            total+=len(vertices)
            flat=vertices.reshape(-1,3); cflat=colors.reshape(-1,3)
            step=max(1,len(flat)//30);samples.append(flat[::step][:30]);sample_colors.append(cflat[::step][:30])
            # Preview alone is bounded; full triangles stay in tile files.
            if sum(len(v) for v in samples)>12000:
                combined=np.concatenate(samples)[::2]; cc=np.concatenate(sample_colors)[::2];samples=[combined];sample_colors=[cc]
        del tile; cp.get_default_memory_pool().free_all_blocks()
        progress(tile_index+1,len(keys),len(accepted),rejected,f'CUDA mosaico {tile_index+1}/{len(keys)}',1.,0.)
    if total==0: raise ValueError('No se obtuvo malla fina consistente')
    np.savez_compressed(folder/'preview.npz',vertices=np.concatenate(samples),colors=np.concatenate(sample_colors))
    report={'settings':__import__('dataclasses').asdict(settings),'captured':metadata,'accepted':len(accepted),'rejected':rejected,
            'device':'CUDA: CuPy TSDF + Marching Cubes','vertices':total*3,'triangles':total,'tiles':len(keys),'folder':str(folder)}
    (folder/'model.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    # Prepared images are a reconstructible PC cache, never the source session.
    import shutil
    if inputs.resolve().parent!=folder.resolve() or inputs.name!='inputs': raise ValueError('Ruta de caché inválida')
    shutil.rmtree(inputs)
    return TiledModel(folder),report
