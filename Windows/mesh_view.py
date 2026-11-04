"""Solid triangle-mesh viewer for reconstruction results.

The viewer never touches the exported geometry. Dense CUDA results are shown
through a level of detail (LOD) that is itself a coherent triangle surface:
marching cubes run on a sub-sampled copy of the same TSDF tile, then welded.

Rendering: CUDA z-buffer rasterizer (CuPy) with per-pixel smooth shading.
Without CUDA a CPU painter's-algorithm renderer draws the same triangles.
"""
from dataclasses import dataclass
import math
import numpy as np

VIEW_BUDGET = 2_500_000      # triangles kept for the GPU viewer
CPU_BUDGET = 60_000          # triangles drawn by the CPU fallback


@dataclass
class ViewMesh:
    vertices: np.ndarray     # (n,3) float32, metres, capture frame of the first camera
    faces: np.ndarray        # (m,3) int32
    normals: np.ndarray      # (n,3) float32 unit
    colors: np.ndarray       # (n,3) uint8 or None
    source: str              # e.g. "malla completa" / "LOD 1/4 (mismo TSDF)"
    full_triangles: int      # triangles of the exported mesh

    @property
    def center(self):
        return (self.vertices.min(0)+self.vertices.max(0))/2

    @property
    def radius(self):
        return float(max(np.linalg.norm(self.vertices-self.center,axis=1).max(),1e-6))


def weld(triangles, colors=None, quantum=None, xp=np):
    """Triangle soup (m,3,3) -> indexed mesh with area-weighted vertex normals.

    `xp` may be numpy or cupy. Vertices closer than `quantum` merge; marching
    cubes emits shared edge points from neighbouring cells with tiny float
    differences, so the quantum is a small fraction of the voxel size.
    """
    tri=xp.asarray(triangles,dtype=xp.float32).reshape(-1,3,3)
    m=len(tri)
    if m==0: raise ValueError('Malla vacía')
    flat=tri.reshape(-1,3)
    low=flat.min(0)
    span=float((flat.max(0)-low).max())
    if quantum is None or quantum<=0: quantum=max(span,1e-9)/2**20
    quantum=max(float(quantum),span/(2**21-2))
    q=xp.floor((flat-low)/quantum+.5).astype(xp.int64)
    keys=(q[:,0]<<42)|(q[:,1]<<21)|q[:,2]
    unique,first,inverse=xp.unique(keys,return_index=True,return_inverse=True)
    inverse=inverse.reshape(-1).astype(xp.int64)
    vertices=flat[first]
    faces=inverse.reshape(m,3)
    keep=(faces[:,0]!=faces[:,1])&(faces[:,1]!=faces[:,2])&(faces[:,0]!=faces[:,2])
    faces=faces[keep]
    corners=vertices[faces]
    face_normals=xp.cross(corners[:,1]-corners[:,0],corners[:,2]-corners[:,0])
    n=len(vertices); index=faces.reshape(-1)
    normals=xp.stack([xp.bincount(index,weights=xp.repeat(face_normals[:,c],3),minlength=n) for c in range(3)],axis=1)
    normals/=xp.maximum(xp.linalg.norm(normals,axis=1,keepdims=True),1e-20)
    vertex_colors=None
    if colors is not None:
        c=xp.asarray(colors).reshape(-1,3)
        if c.dtype.kind=='f': c=xp.clip(c*255+.5,0,255)
        vertex_colors=c[first].astype(xp.uint8)
    to_host=(lambda a:a.get()) if xp is not np else (lambda a:a)
    return (to_host(vertices).astype(np.float32),to_host(faces).astype(np.int32),
            to_host(normals).astype(np.float32),None if vertex_colors is None else to_host(vertex_colors))


def from_open3d(mesh,budget=VIEW_BUDGET):
    """CPU-path results (Open3D TriangleMesh). Decimation keeps a surface."""
    full=len(mesh.triangles); source='malla completa'
    if full>budget:
        extent=float(np.max(mesh.get_max_bound()-mesh.get_min_bound()))
        size=extent/math.sqrt(budget/2)
        mesh=mesh.simplify_vertex_clustering(voxel_size=size)
        source=f'LOD agrupación {size*1000:.2f} mm'
    vertices=np.asarray(mesh.vertices,np.float32); faces=np.asarray(mesh.triangles,np.int32)
    if not mesh.has_vertex_normals(): mesh.compute_vertex_normals()
    normals=np.asarray(mesh.vertex_normals,np.float32)
    colors=(np.clip(np.asarray(mesh.vertex_colors)*255+.5,0,255).astype(np.uint8) if mesh.has_vertex_colors() else None)
    return ViewMesh(vertices,faces,normals,colors,source,full)


def view_mesh_for(model):
    if hasattr(model,'view_mesh'): return model.view_mesh()
    return from_open3d(model)


class Camera:
    """Turntable orbit around a fixed reference; changes only through user input."""
    FOV=math.radians(30)

    def __init__(self):
        self.reference=None   # (center, radius) kept while comparing results
        self.reset()

    def reset(self):
        self.yaw=0.; self.pitch=0.; self.zoom=1.; self.pan=np.zeros(2)

    def frame(self,mesh,keep=False):
        if not keep or self.reference is None:
            self.reference=(mesh.center.astype(np.float64),mesh.radius)
            self.reset()

    def rotation(self):
        cy,sy=math.cos(self.yaw),math.sin(self.yaw); cp,sp=math.cos(self.pitch),math.sin(self.pitch)
        Ry=np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]]); Rx=np.array([[1,0,0],[0,cp,-sp],[0,sp,cp]])
        # TrueDepth camera axes (x right, y down, z forward) -> viewer looking from the phone.
        return Rx@Ry@np.diag([1.,-1.,-1.])

    def distance(self):
        _,radius=self.reference
        return radius/math.sin(self.FOV/2)*1.05/self.zoom

    def focal(self,width,height):
        return .5*min(width,height)/math.tan(self.FOV/2)

    def target(self):
        center,_=self.reference
        R=self.rotation()
        return center+R.T@np.array([self.pan[0],self.pan[1],0.])

    def orbit(self,dx,dy):
        self.yaw+=dx*.008; self.pitch=float(np.clip(self.pitch+dy*.008,-1.55,1.55))

    def pan_pixels(self,dx,dy,width,height):
        scale=self.distance()/self.focal(width,height)
        self.pan+=np.array([-dx*scale,dy*scale])

    def zoom_by(self,factor):
        self.zoom=float(np.clip(self.zoom*factor,.05,200))

    def project(self,vertices,width,height,xp=np):
        R=xp.asarray(self.rotation(),dtype=xp.float32)
        q=(xp.asarray(vertices,dtype=xp.float32)-xp.asarray(self.target(),dtype=xp.float32))@R.T
        depth=self.distance()-q[:,2]
        f=self.focal(width,height)
        safe=xp.maximum(depth,1e-9)
        return width/2+f*q[:,0]/safe,height/2-f*q[:,1]/safe,depth


LIGHT=np.array([-.35,.55,.76]); LIGHT/=np.linalg.norm(LIGHT)
HALF=(LIGHT+[0,0,1]); HALF/=np.linalg.norm(HALF)
CLAY=np.array([.80,.77,.72])

KERNELS=r'''
extern "C" __global__ void raster(const float* sx,const float* sy,const float* sz,const int* faces,int count,
  int width,int height,float near,unsigned long long* zbuf){
 int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=count) return;
 int a=faces[3*i],b=faces[3*i+1],c=faces[3*i+2];
 float x0=sx[a],y0=sy[a],z0=sz[a],x1=sx[b],y1=sy[b],z1=sz[b],x2=sx[c],y2=sy[c],z2=sz[c];
 if(z0<=near||z1<=near||z2<=near) return;
 float area=(x1-x0)*(y2-y0)-(x2-x0)*(y1-y0); if(fabsf(area)<1e-12f) return;
 int minx=max(0,(int)floorf(fminf(x0,fminf(x1,x2)))),maxx=min(width-1,(int)ceilf(fmaxf(x0,fmaxf(x1,x2))));
 int miny=max(0,(int)floorf(fminf(y0,fminf(y1,y2)))),maxy=min(height-1,(int)ceilf(fmaxf(y0,fmaxf(y1,y2))));
 if(minx>maxx||miny>maxy) return;
 for(int y=miny;y<=maxy;y++) for(int x=minx;x<=maxx;x++){
  float px=x+.5f,py=y+.5f;
  float w0=((x1-px)*(y2-py)-(x2-px)*(y1-py))/area,w1=((x2-px)*(y0-py)-(x0-px)*(y2-py))/area,w2=1.f-w0-w1;
  if(w0<0.f||w1<0.f||w2<0.f) continue;
  float depth=1.f/(w0/z0+w1/z1+w2/z2);
  unsigned long long key=((unsigned long long)__float_as_uint(depth)<<32)|(unsigned int)i;
  atomicMin(&zbuf[y*width+x],key);
 }
}
extern "C" __global__ void shade(const unsigned long long* zbuf,const float* sx,const float* sy,const float* sz,
  const int* faces,const float* normals,const unsigned char* colors,int use_color,int width,int height,
  const float* light,unsigned char* out){
 int p=blockIdx.x*blockDim.x+threadIdx.x; if(p>=width*height) return;
 int x=p%width,y=p/width; unsigned long long key=zbuf[p];
 if(key==0xffffffffffffffffULL){
  float t=(float)y/height; out[3*p]=(unsigned char)(24-10*t); out[3*p+1]=(unsigned char)(30-12*t); out[3*p+2]=(unsigned char)(44-16*t); return;
 }
 int i=(int)(key&0xffffffffULL); int a=faces[3*i],b=faces[3*i+1],c=faces[3*i+2];
 float x0=sx[a],y0=sy[a],z0=sz[a],x1=sx[b],y1=sy[b],z1=sz[b],x2=sx[c],y2=sy[c],z2=sz[c];
 float area=(x1-x0)*(y2-y0)-(x2-x0)*(y1-y0),px=x+.5f,py=y+.5f;
 float w0=((x1-px)*(y2-py)-(x2-px)*(y1-py))/area,w1=((x2-px)*(y0-py)-(x0-px)*(y2-py))/area,w2=1.f-w0-w1;
 w0/=z0;w1/=z1;w2/=z2; float s=w0+w1+w2; w0/=s;w1/=s;w2/=s;
 float n[3]; for(int k=0;k<3;k++) n[k]=w0*normals[3*a+k]+w1*normals[3*b+k]+w2*normals[3*c+k];
 float len=sqrtf(n[0]*n[0]+n[1]*n[1]+n[2]*n[2])+1e-20f; for(int k=0;k<3;k++) n[k]/=len;
 int back=n[2]<0.f; if(back) for(int k=0;k<3;k++) n[k]=-n[k];
 float diffuse=fmaxf(0.f,n[0]*light[0]+n[1]*light[1]+n[2]*light[2]);
 float spec=powf(fmaxf(0.f,n[0]*light[3]+n[1]*light[4]+n[2]*light[5]),40.f);
 float intensity=.16f+.52f*diffuse+.30f*n[2];
 float base[3]={%CLAY%};
 if(use_color) for(int k=0;k<3;k++) base[k]=(w0*colors[3*a+k]+w1*colors[3*b+k]+w2*colors[3*c+k])/255.f;
 if(back){base[0]*=.70f;base[1]*=.78f;base[2]*=1.f;}
 for(int k=0;k<3;k++){float v=base[k]*intensity+.22f*spec; out[3*p+k]=(unsigned char)fminf(255.f,fmaxf(0.f,v*255.f));}
}
'''.replace('%CLAY%',','.join(f'{v:.3f}f' for v in CLAY))


class GpuRenderer:
    name='RTX (CUDA)'

    def __init__(self):
        from gpu_backend import cuda_module
        self.cp=cp=cuda_module()
        module=cp.RawModule(code=KERNELS)
        self.raster=module.get_function('raster'); self.shade_kernel=module.get_function('shade')
        self.mesh=None

    def upload(self,mesh):
        cp=self.cp
        self.mesh=mesh
        self.vertices=cp.asarray(mesh.vertices,dtype=cp.float32)
        self.faces=cp.asarray(mesh.faces,dtype=cp.int32)
        self.normals=cp.asarray(mesh.normals,dtype=cp.float32)
        self.colors=cp.asarray(mesh.colors if mesh.colors is not None else np.zeros((1,3),np.uint8))

    def render(self,camera,width,height,use_color=True):
        cp=self.cp
        sx,sy,sz=camera.project(self.vertices,width,height,xp=cp)
        R=cp.asarray(camera.rotation(),dtype=cp.float32)
        normals=cp.ascontiguousarray(self.normals@R.T)
        zbuf=cp.full(width*height,np.uint64(0xffffffffffffffff),dtype=cp.uint64)
        near=np.float32(camera.reference[1]*1e-3)
        count=len(self.faces)
        self.raster(((count+127)//128,),(128,),(cp.ascontiguousarray(sx,dtype=cp.float32),cp.ascontiguousarray(sy,dtype=cp.float32),
            cp.ascontiguousarray(sz,dtype=cp.float32),self.faces,np.int32(count),np.int32(width),np.int32(height),near,zbuf))
        out=cp.empty(width*height*3,dtype=cp.uint8)
        light=cp.asarray(np.r_[LIGHT,HALF],dtype=cp.float32)
        self.shade_kernel(((width*height+255)//256,),(256,),(zbuf,cp.ascontiguousarray(sx,dtype=cp.float32),cp.ascontiguousarray(sy,dtype=cp.float32),
            cp.ascontiguousarray(sz,dtype=cp.float32),self.faces,normals,self.colors,
            np.int32(bool(use_color and self.mesh.colors is not None)),np.int32(width),np.int32(height),light,out))
        return out.reshape(height,width,3).get()


class CpuRenderer:
    """Painter's algorithm on a bounded LOD. Still triangles, never points."""
    name='CPU (sin CUDA)'

    def upload(self,mesh):
        if len(mesh.faces)>CPU_BUDGET:
            step=math.ceil(len(mesh.faces)/CPU_BUDGET)
            # Last resort only: the model normally supplies a coarser welded LOD (view_mesh(budget)).
            faces=mesh.faces[::step]
        else: faces=mesh.faces
        self.mesh=mesh; self.faces=faces

    def render(self,camera,width,height,use_color=True):
        from PIL import Image, ImageDraw
        mesh=self.mesh
        sx,sy,sz=camera.project(mesh.vertices,width,height)
        R=camera.rotation(); normals=mesh.normals@R.T
        f=self.faces
        valid=(sz[f]>camera.reference[1]*1e-3).all(1)
        f=f[valid]
        order=np.argsort(-sz[f].mean(1))
        f=f[order]
        n=normals[f].mean(1); n/=np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-20)
        back=n[:,2]<0; n[back]*=-1
        intensity=.16+.52*np.maximum(0,n@LIGHT)+.30*n[:,2]
        spec=np.maximum(0,n@HALF)**40
        if use_color and mesh.colors is not None: base=mesh.colors[f].mean(1)/255
        else: base=np.tile(CLAY,(len(f),1))
        base=base*np.where(back[:,None],[.70,.78,1.],[1.,1.,1.])
        rgb=np.clip((base*intensity[:,None]+.22*spec[:,None])*255,0,255).astype(np.uint8)
        image=Image.new('RGB',(width,height),(20,26,38)); draw=ImageDraw.Draw(image)
        xs=sx[f]; ys=sy[f]
        for k in range(len(f)):
            draw.polygon([(xs[k,0],ys[k,0]),(xs[k,1],ys[k,1]),(xs[k,2],ys[k,2])],fill=tuple(rgb[k]))
        return np.asarray(image)


def make_renderer(prefer_gpu=True):
    """Returns (renderer, note). The note explains a CPU fallback explicitly."""
    if prefer_gpu:
        try: return GpuRenderer(),''
        except Exception as error:
            return CpuRenderer(),f'Visor en CPU: CUDA no disponible para el visor ({error}).'
    return CpuRenderer(),''
