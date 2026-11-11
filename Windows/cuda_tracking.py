"""Frame-to-model tracking (KinectFusion style) on the RTX.

A coarse TSDF holds everything accepted so far. Each new frame is aligned
against a depth map ray-cast from that volume at the predicted pose, so the
phone may move in any direction without the drift of frame-to-frame chains.
This volume only serves tracking; the exported mesh is fused afterwards at the
requested resolution from the same original frames.
"""
import math
import numpy as np
from gpu_backend import cuda_module

KERNELS=r'''
extern "C" __global__ void integrate_track(float* tsdf,float* weight,const float* depth,const float* E,const float* K,
  int width,int height,int n,float ox,float oy,float oz,float voxel,float trunc,float maxw){
 long long i=(long long)blockDim.x*blockIdx.x+threadIdx.x; long long total=(long long)n*n*n;
 if(i>=total) return;
 int x=i%n,y=(i/n)%n,z=i/((long long)n*n);
 float wx=ox+x*voxel,wy=oy+y*voxel,wz=oz+z*voxel;
 float px=E[0]*wx+E[1]*wy+E[2]*wz+E[3],py=E[4]*wx+E[5]*wy+E[6]*wz+E[7],pz=E[8]*wx+E[9]*wy+E[10]*wz+E[11];
 if(pz<=0.f) return;
 int u=__float2int_rn(K[0]*px/pz+K[2]),v=__float2int_rn(K[1]*py/pz+K[3]);
 if(u<0||v<0||u>=width||v>=height) return;
 float d=depth[v*width+u]; if(!(d>0.f)) return;
 float sdf=d-pz; if(sdf<-trunc) return;
 float value=fminf(1.f,sdf/trunc),w=weight[i];
 tsdf[i]=(tsdf[i]*w+value)/(w+1.f); weight[i]=fminf(w+1.f,maxw);
}
__device__ bool sample(const float* t,const float* w,int n,float gx,float gy,float gz,float* out){
 if(gx<0.f||gy<0.f||gz<0.f||gx>=n-1||gy>=n-1||gz>=n-1) return false;
 int x=(int)gx,y=(int)gy,z=(int)gz; float fx=gx-x,fy=gy-y,fz=gz-z; float acc=0.f;
 for(int c=0;c<8;c++){
  int dx=c&1,dy=(c>>1)&1,dz=(c>>2)&1;
  long long idx=(long long)(x+dx)+(long long)n*((y+dy)+(long long)n*(z+dz));
  if(w[idx]<=0.f) return false;
  acc+=t[idx]*(dx?fx:1.f-fx)*(dy?fy:1.f-fy)*(dz?fz:1.f-fz);
 }
 *out=acc; return true;
}
extern "C" __global__ void raycast(const float* tsdf,const float* weight,float* out,const float* C,const float* K,
  int width,int height,int n,float ox,float oy,float oz,float voxel,float trunc,float near,float far){
 int p=blockIdx.x*blockDim.x+threadIdx.x; if(p>=width*height) return;
 int u=p%width,v=p/width; out[p]=0.f;
 float dx=(u-K[2])/K[0],dy=(v-K[3])/K[1],dz=1.f;
 float rx=C[0]*dx+C[1]*dy+C[2]*dz,ry=C[4]*dx+C[5]*dy+C[6]*dz,rz=C[8]*dx+C[9]*dy+C[10]*dz;
 float len=sqrtf(rx*rx+ry*ry+rz*rz),step=.5f*trunc/len;
 float prev=0.f; bool has=false;
 for(float t=near;t<=far;t+=step){
  float wx=C[3]+rx*t,wy=C[7]+ry*t,wz=C[11]+rz*t,f;
  if(!sample(tsdf,weight,n,(wx-ox)/voxel,(wy-oy)/voxel,(wz-oz)/voxel,&f)){ has=false; continue; }
  if(has&&prev>0.f&&f<0.f){ out[p]=t-step*f/(f-prev); return; }
  if(has&&prev<0.f&&f>0.f) return;
  prev=f; has=true;
 }
}
'''


class TrackingVolume:
    def __init__(self,first_depth,voxel=.005,side_m=1.6):
        self.cp=cp=cuda_module()
        positive=first_depth[first_depth>0]
        center_z=float(np.median(positive)) if positive.size else .4
        self.n=n=int(round(side_m/voxel)); self.voxel=voxel; self.trunc=3*voxel
        self.origin=np.array([-side_m/2,-side_m/2,center_z-side_m/2],np.float32)
        self.tsdf=cp.ones(n**3,dtype=cp.float32); self.weight=cp.zeros(n**3,dtype=cp.float32)
        module=cp.RawModule(code=KERNELS)
        self.k_integrate=module.get_function('integrate_track'); self.k_raycast=module.get_function('raycast')
        self.frames=0

    def integrate(self,depth,intrinsics,pose):
        cp=self.cp; h,w=depth.shape
        E=cp.asarray(np.linalg.inv(pose)[:3,:].astype(np.float32).ravel()); K=cp.asarray(np.asarray(intrinsics,np.float32))
        total=self.n**3
        self.k_integrate(((total+255)//256,),(256,),(self.tsdf,self.weight,cp.ascontiguousarray(cp.asarray(depth,dtype=cp.float32)),E,K,
            np.int32(w),np.int32(h),np.int32(self.n),*[np.float32(v) for v in self.origin],np.float32(self.voxel),np.float32(self.trunc),np.float32(64)))
        self.frames+=1

    def raycast(self,pose,intrinsics,shape,near,far):
        cp=self.cp; h,w=shape
        out=cp.zeros(h*w,dtype=cp.float32)
        C=cp.asarray(np.asarray(pose)[:3,:].astype(np.float32).ravel()); K=cp.asarray(np.asarray(intrinsics,np.float32))
        self.k_raycast(((h*w+127)//128,),(128,),(self.tsdf,self.weight,out,C,K,np.int32(w),np.int32(h),np.int32(self.n),
            *[np.float32(v) for v in self.origin],np.float32(self.voxel),np.float32(self.trunc),np.float32(near),np.float32(far)))
        return out.reshape(h,w)
