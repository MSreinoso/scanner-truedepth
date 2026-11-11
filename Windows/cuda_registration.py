"""Projective coarse-to-fine RGB-D registration. All dense math runs in CUDA.

Only the final 4x4 pose and scalar quality metrics return to the host.
"""
import numpy as np
from gpu_backend import cuda_module,_kernel


def vertices(depth, intrinsics):
    cp=cuda_module()
    z=cp.asarray(depth,dtype=cp.float32)
    y,x=cp.indices(z.shape,dtype=cp.float32)
    fx,fy,cx,cy=intrinsics
    return cp.stack(((x-cx)*z/fx,(y-cy)*z/fy,z),axis=-1)

ICP_REDUCE=r'''
extern "C" __global__ void icp_reduce(const float* pts,int count,const float* T,const float* target,const float* normals,
  const unsigned char* valid,int w,int h,float fx,float fy,float cx,float cy,float limit2,double* out){
 // One pass: projective correspondence, Huber-like weight and the 6x6 normal
 // equations (21 upper terms + 6 + inlier count), reduced per block. No atomics:
 // the same input always gives bit-identical poses (and identical STL files).
 __shared__ double partial[28][8];
 double acc[28]; for(int k=0;k<28;k++) acc[k]=0.;
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<count){
  float x=pts[3*i],y=pts[3*i+1],z=pts[3*i+2];
  float mx=T[0]*x+T[1]*y+T[2]*z+T[3],my=T[4]*x+T[5]*y+T[6]*z+T[7],mz=T[8]*x+T[9]*y+T[10]*z+T[11];
  if(mz>1e-6f){
   int u=__float2int_rn(fx*mx/mz+cx),v=__float2int_rn(fy*my/mz+cy);
   if(u>=0&&v>=0&&u<w&&v<h&&valid[v*w+u]){
    const float* q=target+3*(v*w+u); const float* n=normals+3*(v*w+u);
    float dx=mx-q[0],dy=my-q[1],dz=mz-q[2];
    if(dx*dx+dy*dy+dz*dz<limit2){
     double r=(double)n[0]*dx+(double)n[1]*dy+(double)n[2]*dz;
     double wgt=fmin(1.,.003/fmax(fabs(r),1e-9));
     double J[6]={(double)my*n[2]-(double)mz*n[1],(double)mz*n[0]-(double)mx*n[2],(double)mx*n[1]-(double)my*n[0],n[0],n[1],n[2]};
     int k=0; for(int a=0;a<6;a++) for(int b=a;b<6;b++) acc[k++]=wgt*J[a]*J[b];
     for(int a=0;a<6;a++) acc[21+a]=wgt*J[a]*r;
     acc[27]=1.;
    }
   }
  }
 }
 int lane=threadIdx.x&31,warp=threadIdx.x>>5;
 for(int k=0;k<28;k++){
  double v=acc[k];
  for(int o=16;o>0;o>>=1) v+=__shfl_down_sync(0xffffffff,v,o);
  if(lane==0) partial[k][warp]=v;
 }
 __syncthreads();
 if(threadIdx.x<28){
  double v=0.; for(int k=0;k<(blockDim.x>>5);k++) v+=partial[threadIdx.x][k];
  out[blockIdx.x*28+threadIdx.x]=v;  // per-block partials: summed in a fixed order (deterministic)
 }
}
'''
UPPER=[(a,b) for a in range(6) for b in range(a,6)]


def _rodrigues(update):
    angle=float(np.linalg.norm(update[:3])); axis=update[:3]/max(angle,1e-12)
    x,y,z=axis; skew=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    delta=np.eye(4); delta[:3,:3]=np.eye(3)+np.sin(angle)*skew+(1-np.cos(angle))*(skew@skew); delta[:3,3]=update[3:]
    return delta


def register(depth, previous_depth, intrinsics, previous_intrinsics, initial, settings):
    """Projective point-to-plane ICP. Correspondences, weights and the 6x6
    normal equations come from one CUDA kernel per iteration; only 28 numbers
    cross to the CPU (one synchronisation), where the 6x6 system is solved in
    float64. Real frames took 0.4-0.7 s per alignment with array-level CuPy."""
    register.last_detail={}
    cp=cuda_module()
    transform=np.asarray(initial,dtype=np.float64).copy()
    target=vertices(previous_depth,previous_intrinsics)
    # Normals require valid neighbours; holes and depth discontinuities do not vote.
    normals=cp.cross(target[1:-1,2:]-target[1:-1,:-2],target[2:,1:-1]-target[:-2,1:-1])
    length=cp.linalg.norm(normals,axis=2)
    normals/=cp.maximum(length[:,:,None],1e-12)
    normals=cp.pad(normals,((1,1),(1,1),(0,0)))
    valid_normals=cp.zeros(previous_depth.shape,dtype=cp.bool_)
    z=target[:,:,2]
    valid_normals[1:-1,1:-1]=(length>1e-12)&(z[1:-1,1:-1]>0)&(z[1:-1,2:]>0)&(z[1:-1,:-2]>0)&(z[2:,1:-1]>0)&(z[:-2,1:-1]>0)&(cp.abs(z[1:-1,2:]-z[1:-1,:-2])<.02)&(cp.abs(z[2:,1:-1]-z[:-2,1:-1])<.02)
    source=vertices(depth,intrinsics)
    fx,fy,cx,cy=previous_intrinsics;h,w=previous_depth.shape

    def correspond(points, limit):
        R=cp.asarray(transform[:3,:3],dtype=cp.float32); t=cp.asarray(transform[:3,3],dtype=cp.float32)
        moved=points@R.T+t
        positive=moved[:,2]>1e-6
        safe=cp.maximum(moved[:,2],1e-6)
        u=cp.rint(fx*moved[:,0]/safe+cx).astype(cp.int32)
        v=cp.rint(fy*moved[:,1]/safe+cy).astype(cp.int32)
        inside=positive&(u>=0)&(u<w)&(v>=0)&(v<h)
        u=cp.clip(u,0,w-1);v=cp.clip(v,0,h-1)
        matched=target[v,u];n=normals[v,u]
        squared=cp.sum((moved-matched)**2,axis=1)
        mask=inside&valid_normals[v,u]&(squared<limit**2)
        return moved,matched,n,squared,mask

    median=float(cp.median(source[:,:,2][source[:,:,2]>0])) if bool(cp.any(source[:,:,2]>0)) else 0.
    spacing=max(1,int(round(settings.registration_mm/1000*intrinsics[0]/median))) if median>0 else 1
    limit=settings.correspondence_mm/1000
    target_c=cp.ascontiguousarray(target,dtype=cp.float32); normals_c=cp.ascontiguousarray(normals,dtype=cp.float32); valid_c=cp.ascontiguousarray(valid_normals).view(cp.uint8)
    for stride in (spacing*4,spacing*2,1):
        points=source[::stride,::stride].reshape(-1,3)
        points=points[points[:,2]>0]
        if len(points)<30:
            if stride>1: continue
            raise ValueError('Pocos puntos válidos para ICP CUDA')
        reduce=_kernel(ICP_REDUCE,'icp_reduce'); points=cp.ascontiguousarray(points,dtype=cp.float32)
        blocks=(len(points)+255)//256
        for _ in range(20):
            out=cp.empty((blocks,28),dtype=cp.float64)
            reduce((blocks,),(256,),(points,np.int32(len(points)),cp.asarray(transform[:3,:].astype(np.float32).ravel()),target_c,normals_c,valid_c,
                np.int32(w),np.int32(h),np.float32(fx),np.float32(fy),np.float32(cx),np.float32(cy),np.float32(limit*limit),out))
            packed=out.sum(axis=0).get()
            if packed[27]<30:
                if stride>1: break
                raise ValueError('Solapamiento insuficiente para ICP CUDA')
            normal=np.zeros((6,6))
            for k,(a,b) in enumerate(UPPER): normal[a,b]=normal[b,a]=packed[k]
            # Damped least squares handles planar objects without arbitrary motion
            # in directions that the observed geometry cannot constrain.
            update=np.linalg.solve(normal+np.eye(6)*1e-5,-packed[21:27])
            transform=_rodrigues(update)@transform
            # 1e-6 rad / 1 µm: far below the finest 0.05 mm voxel.
            if np.linalg.norm(update)<1e-6: break
    points=source.reshape(-1,3);points=points[points[:,2]>0]
    p,q,n,squared,mask=correspond(points,limit)
    p,q,n,squared=p[mask],q[mask],n[mask],squared[mask]
    fitness=len(p)/max(1,len(points))
    # Quality = point-to-plane residual (what ICP minimizes). Point-to-point
    # distance also counts sliding along the surface and grows with the
    # sensor's millimetric noise, which rejected valid real TrueDepth frames.
    if len(p):
        plane=cp.abs(cp.sum(n*(p-q),axis=1))
        rmse=float(cp.sqrt(cp.mean(plane**2)))
        register.last_detail={'point_to_point_rmse_mm':float(cp.sqrt(cp.mean(squared)))*1000,
                              'point_to_plane_median_mm':float(cp.median(plane))*1000}
    else: rmse=float('inf')
    pose=transform
    angle=np.degrees(np.arccos(np.clip((np.trace(pose[:3,:3])-1)/2,-1,1)))
    # TrueDepth noise grows with the square of the distance: the RMSE limit
    # is the configured value at 0.30 m and scales up (at most 3x) beyond it.
    # Real 0.40 m captures were otherwise rejected at 3.0-3.1 mm with 80 % overlap.
    limit_rmse=settings.max_rmse_mm/1000*min(3.,max(1.,(median/.30)**2)) if median>0 else settings.max_rmse_mm/1000
    register.last_detail['rmse_limit_mm']=limit_rmse*1000
    valid=bool(np.isfinite(pose).all() and fitness>=settings.min_fitness and rmse<=limit_rmse and np.linalg.norm(pose[:3,3])<=.1 and angle<=45)
    return valid,pose,fitness,rmse


def unique_tiles(keys):
    cp=cuda_module()
    # Packing three signed 21-bit coordinates avoids CuPy's slow axis-wise
    # comparison path. Reject overflow instead of silently merging tiles.
    limit=1<<20
    if bool(cp.any((keys < -limit)|(keys >= limit))): raise ValueError('Escena fuera del rango de mosaicos CUDA')
    shifted=keys+limit
    packed=(shifted[:,0]<<42)|(shifted[:,1]<<21)|shifted[:,2]
    packed=cp.unique(packed)
    mask=(1<<21)-1
    return cp.stack((packed>>42,(packed>>21)&mask,packed&mask),axis=1)-limit


def surface_tile_keys(depth, intrinsics, pose, width):
    cp=cuda_module()
    points=vertices(depth,intrinsics).reshape(-1,3)
    points=points[points[:,2]>0]
    transformed=points@cp.asarray(pose[:3,:3]).T+cp.asarray(pose[:3,3])
    return unique_tiles(cp.floor(transformed/width).astype(cp.int64))
