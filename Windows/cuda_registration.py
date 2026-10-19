"""Projective coarse-to-fine RGB-D registration. All dense math runs in CUDA.

Only the final 4x4 pose and scalar quality metrics return to the host.
"""
import numpy as np
from gpu_backend import cuda_module


def vertices(depth, intrinsics):
    cp=cuda_module()
    z=cp.asarray(depth,dtype=cp.float32)
    y,x=cp.indices(z.shape,dtype=cp.float32)
    fx,fy,cx,cy=intrinsics
    return cp.stack(((x-cx)*z/fx,(y-cy)*z/fy,z),axis=-1)


def register(depth, previous_depth, intrinsics, previous_intrinsics, initial, settings):
    cp=cuda_module()
    transform=cp.asarray(initial,dtype=cp.float64)
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
        moved=points@transform[:3,:3].T+transform[:3,3]
        positive=moved[:,2]>1e-6
        safe=cp.maximum(moved[:,2],1e-6)
        u=cp.rint(fx*moved[:,0]/safe+cx).astype(cp.int32)
        v=cp.rint(fy*moved[:,1]/safe+cy).astype(cp.int32)
        inside=positive&(u>=0)&(u<w)&(v>=0)&(v<h)
        u=cp.clip(u,0,w-1);v=cp.clip(v,0,h-1)
        matched=target[v,u];n=normals[v,u]
        squared=cp.sum((moved-matched)**2,axis=1)
        mask=inside&valid_normals[v,u]&(squared<limit**2)
        return moved[mask],matched[mask],n[mask],squared[mask]

    spacing=max(1,int(round(settings.registration_mm/1000*intrinsics[0]/float(cp.median(source[:,:,2][source[:,:,2]>0]))))) if bool(cp.any(source[:,:,2]>0)) else 1
    for stride in (spacing*4,spacing*2,1):
        points=source[::stride,::stride].reshape(-1,3)
        points=points[points[:,2]>0]
        if len(points)<30:
            if stride>1: continue
            raise ValueError('Pocos puntos válidos para ICP CUDA')
        for _ in range(20):
            p,q,n,_=correspond(points,settings.correspondence_mm/1000)
            if len(p)<30:
                if stride>1: break
                raise ValueError('Solapamiento insuficiente para ICP CUDA')
            residual=cp.sum(n*(p-q),axis=1)
            weight=cp.minimum(1.,.003/cp.maximum(cp.abs(residual),1e-9))
            jacobian=cp.concatenate((cp.cross(p,n),n),axis=1).astype(cp.float64)
            normal=jacobian.T@(weight[:,None]*jacobian)
            # Damped least squares handles planar objects without arbitrary motion
            # in directions that the observed geometry cannot constrain.
            update=cp.linalg.solve(normal+cp.eye(6)*1e-5,-jacobian.T@(weight*residual))
            angle=cp.linalg.norm(update[:3]);axis=update[:3]/cp.maximum(angle,1e-12)
            x,y,z=axis
            skew=cp.stack((cp.stack((x*0,-z,y)),cp.stack((z,x*0,-x)),cp.stack((-y,x,x*0))))
            rotation=cp.eye(3)+cp.sin(angle)*skew+(1-cp.cos(angle))*(skew@skew)
            delta=cp.eye(4);delta[:3,:3]=rotation;delta[:3,3]=update[3:]
            transform=delta@transform
            if float(cp.linalg.norm(update))<1e-7: break
    points=source.reshape(-1,3);points=points[points[:,2]>0]
    p,q,n,squared=correspond(points,settings.correspondence_mm/1000)
    fitness=len(p)/max(1,len(points));rmse=float(cp.sqrt(cp.mean(squared))) if len(p) else float('inf')
    pose=cp.asnumpy(transform)
    angle=np.degrees(np.arccos(np.clip((np.trace(pose[:3,:3])-1)/2,-1,1)))
    valid=bool(np.isfinite(pose).all() and fitness>=settings.min_fitness and rmse<=settings.max_rmse_mm/1000 and np.linalg.norm(pose[:3,3])<=.1 and angle<=45)
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
