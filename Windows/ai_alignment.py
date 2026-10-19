"""Optional pretrained LoFTR pose seed. Never synthesizes or replaces measured depth."""
from pathlib import Path
import hashlib
import json
import numpy as np

WEIGHTS_URL='https://cmp.felk.cvut.cz/~mishkdmy/models/loftr_outdoor.ckpt'
WEIGHTS_SHA256='21f5bec5968178e8bc8b7633441836fe5de4f47d861dd2cd7dc38e271b0479ec'


def rigid_fit(source,target):
    a=source.mean(0); b=target.mean(0)
    U,_,V=np.linalg.svd((source-a).T@(target-b))
    R=V.T@U.T
    if np.linalg.det(R)<0: V[-1]*=-1; R=V.T@U.T
    T=np.eye(4);T[:3,:3]=R;T[:3,3]=b-R@a
    return T


def robust_pose(source,target,threshold=.004):
    if len(source)<8: return None
    rng=np.random.default_rng(0); best=np.empty(0,dtype=int)
    for _ in range(128):
        indices=rng.choice(len(source),3,replace=False)
        if np.linalg.matrix_rank(source[indices]-source[indices].mean(0))<2: continue
        transform=rigid_fit(source[indices],target[indices])
        error=np.linalg.norm(source@transform[:3,:3].T+transform[:3,3]-target,axis=1)
        inliers=np.flatnonzero(error<threshold)
        if len(inliers)>len(best): best=inliers
    if len(best)<8: return None
    return rigid_fit(source[best],target[best])


class NeuralMatcher:
    def __init__(self):
        import torch
        from kornia.feature import LoFTR
        directory=Path(__file__).resolve().parent/'.models'; directory.mkdir(exist_ok=True)
        path=directory/'loftr_outdoor.ckpt'; checksum=directory/'loftr_outdoor.sha256'
        if not path.exists() or not checksum.exists():
            raise ValueError('Faltan pesos LoFTR verificados. Ejecuta Preparar_IA.bat una vez.')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=WEIGHTS_SHA256: raise ValueError('Pesos LoFTR alterados')
        self.torch=torch; self.device='cuda' if torch.cuda.is_available() else 'cpu'
        self.model=LoFTR(pretrained=None)
        weights=torch.load(path,map_location='cpu',weights_only=True)
        self.model.load_state_dict(weights['state_dict']); self.model=self.model.eval().to(self.device)
        torch.set_float32_matmul_precision('high')

    def seed(self,current,previous,intrinsics,previous_intrinsics):
        import cv2
        torch=self.torch
        depth,rgb=current; old_depth,old_rgb=previous
        h,w=depth.shape; oh,ow=old_depth.shape
        def tensor(image):
            gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
            gray=cv2.resize(gray,(320,240),interpolation=cv2.INTER_AREA)
            return torch.from_numpy(gray.copy()).float()[None,None].to(self.device)/255
        with torch.inference_mode():
            matches=self.model({'image0':tensor(rgb),'image1':tensor(old_rgb)})
        points=matches['keypoints0'].cpu().numpy()*[w/320,h/240]
        old_points=matches['keypoints1'].cpu().numpy()*[ow/320,oh/240]
        confidence=matches['confidence'].cpu().numpy()
        def unproject(pixel,z,K):
            fx,fy,cx,cy=K
            return np.column_stack(((pixel[:,0]-cx)*z/fx,(pixel[:,1]-cy)*z/fy,z))
        ix=np.clip(np.rint(points[:,0]).astype(int),0,w-1); iy=np.clip(np.rint(points[:,1]).astype(int),0,h-1)
        ox=np.clip(np.rint(old_points[:,0]).astype(int),0,ow-1); oy=np.clip(np.rint(old_points[:,1]).astype(int),0,oh-1)
        z=depth[iy,ix]; oldz=old_depth[oy,ox]
        mask=np.isfinite(z)&np.isfinite(oldz)&(z>0)&(oldz>0)&(confidence>.6)
        return robust_pose(unproject(points[mask],z[mask],intrinsics),unproject(old_points[mask],oldz[mask],previous_intrinsics))


_matcher=None
def neural_seed(current,previous,intrinsics,previous_intrinsics):
    global _matcher
    if _matcher is None: _matcher=NeuralMatcher()
    return _matcher.seed(current,previous,intrinsics,previous_intrinsics)


def download_weights():
    import urllib.request
    directory=Path(__file__).resolve().parent/'.models'; directory.mkdir(exist_ok=True)
    path=directory/'loftr_outdoor.ckpt'; partial=path.with_suffix('.partial')
    with urllib.request.urlopen(WEIGHTS_URL,timeout=120) as response,open(partial,'wb') as file:
        while True:
            data=response.read(262144)
            if not data: break
            file.write(data)
    if not 40_000_000<partial.stat().st_size<100_000_000: raise ValueError('Tamaño de pesos inesperado')
    digest=hashlib.sha256(partial.read_bytes()).hexdigest()
    if digest!=WEIGHTS_SHA256: raise ValueError('Los pesos no coinciden con la versión validada')
    partial.replace(path); (directory/'loftr_outdoor.sha256').write_text(digest)
    (directory/'SOURCE.json').write_text(json.dumps({'url':WEIGHTS_URL,'sha256':digest,'purpose':'LoFTR alignment only'},indent=2))
    print('Pesos oficiales LoFTR descargados y registrados:',digest)


if __name__=='__main__': download_weights()
