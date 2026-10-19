"""Experimental metric-depth residual denoiser. Original sensor data is never changed."""
from pathlib import Path
import hashlib
import json
import numpy as np

MODEL_NAME='depth_denoiser_v1.pt'

def network():
    import torch
    from torch import nn
    import torch.nn.functional as F
    class Block(nn.Sequential):
        def __init__(self,a,b):
            super().__init__(nn.Conv2d(a,b,3,padding=1),nn.SiLU(),nn.Conv2d(b,b,3,padding=1),nn.SiLU())
    class ResidualUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc1=Block(6,16); self.enc2=Block(16,32); self.mid=Block(32,64)
            self.dec2=Block(96,32); self.dec1=Block(48,16); self.head=nn.Conv2d(16,1,1)
            nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
        def forward(self,depth,rgb,mask):
            # Mask-normalized local mean exposes millimetric noise without losing metric depth.
            mean=F.avg_pool2d(depth*mask,5,1,2)/F.avg_pool2d(mask,5,1,2).clamp_min(.04)
            inputs=torch.cat((depth,(depth-mean).clamp(-.03,.03)/.01,rgb,mask),1)
            a=self.enc1(inputs); b=self.enc2(F.avg_pool2d(a,2)); c=self.mid(F.avg_pool2d(b,2))
            b=self.dec2(torch.cat((F.interpolate(c,size=b.shape[-2:],mode='bilinear',align_corners=False),b),1))
            a=self.dec1(torch.cat((F.interpolate(b,size=a.shape[-2:],mode='bilinear',align_corners=False),a),1))
            return (depth+.01*torch.tanh(self.head(a)))*mask
    return ResidualUNet()

class DepthDenoiser:
    def __init__(self,path=None):
        import torch
        path=Path(path) if path else Path(__file__).resolve().parent/'models'/MODEL_NAME
        card=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
        if hashlib.sha256(path.read_bytes()).hexdigest()!=card['sha256']:
            raise ValueError('Pesos de limpieza neuronal alterados')
        if card.get('architecture')!='residual-unet-v1': raise ValueError('Arquitectura neuronal desconocida')
        self.device='cuda' if torch.cuda.is_available() else 'cpu'; self.torch=torch
        self.model=network().to(self.device).eval()
        self.model.load_state_dict(torch.load(path,map_location=self.device,weights_only=True))

    def denoise(self,depth,rgb):
        torch=self.torch; source=np.asarray(depth,dtype=np.float32)
        valid=np.isfinite(source)&(source>0); safe=np.where(valid,source,0)
        if not valid.any(): return safe
        tensor=torch.from_numpy(safe.copy())[None,None].to(self.device)
        color=torch.from_numpy(np.ascontiguousarray(rgb.transpose(2,0,1))).float()[None].to(self.device)/255
        mask=torch.from_numpy(valid.astype(np.float32))[None,None].to(self.device)
        with torch.inference_mode(): output=self.model(tensor,color,mask)[0,0].cpu().numpy()
        # No synthesized holes or unconstrained changes; retain metric scale and invalid pixels.
        return np.where(valid,np.clip(output,safe-.01,safe+.01),0).astype(np.float32)

_denoiser=None
def denoise_depth(depth,rgb):
    global _denoiser
    if _denoiser is None: _denoiser=DepthDenoiser()
    return _denoiser.denoise(depth,rgb)
