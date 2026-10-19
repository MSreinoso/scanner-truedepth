"""Reproducible from-scratch RGB-D denoising on public NYU; not TrueDepth validation."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from neural_depth import network,MODEL_NAME

ROOT=Path(__file__).resolve().parent

def prepare(path):
    import h5py
    cache=ROOT/'.training-data'/'nyu_near_range.npz'
    if cache.exists(): return dict(np.load(cache))
    depth=[]; colors=[]; masks=[]; scenes=[]
    with h5py.File(path,'r') as data:
        print({key:data[key].shape for key in ('images','depths','rawDepths','scenes')},flush=True)
        for i in range(data['depths'].shape[0]):
            d=np.asarray(data['depths'][i]).T.astype(np.float32)
            raw=np.asarray(data['rawDepths'][i]).T
            rgb=np.asarray(data['images'][i]).transpose(2,1,0)
            d=cv2.resize(d[16:-16,16:-16],(320,240),interpolation=cv2.INTER_NEAREST)
            valid=cv2.resize((np.isfinite(raw)&(raw>0)).astype(np.uint8)[16:-16,16:-16],(320,240),interpolation=cv2.INTER_NEAREST)
            # Scale scenes into near range, explicitly a transfer augmentation, not sensor calibration.
            scale=.45/np.median(d[d>0]); d=np.clip(d*scale,.08,1.5)
            depth.append(d.astype(np.float16)); masks.append(valid)
            colors.append(cv2.resize(rgb[16:-16,16:-16],(320,240),interpolation=cv2.INTER_AREA))
            ref=data['scenes'][0,i]; scenes.append(''.join(chr(int(x)) for x in data[ref][:].ravel()))
    result={'depth':np.stack(depth),'rgb':np.stack(colors),'mask':np.stack(masks),'scene':np.array(scenes)}
    np.savez_compressed(cache,**result)
    return result

def split_scenes(data):
    names=np.unique(data['scene']); rng=np.random.default_rng(2718); rng.shuffle(names)
    n=len(names); groups=(names[:int(n*.7)],names[int(n*.7):int(n*.85)],names[int(n*.85):])
    return [np.flatnonzero(np.isin(data['scene'],group)) for group in groups]

def batch(data,indices,rng,count=16,size=96):
    dep=[]; rgb=[]; masks=[]
    for i in rng.choice(indices,count):
        y=int(rng.integers(0,241-size)); x=int(rng.integers(0,321-size))
        d=data['depth'][i,y:y+size,x:x+size].astype(np.float32)
        c=data['rgb'][i,y:y+size,x:x+size]; m=data['mask'][i,y:y+size,x:x+size].astype(np.float32)
        if rng.random()<.5: d=d[:,::-1].copy(); c=c[:,::-1].copy(); m=m[:,::-1].copy()
        dep.append(d); rgb.append(c); masks.append(m)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    clean=torch.tensor(np.stack(dep)[:,None],device=device)
    color=torch.tensor(np.stack(rgb).transpose(0,3,1,2),device=device,dtype=torch.float32)/255
    mask=torch.tensor(np.stack(masks)[:,None],device=device)
    return clean,color,mask

def corrupt(clean,mask,generator):
    # Randomized synthetic structured-light-like noise. Parameters are not a measured TrueDepth model.
    b,_,h,w=clean.shape
    noise=torch.randn(clean.shape,device=clean.device,generator=generator)
    sigma=(.0004+.0036*torch.rand((b,1,1,1),device=clean.device,generator=generator))
    correlated=F.interpolate(torch.randn((b,1,max(2,h//8),max(2,w//8)),device=clean.device,generator=generator),size=(h,w),mode='bilinear',align_corners=False)
    noisy=clean+noise*sigma+correlated*sigma*.35
    quant=.0001+.0004*torch.rand((b,1,1,1),device=clean.device,generator=generator)
    noisy=torch.round(noisy/quant)*quant
    return noisy*mask

@torch.inference_mode()
def evaluate(model,data,indices,seed,batches=24):
    rng=np.random.default_rng(seed); gen=torch.Generator(device=next(model.parameters()).device).manual_seed(seed)
    totals=np.zeros(5,dtype=np.float64); model.eval()
    for _ in range(batches):
        clean,rgb,mask=batch(data,indices,rng); noisy=corrupt(clean,mask,gen); prediction=model(noisy,rgb,mask)
        total=mask.sum().item()
        totals += [((noisy-clean).square()*mask).sum().item(),((prediction-clean).square()*mask).sum().item(),
                   ((noisy-clean).abs()*mask).sum().item(),((prediction-clean).abs()*mask).sum().item(),total]
    return {'noisy_rmse_mm':float(np.sqrt(totals[0]/totals[4])*1000),'model_rmse_mm':float(np.sqrt(totals[1]/totals[4])*1000),
            'noisy_mae_mm':float(totals[2]/totals[4]*1000),'model_mae_mm':float(totals[3]/totals[4]*1000),'valid_pixels':int(totals[4])}

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--steps',type=int,default=2400)
    parser.add_argument('--data',type=Path,default=ROOT/'.training-data'/'nyu_depth_v2_labeled.mat'); args=parser.parse_args()
    torch.manual_seed(42); np.random.seed(42); torch.set_num_threads(4); torch.set_float32_matmul_precision('high')
    device='cuda' if torch.cuda.is_available() else 'cpu'; print('Training device:',device,flush=True)
    data=prepare(args.data); train,val,test=split_scenes(data)
    print('Scene-separated frame counts:',len(train),len(val),len(test),flush=True)
    model=network().to(device); optimizer=torch.optim.AdamW(model.parameters(),lr=.0005)
    rng=np.random.default_rng(42); generator=torch.Generator(device=device).manual_seed(42)
    folder=ROOT/'models'; folder.mkdir(exist_ok=True); output=folder/MODEL_NAME
    best=float('inf'); best_step=0; history=[]; started=time.time()
    for step in range(1,args.steps+1):
        model.train(); clean,rgb,mask=batch(data,train,rng); noisy=corrupt(clean,mask,generator)
        prediction=model(noisy,rgb,mask)
        loss=((prediction-clean).square()*mask).sum()/mask.sum().clamp_min(1)*1e6
        # Preserve gradients and depth discontinuities rather than optimizing only smooth interiors.
        for axis in (2,3):
            valid=mask.diff(dim=axis).abs()==0
            valid=valid & (mask.narrow(axis,0,mask.shape[axis]-1)>0)
            loss=loss+.15*((prediction.diff(dim=axis)-clean.diff(dim=axis)).abs()*valid).sum()/valid.sum().clamp_min(1)*1000
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        if step%200==0 or step==args.steps:
            metrics=evaluate(model,data,val,701,batches=12)
            print(json.dumps({'step':step,'seconds':round(time.time()-started,1),'loss':loss.item(),**metrics}),flush=True)
            history.append({'step':step,**metrics})
            if metrics['model_rmse_mm']<best:
                best=metrics['model_rmse_mm']; best_step=step; torch.save(model.state_dict(),output)
    model.load_state_dict(torch.load(output,map_location=device,weights_only=True))
    metrics=evaluate(model,data,test,1701,batches=64)
    card={'architecture':'residual-unet-v1','sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
          'training':'from scratch; NYU Depth V2 scaled to near range with synthetic noise',
          'dataset_url':'https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html',
          'dataset_sha256':hashlib.file_digest(args.data.open('rb'),'sha256').hexdigest(),
          'seed':42,'steps':args.steps,'selected_step':best_step,'seconds':round(time.time()-started,1),'device':torch.cuda.get_device_name() if device=='cuda' else device,
          'parameters':sum(p.numel() for p in model.parameters()),'split_by':'scene; 70/15/15 percent; seed 2718',
          'train_frames':len(train),'val_frames':len(val),'test_frames':len(test),
          'splits':{k:np.unique(data['scene'][v]).tolist() for k,v in zip(('train','validation','test'),(train,val,test))},
          'held_out_synthetic_noise_metrics':metrics,'validation_history':history,
          'truedepth_validated':False,'limitations':['Kinect reference depths are not metrology ground truth',
              'Noise is randomized, not calibrated to a specific TrueDepth unit','No measured accuracy guarantee at 0.05 mm',
              'Invalid pixels remain invalid; corrections bounded to 10 mm; opt-in experimental reconstruction only']}
    output.with_suffix('.json').write_text(json.dumps(card,indent=2),encoding='utf-8')
    print('FINAL',json.dumps(metrics),flush=True)
    if metrics['model_rmse_mm']>=metrics['noisy_rmse_mm']: raise RuntimeError('Model failed held-out noisy baseline')

if __name__=='__main__': main()
