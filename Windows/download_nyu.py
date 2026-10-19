"""Download the original NYU labeled RGB-D data, with resumable bounded I/O."""
from pathlib import Path
import urllib.request
import time

URL='https://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat'
EXPECTED_SIZE=2972037809

def download():
    folder=Path(__file__).resolve().parent/'.training-data'; folder.mkdir(exist_ok=True)
    path=folder/'nyu_depth_v2_labeled.mat'
    if path.exists() and path.stat().st_size==EXPECTED_SIZE: return path
    partial=path.with_suffix('.part'); start=partial.stat().st_size if partial.exists() else 0
    request=urllib.request.Request(URL,headers={'Range':f'bytes={start}-','User-Agent':'Bichito3D research training'})
    with urllib.request.urlopen(request,timeout=120) as response:
        if start and response.status!=206: start=0
        count=start; last=time.monotonic()
        with partial.open('ab' if start else 'wb') as output:
            while data:=response.read(4*1024*1024):
                output.write(data); count+=len(data)
                if time.monotonic()-last>20:
                    print(f'NYU {count/1e6:.0f}/{EXPECTED_SIZE/1e6:.0f} MB',flush=True); last=time.monotonic()
    if count!=EXPECTED_SIZE: raise ValueError(f'Incomplete NYU download: {count}')
    partial.replace(path)
    return path

if __name__=='__main__': print(download(),flush=True)
