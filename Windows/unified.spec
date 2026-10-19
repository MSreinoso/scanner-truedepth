from PyInstaller.utils.hooks import collect_all
from pathlib import Path
root=Path(SPECPATH)
datas=[(str(root/'models'),'models'),(str(root/'.models'),'.models')]
binaries=[];hidden=[]
for package in ('torch','cupy','kornia','kornia_rs','open3d','skimage','nvidia.cuda_runtime','nvidia.cuda_nvrtc'):
    d,b,h=collect_all(package);datas+=d;binaries+=b;hidden+=h
skip=('.lib','.h','.hpp','.cpp','.c','.pyi')
datas=[entry for entry in datas if not entry[0].lower().endswith(('.lib','.pyi'))]
binaries=[entry for entry in binaries if not entry[0].lower().endswith(skip)]
a=Analysis([str(root/'Bichito3D_ReconstructionEngine.py')],pathex=[str(root)],binaries=binaries,datas=datas,
           hiddenimports=hidden,excludes=['tensorflow','jax','IPython','matplotlib.tests','pytest'])
a.datas=[entry for entry in a.datas if not entry[0].lower().endswith(('.lib','.pyi'))]
pyz=PYZ(a.pure)
exe=EXE(pyz,a.scripts,a.binaries,a.datas,[],name='Bichito3D-Unificado-v3.1',debug=False,
        strip=False,upx=False,console=False)
