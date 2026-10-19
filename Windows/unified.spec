from PyInstaller.utils.hooks import collect_all,copy_metadata
from PyInstaller.archive.writers import CArchiveWriter
# Standard zlib stream, faster rebuilds than the packager's level-9 default.
CArchiveWriter._COMPRESSION_LEVEL=6
from pathlib import Path
root=Path(SPECPATH)
datas=[(str(root/'models'),'models'),(str(root/'.models'),'.models')]
binaries=[];hidden=[]
for distribution in ('nvidia-cuda-runtime-cu12','nvidia-cuda-nvrtc-cu12'):
    datas+=copy_metadata(distribution)
for package in ('torch','cupy','cupy_backends','cupyx','kornia','kornia_rs','open3d','skimage','nvidia.cuda_runtime','nvidia.cuda_nvrtc'):
    d,b,h=collect_all(package);datas+=d;binaries+=b;hidden+=h
skip=('.lib','.h','.hpp','.cpp','.c','.pyi')
datas=[entry for entry in datas if not entry[0].lower().endswith('.lib')]
binaries=[entry for entry in binaries if not entry[0].lower().endswith(skip)]
a=Analysis([str(root/'Bichito3D_ReconstructionEngine.py')],pathex=[str(root)],binaries=binaries,datas=datas,
           hiddenimports=hidden,excludes=['tensorflow','jax','IPython','matplotlib.tests','pytest','__main__'])
a.datas=[entry for entry in a.datas if not entry[0].lower().endswith('.lib')]
# Cython imports fastrlock dynamically; include its Python package and native extension.
import fastrlock
import importlib.metadata
lock_root=Path(fastrlock.__file__).parent
lock_dist=importlib.metadata.distribution('fastrlock')
for file in lock_dist.files:
    if 'LICENSE' in str(file).upper():
        a.datas.append((str(Path('licenses')/'fastrlock'/Path(file).name),str(lock_dist.locate_file(file)),'DATA'))
for file in lock_root.rglob('*'):
    if file.is_file() and '__pycache__' not in file.parts:
        a.datas.append((str(Path('fastrlock')/file.relative_to(lock_root)),str(file),'DATA'))
pyz=PYZ(a.pure)
exe=EXE(pyz,a.scripts,a.binaries,a.datas,[],name='Bichito3D-Unificado-v3.2',debug=False,
        strip=False,upx=False,console=False)
