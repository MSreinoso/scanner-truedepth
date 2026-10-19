"""Persistent, validated selection of compatible neural depth checkpoints."""
from pathlib import Path
import os,json,hashlib,shutil,uuid

BUILTIN=Path(__file__).resolve().parent/'models'/'depth_denoiser_v1.pt'
def directory():
    path=Path(os.environ.get('LOCALAPPDATA',str(Path.home())))/'Bichito3D'/'Models'
    path.mkdir(parents=True,exist_ok=True);return path
def validate(path):
    path=Path(path);card=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    if card.get('architecture')!='residual-unet-v1':raise ValueError('Modelo incompatible: se requiere residual-unet-v1 y su JSON.')
    if hashlib.sha256(path.read_bytes()).hexdigest()!=card.get('sha256'):raise ValueError('Checksum del modelo inválido')
    return path
def models():return [BUILTIN]+sorted(directory().glob('*.pt'))
def default():
    try:return validate(json.loads((directory()/'default.json').read_text())['path'])
    except (OSError,ValueError,KeyError):return BUILTIN
def set_default(path):
    path=validate(path);tmp=directory()/'default.tmp'
    tmp.write_text(json.dumps({'path':str(path)}));tmp.replace(directory()/'default.json')
def install(path):
    path=validate(path)
    from neural_depth import DepthDenoiser
    DepthDenoiser(path) # Validate tensors/architecture using safe weights-only loading before adding.
    target=directory()/(path.stem+'-'+uuid.uuid4().hex[:8]+'.pt')
    shutil.copyfile(path,target);shutil.copyfile(path.with_suffix('.json'),target.with_suffix('.json'))
    return target
