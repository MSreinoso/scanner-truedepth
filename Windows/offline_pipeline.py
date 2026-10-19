"""Offline RGB-D sessions: append-only capture, verified upload, deferred fusion."""
from dataclasses import dataclass, asdict
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib
import json
import math
import os
import queue
import re
import shutil
import socketserver
import struct
import threading
import time
import uuid
import zlib
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, filedialog
import urllib.request

HEADER = b'B3DREC02'
RAW_MAGIC = 0xB1C01702
MAX_RECORD = 8 * 1024 * 1024
MAX_SESSION = 32 * 1024**3


@dataclass(frozen=True)
class Settings:
    resolution_mm: float = 0.5
    min_depth_m: float = 0.15
    max_depth_m: float = 0.60
    registration_mm: float = 1.0
    correspondence_mm: float = 15.0
    max_rmse_mm: float = 3.0
    min_fitness: float = 0.45
    geometric_weight: float = 0.968
    memory_mb: int = 1024
    stride: int = 1
    min_observations: float = 2.0
    device: str = 'Auto'
    rgb_guidance: bool = False
    neural_alignment: bool = False
    neural_denoising: bool = False

    model_path: str = ""
    tile_cells: int = 64

    def validate(self):
        for name,value in asdict(self).items():
            if isinstance(value,(int,float)) and not math.isfinite(value):
                raise ValueError(f'{name}: valor no finito')
        if not .01 <= self.resolution_mm <= 5: raise ValueError('Resolución: 0.01–5 mm')
        if not .05 <= self.min_depth_m < self.max_depth_m <= 2: raise ValueError('Rango de profundidad inválido')
        if not .2 <= self.registration_mm <= 10 or not 1 <= self.correspondence_mm <= 100: raise ValueError('Escalas ICP inválidas')
        if not .1 <= self.max_rmse_mm <= self.correspondence_mm: raise ValueError('RMSE debe ser menor que la correspondencia')
        if not .1 <= self.min_fitness <= 1 or not .1 <= self.geometric_weight <= 1: raise ValueError('Solapamiento/peso: 0.1–1')
        if not 128 <= self.memory_mb <= 16384 or not 1 <= self.stride <= 30: raise ValueError('Memoria: 128–16384 MB; paso: 1–30')
        if not 0 <= self.min_observations <= 20 or self.device not in ('Auto','CPU','CUDA'): raise ValueError('Configuración de fusión inválida')
        if self.tile_cells not in (16,32,64,96,128): raise ValueError('Mosaico inválido')
        if any(not isinstance(value,bool) for value in (self.rgb_guidance,self.neural_alignment,self.neural_denoising)): raise ValueError('Opciones RGB/IA inválidas')
        return self


PRESETS = {
    'Escaneo Estándar': Settings(resolution_mm=.5, registration_mm=2., memory_mb=2048,rgb_guidance=True),
    'Sketch / Vista Previa': Settings(resolution_mm=2., registration_mm=3., stride=3, min_observations=1.),
    'Resolución Alta': Settings(resolution_mm=.2, registration_mm=1., memory_mb=2048,rgb_guidance=True),
    'Máxima Resolución 1:1*': Settings(resolution_mm=.05, registration_mm=.5, memory_mb=2048),
}


def exact(source,count):
    data=source.read(count)
    if len(data)!=count: raise ValueError('Sesión truncada')
    return data


@dataclass
class RawFrame:
    frame_id: int
    timestamp: float
    rotation: np.ndarray
    intrinsics: np.ndarray
    depth: np.ndarray
    rgb: np.ndarray
    calibration: dict


def decode_frame(packet):
    if not 100 <= len(packet) <= MAX_RECORD: raise ValueError('Registro inválido')
    magic,frame_id,timestamp=struct.unpack_from('<IId',packet)
    if magic!=RAW_MAGIC or not math.isfinite(timestamp): raise ValueError('Formato de captura incompatible')
    rotation=np.array(struct.unpack_from('<9f',packet,16)).reshape(3,3)
    intrinsics=np.array(struct.unpack_from('<4f',packet,64))
    if not np.isfinite(rotation).all() or not np.isfinite(intrinsics).all() or np.any(intrinsics[:2]<=0): raise ValueError('Calibración inválida')
    if not np.allclose(rotation@rotation.T,np.eye(3),atol=.02) or np.linalg.det(rotation)<.98: raise ValueError('Rotación inválida')
    w,h,size=struct.unpack_from('<HHI',packet,80)
    if not 0<w*h<=2_000_000 or size!=w*h*4 or 88+size+8>len(packet): raise ValueError('Profundidad inválida')
    depth=np.frombuffer(packet,dtype='<f4',count=w*h,offset=88).reshape(h,w).copy()
    offset=88+size
    rw,rh,size=struct.unpack_from('<HHI',packet,offset); offset+=8
    if not 0<rw*rh<=2_000_000 or size!=rw*rh*4 or offset+size+4>len(packet): raise ValueError('RGB inválido')
    bgra=np.frombuffer(packet,np.uint8,count=size,offset=offset).reshape(rh,rw,4)
    rgb=bgra[:,:,[2,1,0]].copy(); offset+=size
    meta_size=struct.unpack_from('<I',packet,offset)[0]
    if meta_size>262144 or offset+4+meta_size!=len(packet): raise ValueError('Metadatos inválidos')
    calibration=json.loads(packet[offset+4:].decode())
    if not isinstance(calibration,dict): raise ValueError('Calibración inválida')
    return RawFrame(frame_id,timestamp,rotation,intrinsics,depth,rgb,calibration)


def records(path,decode=True):
    if str(path).endswith('.b3dstream'):
        from chunk_protocol import stream_records
        return (yield from stream_records(path))
    with open(path,'rb') as file:
        if exact(file,8)!=HEADER: raise ValueError('Cabecera incompatible')
        count,previous_time=0,-math.inf
        while True:
            size=struct.unpack('<I',exact(file,4))[0]
            if size==0:
                length=struct.unpack('<I',exact(file,4))[0]
                if not 0<length<=65536: raise ValueError('Manifiesto inválido')
                meta=json.loads(exact(file,length).decode())
                if not isinstance(meta,dict) or meta.get('version')!=2 or meta.get('frames')!=count or not isinstance(meta.get('dropped'),int) or meta['dropped']<0 or count==0 or file.read(1): raise ValueError('Manifiesto no coincide')
                return meta
            if not 100<=size<=MAX_RECORD: raise ValueError('Frame fuera de límites')
            frame=decode_frame(exact(file,size))
            if frame.frame_id!=count or frame.timestamp<=previous_time: raise ValueError('Secuencia temporal inválida')
            previous_time=frame.timestamp; count+=1
            yield frame if decode else None


def validate_session(path):
    iterator=records(path,decode=False)
    while True:
        try: next(iterator)
        except StopIteration as result: return result.value


def file_digest(path):
    digest=hashlib.sha256()
    with open(path,'rb') as file:
        for chunk in iter(lambda:file.read(262144),b''): digest.update(chunk)
    return digest.hexdigest()


def receive_archive(source,length,expected_digest,destination,max_size=MAX_SESSION):
    if not 0<length<=max_size or not re.fullmatch(r'[a-f0-9]{64}',expected_digest): raise ValueError('Longitud o checksum inválidos')
    destination=Path(destination)
    partial=destination.with_suffix('.'+uuid.uuid4().hex+'.partial')
    decoder=zlib.decompressobj(-15); digest=hashlib.sha256()
    remaining,expanded=length,0
    try:
        with open(partial,'xb') as output:
            while remaining:
                chunk=source.read(min(262144,remaining))
                if not chunk: raise ValueError('Transferencia interrumpida')
                remaining-=len(chunk); digest.update(chunk); pending=chunk
                while pending:
                    data=decoder.decompress(pending,262144); pending=decoder.unconsumed_tail
                    expanded+=len(data)
                    if expanded>max_size: raise ValueError('Sesión excede límite de tamaño')
                    if shutil.disk_usage(destination.parent).free<len(data)+128*1024**2: raise ValueError('Disco de Windows lleno')
                    output.write(data)
                if decoder.unused_data: raise ValueError('Datos después del flujo comprimido')
            if digest.hexdigest()!=expected_digest or not decoder.eof: raise ValueError('Paquete incompleto o checksum incorrecto')
            output.flush(); os.fsync(output.fileno())
        meta=validate_session(partial)
        if destination.exists():
            if file_digest(destination)!=file_digest(partial): raise ValueError('UUID ya usado por otra captura')
            partial.unlink()
        else: os.replace(partial,destination)
        return {'sha256':expected_digest,**meta}
    finally:
        if partial.exists(): partial.unlink()


def relative_rotation(previous,current):
    basis=np.diag([1.,-1.,-1.])
    return basis@previous@current.T@basis


def align_cloud(current,previous,previous_imu,current_imu,settings,initial=None):
    import open3d as o3d
    transform=np.eye(4); transform[:3,:3]=relative_rotation(previous_imu,current_imu)
    if initial is not None: transform=initial
    last=None
    for scale in (4.,2.,1.):
        voxel=settings.registration_mm*scale/1000
        source=current.voxel_down_sample(voxel); target=previous.voxel_down_sample(voxel)
        if min(len(source.points),len(target.points))<30: raise ValueError('Pocos puntos para alinear')
        for cloud in (source,target): cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel*3,max_nn=30))
        distance=max(voxel*2,settings.correspondence_mm/1000/scale)
        last=o3d.pipelines.registration.registration_icp(source,target,distance,transform,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=40))
        transform=last.transformation
        if scale==1 and source.has_colors() and target.has_colors():
            try:
                colored=o3d.pipelines.registration.registration_colored_icp(source,target,distance,transform,
                    o3d.pipelines.registration.TransformationEstimationForColoredICP(settings.geometric_weight),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30))
                if colored.fitness>=last.fitness*.95 and colored.inlier_rmse<=last.inlier_rmse*1.2:
                    last,transform=colored,colored.transformation
            except RuntimeError: pass
    angle=math.degrees(math.acos(float(np.clip((np.trace(transform[:3,:3])-1)/2,-1,1))))
    valid=(np.isfinite(transform).all() and last.fitness>=settings.min_fitness and last.inlier_rmse<=settings.max_rmse_mm/1000
           and np.linalg.norm(transform[:3,3])<=.10 and angle<=45)
    return valid,transform,last.fitness,last.inlier_rmse


def prepare_frame(frame,settings):
    import cv2
    depth=frame.depth.copy(); h,w=depth.shape
    rgb=cv2.resize(frame.rgb,(w,h),interpolation=cv2.INTER_AREA)
    cal=frame.calibration; table=np.asarray(cal.get('lensTable',[]),dtype=np.float32)
    if len(table)>1:
        reference=np.asarray(cal.get('referenceSize'),dtype=float); center=np.asarray(cal.get('distortionCenter'),dtype=float)
        if reference.shape!=(2,) or center.shape!=(2,) or not np.isfinite(table).all() or np.any(reference<=0): raise ValueError('Distorsión inválida')
        center=center/reference*[w,h]
        u,v=np.meshgrid(np.arange(w,dtype=np.float32),np.arange(h,dtype=np.float32))
        radius=np.hypot(u-center[0],v-center[1])
        max_radius=np.hypot(max(center[0],w-center[0]),max(center[1],h-center[1]))
        magnification=np.interp(radius,np.linspace(0,max_radius,len(table)),table)
        mx=(u+(u-center[0])*magnification).astype(np.float32); my=(v+(v-center[1])*magnification).astype(np.float32)
        depth=cv2.remap(depth,mx,my,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)
        rgb=cv2.remap(rgb,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    valid=np.isfinite(depth)&(depth>=settings.min_depth_m)&(depth<=settings.max_depth_m)
    safe=np.where(valid,depth,0).astype(np.float32)
    if settings.neural_denoising:
        from neural_depth import denoise_depth

        safe=denoise_depth(safe,rgb,settings.model_path or None)
        safe=np.where(valid,np.clip(safe,settings.min_depth_m,settings.max_depth_m),0).astype(np.float32)
    if settings.rgb_guidance:
        total=safe.copy(); weight=(safe>0).astype(np.float32)
        color=rgb.astype(np.float32)
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                if not dx and not dy: continue
                neighbor=np.roll(safe,(dy,dx),(0,1)); other=np.roll(color,(dy,dx),(0,1))
                valid_neighbor=(neighbor>0)&(safe>0)
                if dy>0: valid_neighbor[:dy]=False
                if dy<0: valid_neighbor[dy:]=False
                if dx>0: valid_neighbor[:,:dx]=False
                if dx<0: valid_neighbor[:,dx:]=False
                weights=np.exp(-((neighbor-safe)/.003)**2-np.sum((color-other)**2,axis=2)/(30**2))*valid_neighbor
                total+=neighbor*weights;weight+=weights
        safe=np.where(safe>0,total/np.maximum(weight,1e-8),0).astype(np.float32)
    median=cv2.medianBlur(safe,3)
    safe[np.abs(safe-median)>.01]=0
    return safe,np.ascontiguousarray(rgb)


class TensorFusion:
    def __init__(self,settings):
        import open3d as o3d
        self.o3d,self.settings=o3d,settings
        cuda=o3d.core.cuda.is_available()
        if settings.device=='CUDA' and not cuda: raise ValueError('Open3D sin CUDA. Selecciona Auto o CPU.')
        self.device=o3d.core.Device('CUDA:0' if cuda and settings.device!='CPU' else 'CPU:0')
        self.max_blocks=max(128,int(settings.memory_mb*1024**2*.40/(16**3*20)))
        self.grid=o3d.t.geometry.VoxelBlockGrid(attr_names=('tsdf','weight','color'),
            attr_dtypes=(o3d.core.float32,o3d.core.float32,o3d.core.float32),attr_channels=(1,1,3),
            voxel_size=settings.resolution_mm/1000,block_resolution=16,block_count=self.max_blocks,device=self.device)

    def integrate(self,depth,rgb,intrinsics,pose):
        o3d=self.o3d
        depth_image=o3d.t.geometry.Image(o3d.core.Tensor(depth,device=self.device))
        color_image=o3d.t.geometry.Image(o3d.core.Tensor(rgb.astype(np.float32),device=self.device))
        fx,fy,cx,cy=intrinsics
        K=o3d.core.Tensor([[fx,0,cx],[0,fy,cy],[0,0,1]],dtype=o3d.core.float64)
        extrinsic=o3d.core.Tensor(np.linalg.inv(pose),dtype=o3d.core.float64)
        coords=self.grid.compute_unique_block_coordinates(depth_image,K,extrinsic,1.,self.settings.max_depth_m,8.)
        _,present=self.grid.hashmap().find(coords)
        needed=int((~present.cpu().numpy()).sum())
        if self.grid.hashmap().size()+needed>self.max_blocks:
            raise MemoryError('La resolución excede el presupuesto de fusión. Aumenta memoria o cambia resolución/recorte y reconstruye otra vez. Originales conservados.')
        self.grid.integrate(coords,depth_image,color_image,K,extrinsic,1.,self.settings.max_depth_m,8.)

    def mesh(self):
        return self.grid.cpu().extract_triangle_mesh(weight_threshold=self.settings.min_observations).to_legacy()


def reconstruct(path,settings,progress=lambda *args:None,cancel=None):
    settings.validate()
    if settings.device!='CPU':
        try:
            from gpu_backend import cuda_module,reconstruct_tiled
            cuda_module()
        except ImportError:
            if settings.device=='CUDA': raise ValueError('Abre el modo RTX para usar el runtime CUDA instalado.')
        else:
            return reconstruct_tiled(path,settings,progress,cancel)
    import open3d as o3d
    settings.validate(); metadata=validate_session(path); fusion=TensorFusion(settings)
    previous,previous_imu,pose=None,None,np.eye(4)
    previous_images,previous_intrinsics=None,None
    accepted,rejected=0,0
    for index,frame in enumerate(records(path)):
        if cancel is not None and cancel.is_set(): raise InterruptedError('Reconstrucción cancelada; originales conservados')
        if index%settings.stride: continue
        depth,rgb=prepare_frame(frame,settings); fx,fy,cx,cy=frame.intrinsics; h,w=depth.shape
        intrinsic=o3d.camera.PinholeCameraIntrinsic(w,h,fx,fy,cx,cy)
        rgbd=o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(rgb),o3d.geometry.Image(depth),
            depth_scale=1.,depth_trunc=settings.max_depth_m,convert_rgb_to_intensity=False)
        cloud=o3d.geometry.PointCloud.create_from_rgbd_image(rgbd,intrinsic).voxel_down_sample(settings.registration_mm/1000)
        if len(cloud.points)<100: rejected+=1; continue
        valid,delta,fitness,rmse=True,np.eye(4),1.,0.
        if previous is not None:
            initial=None
            if settings.neural_alignment:
                from ai_alignment import neural_seed
                initial=neural_seed((depth,rgb),previous_images,frame.intrinsics,previous_intrinsics)
            try: valid,delta,fitness,rmse=align_cloud(cloud,previous,previous_imu,frame.rotation,settings,initial)
            except ValueError: valid=False
        if valid:
            candidate=pose@delta
            fusion.integrate(depth,rgb,frame.intrinsics,candidate)
            pose,previous,previous_imu=candidate,cloud,frame.rotation
            previous_images,previous_intrinsics=(depth,rgb),frame.intrinsics
            accepted+=1
        else: rejected+=1
        progress(index+1,metadata['frames'],accepted,rejected,str(fusion.device),fitness,rmse)
    if accepted<2: raise ValueError('Menos de dos frames alineados. Revisa captura y parámetros.')
    mesh=fusion.mesh()
    if len(mesh.triangles)==0: raise ValueError('Sin superficie consistente. Revisa captura/observaciones mínimas.')
    mesh.remove_degenerate_triangles(); mesh.remove_duplicated_triangles(); mesh.remove_unreferenced_vertices(); mesh.compute_vertex_normals()
    return mesh,{'settings':asdict(settings),'captured':metadata,'accepted':accepted,'rejected':rejected,
                 'device':str(fusion.device),'vertices':len(mesh.vertices),'triangles':len(mesh.triangles)}


class UploadServer(ThreadingHTTPServer):
    daemon_threads=True
    def __init__(self,address,directory,callback):
        self.directory=Path(directory); self.directory.mkdir(parents=True,exist_ok=True)
        self.callback=callback; self.upload_lock=threading.Lock()
        super().__init__(address,UploadHandler)


class UploadHandler(BaseHTTPRequestHandler):
    def log_message(self,*args): pass

    def respond(self,code,data):
        raw=json.dumps(data).encode()
        self.send_response(code); self.send_header('Content-Type','application/json')
        self.send_header('Content-Length',str(len(raw))); self.end_headers(); self.wfile.write(raw)

    def do_GET(self): self.respond(200,{'service':'Bichito3D Progressive','version':3})

    def do_PUT(self):
        match=re.fullmatch(r'/sessions/([A-Fa-f0-9-]{36})',self.path)
        if not match: self.respond(400,{'error':'ID inválido'}); return
        if not self.server.upload_lock.acquire(blocking=False): self.respond(409,{'error':'Otra transferencia en curso'}); return
        try:
            self.connection.settimeout(120)
            path=self.server.directory/(str(uuid.UUID(match[1]))+'.b3draw')
            receipt=receive_archive(self.rfile,int(self.headers.get('Content-Length','0')),self.headers.get('X-Content-SHA256',''),path)
            self.respond(200,receipt)
            self.server.callback(path,receipt)
        except (ValueError,OSError,zlib.error) as error: self.respond(400,{'error':str(error)})
        finally: self.server.upload_lock.release()


class PairHandler(socketserver.BaseRequestHandler):
    def handle(self):
        self.server.callback(self.client_address[0]); self.request.settimeout(180)
        try:
            while self.request.recv(1024): pass
        except OSError: pass


class PairServer(socketserver.ThreadingTCPServer):
    allow_reuse_address=True
    daemon_threads=True


class StableViewer(tk.Canvas):
    """No timer or sensor rotation; dragging is the only source of angle changes."""
    def __init__(self,parent):
        super().__init__(parent,bg='#080e19',highlightthickness=0)
        self.points=None; self.yaw,self.pitch,self.zoom=0.,0.,1.; self.last=None
        self.bind('<Configure>',lambda event:self.render())
        self.bind('<ButtonPress-1>',lambda event:setattr(self,'last',(event.x,event.y)))
        self.bind('<B1-Motion>',self.drag); self.bind('<MouseWheel>',self.wheel)

    def drag(self,event):
        if self.last:
            self.yaw+=(event.x-self.last[0])*.01
            self.pitch=float(np.clip(self.pitch+(event.y-self.last[1])*.01,-1.5,1.5))
            self.last=(event.x,event.y); self.render()

    def wheel(self,event):
        self.zoom=float(np.clip(self.zoom*(1.1 if event.delta>0 else 1/1.1),.1,10)); self.render()

    def reset(self): self.yaw,self.pitch,self.zoom=0.,0.,1.; self.render()

    def set_mesh(self,mesh):
        vertices=np.asarray(mesh.vertices); step=max(1,len(vertices)//6000)
        self.points=vertices[::step].copy(); colors=np.asarray(mesh.vertex_colors)
        self.colors=colors[::step].copy() if len(colors)==len(vertices) else np.tile([.2,.8,1.],(len(self.points),1))
        self.points-=(vertices.min(axis=0)+vertices.max(axis=0))/2
        self.radius=max(float(np.linalg.norm(self.points,axis=1).max()),1e-6); self.render()

    def render(self):
        self.delete('all')
        w,h=max(1,self.winfo_width()),max(1,self.winfo_height())
        if self.points is None:
            self.create_text(w/2,h/2,text='Visor estable · sin giro automático\nReconstrucción después de recibir la sesión.',fill='#94a3b8',justify='center'); return
        cy,sy,cp,sp=math.cos(self.yaw),math.sin(self.yaw),math.cos(self.pitch),math.sin(self.pitch)
        R=np.array([[cy,0,sy],[sp*sy,cp,-sp*cy],[-cp*sy,sp,cp*cy]])
        view=self.points@R.T; scale=min(w,h)*.42*self.zoom/self.radius
        for i in np.argsort(view[:,2]):
            x,y=view[i,0]*scale+w/2,view[i,1]*scale+h/2
            color='#%02x%02x%02x'%tuple(np.clip(self.colors[i]*255,0,255).astype(int))
            self.create_oval(x-1,y-1,x+1,y+1,fill=color,outline='')


class OfflineStudio:
    def __init__(self,root):

        self.root=root; root.title('Bichito 3D 3.1 · Simple / Avanzado · Captura progresiva'); root.geometry('1380x900')
        self.events=queue.Queue(maxsize=100); self.cancel=threading.Event()
        self.busy,self.mesh,self.session=False,None,None

        self.results={}
        self.phone_active,self.phone_busy=False,False
        self.ip=tk.StringVar(); self.status=tk.StringVar(value='Conecta el iPhone. Durante captura solo se muestra vídeo ligero.')
        self.quality=tk.DoubleVar(value=.5); self.params={}
        self.storage=Path(os.environ.get('LOCALAPPDATA',str(Path.home())))/'Bichito3D'/'Sessions'
        self.storage.mkdir(parents=True,exist_ok=True); self.build_ui()
        try:
            from chunk_protocol import BlockServer
            self.blocks=BlockServer(self.storage,lambda path,receipt:self.post('received',(path,receipt)))
            self.blocks.start()
            self.server=UploadServer(('0.0.0.0',50507),self.storage,lambda path,receipt:self.post('received',(path,receipt)))
            threading.Thread(target=self.server.serve_forever,daemon=True).start()
            self.pair=PairServer(('0.0.0.0',50506),PairHandler); self.pair.callback=lambda ip:self.post('paired',ip)
            threading.Thread(target=self.pair.serve_forever,daemon=True).start()
        except OSError as error: self.status.set(f'Puerto ocupado: {error}. Cierra la versión anterior.')
        self.closed=False; self.preview_stop=threading.Event(); self.preview_ip=''
        threading.Thread(target=self.preview_worker,daemon=True).start()
        root.after(100,self.poll); root.protocol('WM_DELETE_WINDOW',self.close)
        # Keep discovery compatible; it carries only the PC address, never frames.
        from Bichito3D_ReconstructionEngine import UDPServerBeacon
        self.beacon=UDPServerBeacon(); self.beacon.start()
        self.status.set(self.status.get()+f' IP PC: {self.beacon._local_ip} · recepción 50507.')

    def post(self,kind,value):
        try: self.events.put_nowait((kind,value))
        except queue.Full:
            if kind not in ('preview','progress'): self.events.put((kind,value),timeout=10)

    def build_ui(self):
        top=ttk.Frame(self.root,padding=10); top.pack(fill='x')
        ttk.Label(top,text='IP iPhone').pack(side='left'); ttk.Entry(top,textvariable=self.ip,width=18).pack(side='left',padx=6)
        for text,command in [('Conectar',self.connect),('Iniciar/finalizar captura',self.toggle_phone),
            ('Abrir sesión',self.import_session),('Reconstruir / aplicar parámetros',self.start_reconstruction),('Cancelar procesado',self.cancel.set)]:
            ttk.Button(top,text=text,command=command).pack(side='left',padx=3)
        body=ttk.PanedWindow(self.root,orient='horizontal'); body.pack(fill='both',expand=True)
        left=ttk.Frame(body,padding=10); body.add(left,weight=1)
        self.video=tk.Label(left,text='Vista ligera del iPhone',bg='#080e19',fg='white',height=10); self.video.pack(fill='x')
        tabs=ttk.Notebook(left); tabs.pack(fill='both',expand=True,pady=8)
        simple=ttk.Frame(tabs,padding=8); advanced=ttk.Frame(tabs,padding=8)
        tabs.add(simple,text='Simple'); tabs.add(advanced,text='Avanzado')
        panel=advanced; canvas=tk.Canvas(panel,height=360,highlightthickness=0)
        scrollbar=ttk.Scrollbar(panel,orient='vertical',command=canvas.yview)
        scrollbar.pack(side='right',fill='y');canvas.pack(side='left',fill='both',expand=True)
        canvas.configure(yscrollcommand=scrollbar.set)
        advanced=ttk.Frame(canvas);window=canvas.create_window((0,0),window=advanced,anchor='nw')
        advanced.bind('<Configure>',lambda event:canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.bind('<Configure>',lambda event:canvas.itemconfigure(window,width=event.width))
        self.preset_name=tk.StringVar(value='Escaneo Estándar')
        descriptions=['Balance entre velocidad y detalle','Comprobar forma rápidamente','Más detalle en piezas intermedias','0.05 mm solicitados; procesamiento intensivo']
        for name,description in zip(PRESETS,descriptions):
            ttk.Radiobutton(simple,text=name,variable=self.preset_name,value=name,command=self.apply_preset).pack(anchor='w',pady=(10,2))
            ttk.Label(simple,text=description,wraplength=320).pack(anchor='w',padx=18)
        ttk.Label(simple,text='* 1:1 describe el muestreo de salida, no una exactitud física de 0.05 mm. Los originales no se alteran.',wraplength=330).pack(anchor='w',pady=18)
        ttk.Label(advanced,text='Resolución de reconstrucción (mm)').pack(anchor='w',pady=(8,0))
        self.res_label=ttk.Label(advanced,text='0.050 mm'); self.res_label.pack(anchor='w')
        ttk.Scale(advanced,from_=.01,to=2.,variable=self.quality,command=lambda value:self.res_label.config(text=f'{float(value):.3f} mm')).pack(fill='x')
        defaults=PRESETS['Escaneo Estándar']
        fields=[('resolution_mm','Resolución exacta (mm)'),('min_depth_m','Distancia mínima (m)'),('max_depth_m','Distancia máxima (m)'),
            ('registration_mm','Muestreo ICP (mm)'),('correspondence_mm','Correspondencia ICP (mm)'),('max_rmse_mm','Error máximo ICP (mm)'),
            ('min_fitness','Solapamiento mínimo (0–1)'),('geometric_weight','Peso geometría (resto color)'),('memory_mb','Presupuesto fusión (MB)'),
            ('stride','Usar cada N frames'),('min_observations','Observaciones mínimas')]
        for key,label in fields:
            row=ttk.Frame(advanced); row.pack(fill='x',pady=1); ttk.Label(row,text=label).pack(side='left')
            variable=self.quality if key=='resolution_mm' else tk.StringVar(value=str(getattr(defaults,key)))
            self.params[key]=variable; ttk.Entry(row,textvariable=variable,width=10).pack(side='right')
        self.device=tk.StringVar(value='Auto')
        ttk.Combobox(advanced,textvariable=self.device,values=['Auto','CPU','CUDA'],state='readonly',width=12).pack(anchor='e',pady=4)
        ttk.Label(advanced,text='La resolución no equivale a precisión del sensor.\nCUDA por mosaicos limita VRAM sin aumentar el voxel.',wraplength=390).pack(anchor='w',pady=8)
        self.rgb_guidance=tk.BooleanVar(value=False)
        self.neural_alignment=tk.BooleanVar(value=False)
        self.neural_denoising=tk.BooleanVar(value=False)
        ttk.Checkbutton(advanced,text='Limpieza guiada por RGB (modifica solo reconstrucción)',variable=self.rgb_guidance).pack(anchor='w')
        ttk.Checkbutton(advanced,text='IA LoFTR para alineación (pesos locales)',variable=self.neural_alignment).pack(anchor='w')
        ttk.Checkbutton(advanced,text='IA profundidad experimental (NYU; requiere modo RTX / IA)',variable=self.neural_denoising).pack(anchor='w')

        from model_registry import models,default

        self.model_paths=models(); selected=default()

        if selected not in self.model_paths:self.model_paths.append(selected)

        self.model_choice=tk.StringVar(value=selected.stem)

        self.model_combo=ttk.Combobox(advanced,textvariable=self.model_choice,values=[p.stem for p in self.model_paths],state='readonly')

        self.model_combo.pack(fill='x')

        ttk.Button(advanced,text='Añadir modelo compatible (.pt + .json)',command=self.add_model).pack(fill='x')

        ttk.Button(advanced,text='Establecer modelo predeterminado',command=self.default_model).pack(fill='x')

        self.reconstruction_mode=tk.StringVar(value='RTX sola')

        ttk.Combobox(left,textvariable=self.reconstruction_mode,values=['RTX sola','RTX + IA','Comparar ambas'],state='readonly').pack(fill='x')

        self.result_choice=tk.StringVar()

        self.result_combo=ttk.Combobox(left,textvariable=self.result_choice,state='readonly')

        self.result_combo.pack(fill='x'); self.result_combo.bind('<<ComboboxSelected>>',self.select_result)
        self.apply_preset()
        ttk.Button(left,text='Exportar STL (mm)',command=lambda:self.export('stl')).pack(fill='x')
        ttk.Button(left,text='Exportar PLY color (mm)',command=lambda:self.export('ply')).pack(fill='x',pady=3)
        right=ttk.Frame(body,padding=10); body.add(right,weight=3)
        ttk.Label(right,text='Visor fijo · arrastrar: giro manual · rueda: zoom').pack()
        ttk.Button(right,text='Restablecer vista',command=lambda:self.viewer.reset()).pack(anchor='e')
        self.viewer=StableViewer(right); self.viewer.pack(fill='both',expand=True)
        self.progress=ttk.Progressbar(self.root,maximum=100); self.progress.pack(fill='x',padx=10)
        ttk.Label(self.root,textvariable=self.status,wraplength=1300,padding=10).pack(fill='x')

    def apply_preset(self):
        preset=PRESETS[self.preset_name.get()]
        for key,var in self.params.items(): var.set(getattr(preset,key))
        self.device.set(preset.device)
        self.rgb_guidance.set(preset.rgb_guidance)
        self.neural_alignment.set(preset.neural_alignment)
        self.neural_denoising.set(preset.neural_denoising)
        self.res_label.config(text=f'{preset.resolution_mm:.3f} mm')


    def selected_model(self):

        return next(p for p in self.model_paths if p.stem==self.model_choice.get())


    def add_model(self):

        path=filedialog.askopenfilename(filetypes=[('Checkpoint compatible','*.pt')])

        if not path:return

        try:

            from model_registry import install

            target=install(path);self.model_paths.append(target)

            self.model_combo['values']=[p.stem for p in self.model_paths];self.model_choice.set(target.stem)

            self.status.set('Modelo validado y añadido. Usa Establecer predeterminado para conservar la selección.')

        except Exception as error:self.status.set(str(error))


    def default_model(self):

        try:

            from model_registry import set_default

            set_default(self.selected_model());self.status.set('Modelo predeterminado guardado.')

        except Exception as error:self.status.set(str(error))


    def select_result(self,event=None):

        if self.busy:return

        value=self.results.get(self.result_choice.get())

        if value:

            self.mesh,report=value;self.viewer.set_mesh(self.mesh)

            self.status.set(f'Resultado seleccionado: {self.result_choice.get()} · {report["triangles"]:,} triángulos. Exportar usa este resultado.')

    def read_settings(self):
        data={key:float(var.get()) for key,var in self.params.items()}
        for name in ('memory_mb','stride'):
            if data[name]!=int(data[name]): raise ValueError(f'{name} debe ser entero')
            data[name]=int(data[name])

        data['model_path']=str(self.selected_model()); data['device']=self.device.get(); data['rgb_guidance']=self.rgb_guidance.get(); data['neural_alignment']=self.neural_alignment.get(); data['neural_denoising']=self.neural_denoising.get(); return Settings(**data).validate()

    def phone_request(self,path):
        ip=self.ip.get().strip()
        def worker():
            try:
                with urllib.request.urlopen(f'http://{ip}:8080/{path}',timeout=5) as response: response.read()
                self.post('info','Solicitud enviada al iPhone. Finaliza para comprimir y transferir la captura.')
            except Exception as error: self.post('error',str(error))
        threading.Thread(target=worker,daemon=True).start()

    def connect(self): self.phone_request('connect')
    def toggle_phone(self):
        if self.busy:
            self.status.set('Cancela o finaliza la reconstrucción antes de iniciar otra captura.'); return
        self.phone_request('toggle')

    def preview_worker(self):
        last_status=0.
        while not self.preview_stop.wait(.2):
            ip=self.preview_ip
            if not ip: continue
            if time.monotonic()-last_status>=1:
                last_status=time.monotonic()
                try:
                    with urllib.request.urlopen(f'http://{ip}:8080/status',timeout=1) as response:
                        status=json.loads(response.read(65536))
                    self.post('phone_status',(bool(status.get('isScanning')),bool(status.get('busy'))))
                except Exception: pass
            try:
                with urllib.request.urlopen(f'http://{ip}:8080/frame.jpg',timeout=1) as response:
                    import io
                    image=Image.open(io.BytesIO(response.read(2*1024*1024))).copy(); image.thumbnail((340,220))
                    self.post('preview',image)
            except Exception: pass

    def poll(self):
        if self.closed: return
        self.preview_ip=self.ip.get().strip()
        for _ in range(40):
            try: kind,value=self.events.get_nowait()
            except queue.Empty: break
            if kind=='preview':
                photo=ImageTk.PhotoImage(value); self.video.configure(image=photo,text='',height=0); self.video.image=photo
            elif kind=='paired': self.ip.set(value)
            elif kind=='phone_status':
                self.phone_active,self.phone_busy=value
                if self.phone_active and self.busy:
                    self.cancel.set()
                    self.status.set('Captura iniciada en el iPhone: cancelando reconstrucción para priorizar la captura.')
            elif kind=='received':

                self.results={};self.result_combo['values']=[];self.result_choice.set('');self.mesh=None
                self.session,receipt=value
                self.status.set(f'Sesión verificada: {receipt["frames"]} frames, {receipt["dropped"]} omitidos en captura. Pulsa Reconstruir.')
            elif kind=='progress':
                current,total,accepted,rejected,device,fitness,rmse=value; self.progress['value']=current/total*100
                self.status.set(f'Reconstruyendo {current}/{total} · {accepted} aceptados · {rejected} rechazados · {device} · RMSE {rmse*1000:.2f} mm')

            elif kind=='comparison':

                self.busy=False;self.results=value;self.result_combo['values']=list(value)

                self.result_choice.set(next(iter(value)));self.select_result()
            elif kind=='result':
                self.busy=False; self.mesh,report=value; self.viewer.set_mesh(self.mesh)
                self.status.set(f'Modelo listo: {report["vertices"]:,} vértices, {report["triangles"]:,} triángulos · {report["device"]}')
            elif kind=='reconstruction_error': self.busy=False; self.status.set(str(value))
            elif kind=='error': self.status.set(str(value))
            else: self.status.set(str(value))
        self.root.after(100,self.poll)

    def import_session(self):
        path=filedialog.askopenfilename(initialdir=self.storage,filetypes=[('Sesión Bichito','*.b3draw *.b3dz *.b3dstream')])
        if not path: return
        def worker():
            try:
                source=Path(path)
                if source.suffix=='.b3dz':
                    target=self.storage/(source.stem+'.b3draw')
                    with open(source,'rb') as file: receipt=receive_archive(file,source.stat().st_size,file_digest(source),target)
                else: target=source; receipt=validate_session(source)
                self.post('received',(target,receipt))
            except Exception as error: self.post('error',str(error))
        threading.Thread(target=worker,daemon=True).start()

    def start_reconstruction(self):
        if self.busy: return
        if self.phone_active or self.phone_busy or (hasattr(self,'server') and self.server.upload_lock.locked()):
            self.status.set('Espera a que finalice la captura, compresión y transferencia.'); return
        if self.session is None: self.status.set('Primero finaliza y transfiere una captura, o abre una sesión.'); return
        try: settings=self.read_settings()
        except (ValueError,tk.TclError) as error: self.status.set(str(error)); return
        if settings.neural_denoising or settings.neural_alignment:
            import importlib.util
            if importlib.util.find_spec('torch') is None:
                self.status.set('La IA necesita el runtime local. Pulsa Abrir modo RTX / IA y activa allí la opción.'); return

        self.reconstruction_mode_value=self.reconstruction_mode.get()
        self.results={};self.mesh=None;self.result_combo["values"]=[];self.result_choice.set("")
        self.busy=True; self.cancel.clear(); path=self.session; self.status.set('Preparando reconstrucción diferida…')
        def worker():
            try:

                from dataclasses import replace

                mode=self.reconstruction_mode_value

                variants=['RTX sola','RTX + IA'] if mode=='Comparar ambas' else [mode]

                results={}

                for variant in variants:

                    config=replace(settings,device='CUDA',neural_denoising=variant=='RTX + IA',neural_alignment=settings.neural_alignment if variant=='RTX + IA' else False)

                    mesh,report=reconstruct(path,config,lambda *args:self.post('progress',args),self.cancel)

                    Path(path).with_suffix('.'+('rtx-ai' if variant=='RTX + IA' else 'rtx')+'.json').write_text(json.dumps(report,indent=2),encoding='utf-8')

                    results[variant]=(mesh,report)

                self.post('comparison',results)
            except Exception as error: self.post('reconstruction_error',str(error))
        threading.Thread(target=worker,daemon=True).start()

    def export(self,extension):
        if self.mesh is None: return
        path=filedialog.asksaveasfilename(defaultextension='.'+extension,filetypes=[(extension.upper(),'*.'+extension)])
        if not path: return
        if hasattr(self.mesh,'export_mm'):
            try: self.mesh.export_mm(path); self.status.set(f'Exportado en milímetros: {path}')
            except Exception as error: self.status.set(str(error))
            return
        import copy,open3d as o3d
        mesh=copy.deepcopy(self.mesh); mesh.scale(1000,center=(0,0,0))
        if not o3d.io.write_triangle_mesh(path,mesh,write_ascii=False): self.status.set('No se pudo exportar')
        else: self.status.set(f'Exportado en milímetros: {path}')

    def close(self):
        self.closed=True; self.cancel.set(); self.preview_stop.set()
        if hasattr(self,'beacon'): self.beacon.stop()
        if hasattr(self,'blocks'): threading.Thread(target=self.blocks.close,daemon=True).start()
        if hasattr(self,'server'): self.server.server_close()
        if hasattr(self,'pair'): self.pair.server_close()
        self.root.destroy()


def main():
    root=tk.Tk(); OfflineStudio(root); root.mainloop()
