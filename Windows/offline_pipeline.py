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

from capture_format import (HEADER, RAW_MAGIC, MAX_RECORD, MAX_SESSION, exact, RawFrame, decode_frame,
    records, validate_session, file_digest, receive_archive)


@dataclass(frozen=True)
class Settings:
    resolution_mm: float = 0.05
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

        safe=denoise_depth(safe,rgb,settings.model_path or None,require_cuda=settings.device=='CUDA')
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


class MeshViewer(ttk.Frame):
    """Solid shaded triangles. The camera changes only through user input."""
    def __init__(self,parent):
        super().__init__(parent)
        from mesh_view import Camera
        self.camera=Camera(); self.renderer=None; self.renderer_note=''; self.view=None
        self.photo=None; self.scheduled=False; self.last=None; self.mode=None
        bar=ttk.Frame(self); bar.pack(fill='x',pady=(0,4))
        self.info=tk.StringVar(value='Sin malla. Reconstruye una sesión para inspeccionarla aquí antes de exportar.')
        self.info_label=ttk.Label(bar,textvariable=self.info,wraplength=420); self.info_label.pack(side='left',fill='x',expand=True)
        self.use_color=tk.BooleanVar(value=False)
        ttk.Checkbutton(bar,text='Color RGB',variable=self.use_color,command=self.request).pack(side='right',padx=(6,0))
        ttk.Button(bar,text='Restablecer vista',command=self.reset).pack(side='right')
        self.canvas=tk.Canvas(self,bg='#141a26',highlightthickness=0,width=200,height=160); self.canvas.pack(fill='both',expand=True)
        ttk.Label(self,text='Arrastrar: girar · Clic derecho o Shift+arrastrar: desplazar · Rueda: zoom · Doble clic: restablecer',
                  foreground='#64748b').pack(fill='x')
        c=self.canvas
        c.bind('<Configure>',lambda event:self.request())
        c.bind('<ButtonPress-1>',lambda event:self.press(event,'orbit'))
        c.bind('<Shift-ButtonPress-1>',lambda event:self.press(event,'pan'))
        for button in (2,3): c.bind(f'<ButtonPress-{button}>',lambda event:self.press(event,'pan'))
        for sequence in ('<B1-Motion>','<B2-Motion>','<B3-Motion>'): c.bind(sequence,self.drag)
        c.bind('<MouseWheel>',lambda event:self.zoom(1.15 if event.delta>0 else 1/1.15))
        c.bind('<Button-4>',lambda event:self.zoom(1.15)); c.bind('<Button-5>',lambda event:self.zoom(1/1.15))
        c.bind('<Double-Button-1>',lambda event:self.reset())
        bar.bind('<Configure>',lambda event:self.info_label.configure(wraplength=max(150,event.width-260)))

    def press(self,event,mode): self.last=(event.x,event.y); self.mode=mode

    def drag(self,event):
        if self.view is None or self.last is None: return
        dx,dy=event.x-self.last[0],event.y-self.last[1]; self.last=(event.x,event.y)
        if self.mode=='pan': self.camera.pan_pixels(dx,dy,*self.size())
        else: self.camera.orbit(dx,dy)
        self.request()

    def zoom(self,factor):
        if self.view is None: return
        self.camera.zoom_by(factor); self.request()

    def reset(self):
        self.camera.reset(); self.request()

    def size(self):
        return max(64,min(1920,self.canvas.winfo_width())),max(64,min(1200,self.canvas.winfo_height()))

    def set_mesh(self,model,keep_camera=False):
        from mesh_view import view_mesh_for,make_renderer,CPU_BUDGET
        if self.renderer is None: self.renderer,self.renderer_note=make_renderer(True)
        if getattr(self.renderer,'name','').startswith('CPU') and hasattr(model,'view_mesh'): view=model.view_mesh(CPU_BUDGET)
        else: view=view_mesh_for(model)
        self.view=view; self.renderer.upload(view)
        self.camera.frame(view,keep=keep_camera)
        self.info.set(f'Superficie sólida · {len(view.faces):,} triángulos mostrados ({view.source}) de {view.full_triangles:,} exportables · visor {self.renderer.name}. {self.renderer_note}')
        self.request()

    def clear(self,text='Sin malla.'):
        self.view=None; self.camera.reference=None; self.info.set(text); self.canvas.delete('all')

    def request(self):
        if not self.scheduled:
            self.scheduled=True; self.after_idle(self.render)

    def render(self):
        self.scheduled=False
        if self.view is None: return
        width,height=self.size()
        try: pixels=self.renderer.render(self.camera,width,height,self.use_color.get())
        except Exception as error:
            self.info.set(f'Error del visor: {error}'); return
        self.photo=ImageTk.PhotoImage(Image.fromarray(pixels))
        self.canvas.delete('all'); self.canvas.create_image(0,0,image=self.photo,anchor='nw')


PHASES={'preparing':'Preparando cámara','idle':'Listo para capturar','starting':'Iniciando captura','waitingSensor':'Esperando al sensor TrueDepth',
        'capturing':'Capturando','pausedCache':'Pausa: caché llena (red lenta)','finishing':'Cerrando captura',
        'blockedCache':'Bloqueado: caché ocupada por sesiones antiguas','error':'Error'}
STEPS=['1 Conectar','2 Capturar','3 Verificar transferencia','4 Reconstruir / comparar','5 Inspeccionar malla','6 Exportar']


class OfflineStudio:
    def __init__(self,root):
        self.root=root; root.title('Bichito 3D 3.3 · Escaneo de objetos · RTX'); root.geometry('1380x900')
        self.events=queue.Queue(maxsize=200); self.cancel=threading.Event()
        self.busy,self.mesh,self.session=False,None,None
        self.results={}
        self.phone_active,self.phone_busy=False,False
        self.ip=tk.StringVar(); self.status=tk.StringVar(value='Conecta el iPhone. Durante la captura solo se muestra vídeo ligero.')
        self.quality=tk.DoubleVar(value=.05); self.params={}
        self.storage=Path(os.environ.get('LOCALAPPDATA',str(Path.home())))/'Bichito3D'/'Sessions'
        self.storage.mkdir(parents=True,exist_ok=True); self.build_ui()
        try:
            from chunk_protocol import BlockServer
            self.blocks=BlockServer(self.storage,lambda path,receipt:self.post('received',(path,receipt)),
                on_block=lambda session,seq:self.post('block',(session,seq)),on_error=lambda session,error:self.post('transfer_error',(session,error)))
            self.blocks.start()
            self.server=UploadServer(('0.0.0.0',50507),self.storage,lambda path,receipt:self.post('received',(path,receipt)))
            threading.Thread(target=self.server.serve_forever,daemon=True).start()
            self.pair=PairServer(('0.0.0.0',50506),PairHandler); self.pair.callback=lambda ip:self.post('paired',ip)
            threading.Thread(target=self.pair.serve_forever,daemon=True).start()
        except OSError as error: self.status.set(f'Puerto ocupado ({error}). Cierra otras versiones de Bichito3D y vuelve a abrir esta.')
        self.closed=False; self.preview_stop=threading.Event(); self.preview_ip=''
        self.last_phone_ok=0.; self.blocks_received=0
        threading.Thread(target=self.preview_worker,daemon=True).start()
        root.after(100,self.poll); root.protocol('WM_DELETE_WINDOW',self.close)
        # Discovery carries only the PC address, never frames.
        from Bichito3D_ReconstructionEngine import UDPServerBeacon
        self.beacon=UDPServerBeacon(); self.beacon.start()
        self.pc_info.set(f'IP de este PC: {self.beacon._local_ip} · descubrimiento UDP 50505 · emparejado 50506 · bloques 50508. '
                         'El iPhone aparece solo si ambos están en la misma red; si no, escribe aquí la IP que muestra el iPhone.')

    def post(self,kind,value):
        try: self.events.put_nowait((kind,value))
        except queue.Full:
            if kind not in ('preview','progress','phone_status','block'): self.events.put((kind,value),timeout=10)

    # ---------- layout ----------
    def build_ui(self):
        root=self.root; root.minsize(680,520)
        self.wrap_labels=[]
        top=ttk.Frame(root,padding=(10,8,10,0)); top.pack(fill='x')
        self.step_frame=ttk.Frame(top); self.step_frame.pack(fill='x')
        self.step_labels=[tk.Label(self.step_frame,text=text,padx=6,pady=3,bg='#1e293b',fg='#94a3b8',font=('Segoe UI',9,'bold')) for text in STEPS]
        self.step_state=[0]*len(STEPS)
        row=ttk.Frame(top); row.pack(fill='x',pady=(8,0)); row.columnconfigure(1,weight=1)
        ttk.Label(row,text='IP del iPhone').grid(row=0,column=0,sticky='w',padx=(0,8))
        ttk.Entry(row,textvariable=self.ip).grid(row=0,column=1,sticky='ew')
        ttk.Button(row,text='Conectar',command=self.connect).grid(row=0,column=2,padx=(8,0))
        self.pc_info=tk.StringVar(value='')
        self.label(top,self.pc_info,muted=True)
        actions=ttk.Frame(top); actions.pack(fill='x',pady=(6,0))
        self.capture_button=ttk.Button(actions,text='Iniciar captura',command=self.toggle_phone)
        self.stop_button=ttk.Button(actions,text='Finalizar captura',command=lambda:self.phone_request('stop'),state='disabled')
        self.reconstruct_button=ttk.Button(actions,text='Reconstruir',command=self.start_reconstruction)
        self.cancel_button=ttk.Button(actions,text='Cancelar procesado',command=self.cancel_processing,state='disabled')
        buttons=[self.capture_button,self.stop_button,ttk.Button(actions,text='Abrir sesión',command=self.import_session),
                 self.reconstruct_button,self.cancel_button]
        body=ttk.Frame(root,padding=(10,6,10,0)); body.pack(fill='both',expand=True)
        body.rowconfigure(0,weight=1); body.columnconfigure(1,weight=1)
        controls=ttk.Frame(body,width=360); controls.grid(row=0,column=0,sticky='nsew',padx=(0,10)); controls.grid_propagate(False)
        tabs=ttk.Notebook(controls); tabs.pack(fill='both',expand=True)
        def page(title):
            panel=ttk.Frame(tabs); tabs.add(panel,text=title)
            canvas=tk.Canvas(panel,highlightthickness=0,width=200)
            scrollbar=ttk.Scrollbar(panel,orient='vertical',command=canvas.yview)
            scrollbar.pack(side='right',fill='y'); canvas.pack(side='left',fill='both',expand=True)
            canvas.configure(yscrollcommand=scrollbar.set)
            frame=ttk.Frame(canvas,padding=10); window=canvas.create_window((0,0),window=frame,anchor='nw')
            frame.bind('<Configure>',lambda event:canvas.configure(scrollregion=canvas.bbox('all')))
            canvas.bind('<Configure>',lambda event:canvas.itemconfigure(window,width=event.width))
            def wheel(event): canvas.yview_scroll(-int(event.delta/120),'units')
            for widget in (canvas,frame): widget.bind('<MouseWheel>',wheel)
            return frame
        capture=page('Captura'); quality=page('Reconstrucción'); report=page('Informe')
        # Capture page: state of the phone, transfer and session.
        self.video=tk.Label(capture,text='Vista ligera del iPhone',bg='#080e19',fg='white',height=10); self.video.pack(fill='x')
        self.phone_phase=tk.StringVar(value='iPhone: sin estado (conecta primero)')
        self.phone_message=tk.StringVar(value='')
        self.transfer_message=tk.StringVar(value='Transferencia: sin bloques recibidos todavía.')
        self.session_message=tk.StringVar(value='Sesión: ninguna verificada.')
        self.phone_phase_label=self.label(capture,self.phone_phase,bold=True)
        self.label(capture,self.phone_message)
        self.label(capture,self.transfer_message)
        self.label(capture,self.session_message,bold=True)
        # Reconstruction page: Simple / Advanced.
        self.ui_mode=tk.StringVar(value='Simple')
        modes=ttk.Frame(quality); modes.pack(fill='x',pady=(0,6))
        ttk.Radiobutton(modes,text='Modo Simple',variable=self.ui_mode,value='Simple',command=self.show_mode).pack(side='left')
        ttk.Radiobutton(modes,text='Modo Avanzado',variable=self.ui_mode,value='Avanzado',command=self.show_mode).pack(side='left',padx=(12,0))
        self.label(quality,'Calidad (objetos generales)',bold=True)
        self.preset_name=tk.StringVar(value='Máxima Resolución 1:1*')
        for name in PRESETS:
            ttk.Radiobutton(quality,text=f'{name.rstrip("*")} · {PRESETS[name].resolution_mm:g} mm',variable=self.preset_name,value=name,command=self.apply_preset).pack(anchor='w',pady=3)
        self.label(quality,'El valor en mm es el tamaño de vóxel (muestreo) de la reconstrucción. La precisión física la limita el sensor TrueDepth y la captura; no se garantiza 0,05 mm.',muted=True)
        self.label(quality,'Procesado',bold=True)
        self.reconstruction_mode=tk.StringVar(value='RTX sola')
        ttk.Combobox(quality,textvariable=self.reconstruction_mode,values=['RTX sola','RTX + IA','Comparar ambas'],state='readonly',width=1).pack(fill='x',pady=4)
        self.label(quality,'RTX sola: sin red neuronal. RTX + IA: limpieza neuronal de profundidad con el modelo seleccionado (y LoFTR solo si lo activas). Comparar ambas: dos resultados independientes de la misma captura.',muted=True)
        self.advanced=ttk.Frame(quality)
        adv=self.advanced
        self.label(adv,'Resolución exacta (mm)',bold=True)
        self.res_label=self.label(adv,'0.050 mm')
        ttk.Scale(adv,from_=.05,to=2.,variable=self.quality,command=lambda value:self.res_label.config(text=f'{float(value):.3f} mm')).pack(fill='x')
        fields=[('resolution_mm','Resolución exacta (mm, 0,01–5)'),('min_depth_m','Distancia mínima de captura (m)'),('max_depth_m','Distancia máxima de captura (m)'),
            ('registration_mm','Muestreo ICP (mm)'),('correspondence_mm','Distancia de correspondencia ICP (mm)'),('max_rmse_mm','Error máximo aceptado ICP (mm)'),
            ('min_fitness','Solapamiento mínimo (0–1)'),('stride','Usar cada N fotogramas'),('min_observations','Observaciones mínimas por vóxel'),
            ('memory_mb','Memoria para caché de imágenes en GPU (MB)')]
        defaults=PRESETS['Máxima Resolución 1:1*']
        for key,text in fields:
            self.label(adv,text)
            variable=self.quality if key=='resolution_mm' else tk.StringVar(value=str(getattr(defaults,key)))
            self.params[key]=variable; ttk.Entry(adv,textvariable=variable,width=1).pack(fill='x')
        self.label(adv,'Tamaño de mosaico TSDF (vóxeles por lado; controla VRAM, no cambia la resolución)')
        self.tile_cells=tk.StringVar(value='64')
        ttk.Combobox(adv,textvariable=self.tile_cells,values=['32','64','96','128'],state='readonly',width=1).pack(fill='x',pady=2)
        self.device=tk.StringVar(value='CUDA')
        self.rgb_guidance=tk.BooleanVar(value=False);self.neural_alignment=tk.BooleanVar(value=False);self.neural_denoising=tk.BooleanVar(value=False)
        self.label(adv,'Limpieza y alineación',bold=True)
        ttk.Checkbutton(adv,text='Limpieza guiada por RGB (filtro bilateral clásico, no IA)',variable=self.rgb_guidance).pack(anchor='w',pady=2)
        ttk.Checkbutton(adv,text='Alineación LoFTR (red de correspondencias RGB, solo en RTX + IA)',variable=self.neural_alignment).pack(anchor='w',pady=2)
        self.label(adv,'Gestor de modelos de limpieza neuronal de profundidad',bold=True)
        from model_registry import models,default
        self.model_paths=models(); selected=default()
        if selected not in self.model_paths: self.model_paths.append(selected)
        self.model_choice=tk.StringVar(value=selected.stem)
        self.model_combo=ttk.Combobox(adv,textvariable=self.model_choice,values=[p.stem for p in self.model_paths],state='readonly',width=1)
        self.model_combo.pack(fill='x',pady=4); self.model_combo.bind('<<ComboboxSelected>>',lambda event:self.describe_model())
        self.model_info=tk.StringVar(value=''); self.label(adv,self.model_info,muted=True)
        ttk.Button(adv,text='Añadir modelo compatible (.pt + .json)',command=self.add_model).pack(fill='x',pady=2)
        ttk.Button(adv,text='Establecer como predeterminado',command=self.default_model).pack(fill='x',pady=2)
        # Report page.
        self.report_text=tk.StringVar(value='Sin reconstrucción todavía.')
        self.label(report,self.report_text)
        # Right side: result selection, viewer, export.
        right=ttk.Frame(body); right.grid(row=0,column=1,sticky='nsew')
        head=ttk.Frame(right); head.pack(fill='x',pady=(0,4)); head.columnconfigure(1,weight=1)
        ttk.Label(head,text='Resultado').grid(row=0,column=0,sticky='w',padx=(0,6))
        self.result_choice=tk.StringVar()
        self.result_combo=ttk.Combobox(head,textvariable=self.result_choice,state='readonly',width=1)
        self.result_combo.grid(row=0,column=1,sticky='ew'); self.result_combo.bind('<<ComboboxSelected>>',self.select_result)
        self.swap_button=ttk.Button(head,text='⇄ Alternar',command=self.swap_result,state='disabled'); self.swap_button.grid(row=0,column=2,padx=(6,0))
        self.export_stl=ttk.Button(head,text='Exportar STL (mm)',command=lambda:self.export('stl'),state='disabled')
        self.export_ply=ttk.Button(head,text='Exportar PLY color (mm)',command=lambda:self.export('ply'),state='disabled')
        self.viewer=MeshViewer(right); self.viewer.pack(fill='both',expand=True)
        self.progress=ttk.Progressbar(root,maximum=100); self.progress.pack(fill='x',padx=10,pady=(6,0))
        status_label=ttk.Label(root,textvariable=self.status,wraplength=900,padding=(10,6)); status_label.pack(fill='x')
        self._layout_width=None
        def responsive(event):
            if event.widget is not self.root: return
            width=event.width
            if width==self._layout_width: return
            self._layout_width=width
            columns=5 if width>=1050 else 3
            for index in range(5): actions.columnconfigure(index,weight=1 if index<columns else 0,uniform='a' if index<columns else '')
            for index,button in enumerate(buttons): button.grid(row=index//columns,column=index%columns,sticky='ew',padx=2,pady=2)
            per_row=6 if width>=1150 else 3
            for index in range(6): self.step_frame.columnconfigure(index,weight=1 if index<per_row else 0,uniform='s' if index<per_row else '')
            for index,item in enumerate(self.step_labels): item.grid(row=index//per_row,column=index%per_row,sticky='ew',padx=1,pady=1)
            narrow=width<1000
            self.export_stl.grid(row=1 if narrow else 0,column=0 if narrow else 3,columnspan=2 if narrow else 1,sticky='ew',padx=(0 if narrow else 6,0),pady=(4 if narrow else 0,0))
            self.export_ply.grid(row=1 if narrow else 0,column=2 if narrow else 4,columnspan=2 if narrow else 1,sticky='ew',padx=(6,0),pady=(4 if narrow else 0,0))
            controls.configure(width=280 if width<900 else 360)
            status_label.configure(wraplength=max(300,width-30))
            self.root.after_idle(self.rewrap)
        self.root.bind('<Configure>',responsive,add='+')
        self.root.update_idletasks()
        responsive(type('Layout',(),{'widget':self.root,'width':self.root.winfo_width()})())
        self.apply_preset(); self.show_mode(); self.describe_model(); self.update_steps()

    def label(self,parent,text,bold=False,muted=False):
        options={'wraplength':300}
        if bold: options['font']=('Segoe UI',9,'bold')
        if muted: options['foreground']='#64748b'
        item=ttk.Label(parent,textvariable=text,**options) if isinstance(text,tk.Variable) else ttk.Label(parent,text=text,**options)
        item.pack(fill='x',pady=3); self.wrap_labels.append(item); return item

    def rewrap(self):
        for item in self.wrap_labels:
            try: item.configure(wraplength=max(150,item.winfo_width()-8))
            except tk.TclError: pass

    def show_mode(self):
        if self.ui_mode.get()=='Avanzado': self.advanced.pack(fill='x',pady=(8,0))
        else: self.advanced.pack_forget()

    def update_steps(self):
        """0 pending, 1 active, 2 done."""
        get=lambda name,default=None:getattr(self,name,default)
        connected=bool(self.ip.get().strip()) and time.monotonic()-get('last_phone_ok',0)<4
        session=get('session')
        states=[2 if connected else 1,
                1 if get('phone_active') else (2 if session is not None else 0),
                2 if session is not None else (1 if get('blocks_received',0) else 0),
                1 if get('busy') else (2 if get('results') else 0),
                2 if get('mesh') is not None else 0,
                0]
        colors={0:('#1e293b','#94a3b8'),1:('#0369a1','#ffffff'),2:('#166534','#dcfce7')}
        for item,state in zip(self.step_labels,states):
            bg,fg=colors[state]; item.configure(bg=bg,fg=fg)

    def apply_preset(self):
        preset=PRESETS[self.preset_name.get()]
        for key,var in self.params.items(): var.set(getattr(preset,key))
        self.device.set('CUDA')
        self.rgb_guidance.set(preset.rgb_guidance)
        self.neural_alignment.set(preset.neural_alignment)
        self.neural_denoising.set(preset.neural_denoising)
        self.tile_cells.set(str(preset.tile_cells))
        self.res_label.config(text=f'{preset.resolution_mm:.3f} mm')

    # ---------- models ----------
    def selected_model(self):
        return next(p for p in self.model_paths if p.stem==self.model_choice.get())

    def describe_model(self):
        try:
            card=json.loads(self.selected_model().with_suffix('.json').read_text(encoding='utf-8'))
            metrics=card.get('held_out_synthetic_noise_metrics',{})
            text=f"Arquitectura {card.get('architecture')} · {card.get('parameters','?')} parámetros · entrenamiento: {card.get('training','?')}."
            if metrics: text+=f" Prueba sintética apartada: RMSE {metrics.get('noisy_rmse_mm',0):.2f} → {metrics.get('model_rmse_mm',0):.2f} mm."
            text+=' Validado con TrueDepth real: '+('sí' if card.get('truedepth_validated') else 'NO (ruido sintético; experimental).')
            self.model_info.set(text)
        except Exception as error: self.model_info.set(f'Modelo no legible: {error}')

    def add_model(self):
        path=filedialog.askopenfilename(filetypes=[('Checkpoint compatible','*.pt')])
        if not path: return
        try:
            from model_registry import install
            target=install(path); self.model_paths.append(target)
            self.model_combo['values']=[p.stem for p in self.model_paths]; self.model_choice.set(target.stem); self.describe_model()
            self.status.set('Modelo verificado (arquitectura, SHA-256 y carga segura weights_only) y añadido. Usa Establecer como predeterminado para conservarlo.')
        except Exception as error: self.status.set(f'Modelo rechazado: {error}')

    def default_model(self):
        try:
            from model_registry import set_default
            set_default(self.selected_model()); self.status.set(f'Modelo predeterminado guardado: {self.selected_model().stem}.')
        except Exception as error: self.status.set(str(error))

    # ---------- results ----------
    def select_result(self,event=None,keep_camera=True):
        if self.busy: return
        value=self.results.get(self.result_choice.get())
        if not value: return
        self.mesh,report=value
        try: self.viewer.set_mesh(self.mesh,keep_camera=keep_camera)
        except Exception as error: self.status.set(f'No se pudo mostrar la malla: {error}')
        else: self.status.set(f'Resultado seleccionado: {self.result_choice.get()} · {report["triangles"]:,} triángulos. Exportar usa exactamente este resultado.')
        self.show_report(self.result_choice.get(),report)
        for button in (self.export_stl,self.export_ply): button.configure(state='normal')
        self.update_steps()

    def swap_result(self):
        names=list(self.results)
        if len(names)<2: return
        current=self.result_choice.get()
        self.result_choice.set(names[(names.index(current)+1)%len(names)] if current in names else names[0])
        self.select_result(keep_camera=True)

    def show_report(self,name,report):
        lines=[f'Resultado: {name}',f"GPU: {report.get('gpu_name','—')} · {report.get('device','')}",
               f"Fotogramas: {report.get('accepted',0)} aceptados, {report.get('rejected',0)} rechazados, {report.get('keyframes','—')} keyframes",
               f"Triángulos exportables: {report.get('triangles',0):,} · mosaicos: {report.get('tiles','—')} · pico de memoria CUDA: {report.get('vram_pool_peak_mb','—')} MB",
               f"Vóxel: {report.get('settings',{}).get('resolution_mm','?')} mm (muestreo, no precisión física)",'Etapas y dispositivo:']
        lines+=[f"  • {stage['etapa']}: {stage['dispositivo']}" for stage in report.get('stages',[])]
        if report.get('folder'): lines.append(f"Carpeta: {report['folder']}")
        self.report_text.set('\n'.join(lines))

    def read_settings(self):
        data={key:float(var.get()) for key,var in self.params.items()}
        for name in ('memory_mb','stride'):
            if data[name]!=int(data[name]): raise ValueError(f'{name} debe ser entero')
            data[name]=int(data[name])
        data['model_path']=str(self.selected_model()); data['device']=self.device.get(); data['rgb_guidance']=self.rgb_guidance.get()
        data['neural_alignment']=self.neural_alignment.get(); data['neural_denoising']=self.neural_denoising.get(); data['tile_cells']=int(self.tile_cells.get())
        return Settings(**data).validate()

    # ---------- phone ----------
    def phone_request(self,path):
        ip=self.ip.get().strip()
        if not ip: self.status.set('Escribe la IP que muestra el iPhone o espera al descubrimiento automático.'); return
        labels={'start':'inicio','stop':'finalización','connect':'conexión'}
        def worker():
            try:
                with urllib.request.urlopen(f'http://{ip}:8080/{path}',timeout=5) as response: response.read()
                self.post('info',f'Orden de {labels.get(path,path)} entregada al iPhone ({ip}). El estado real llega desde el iPhone.')
            except Exception as error: self.post('error',f'Desconexión: el iPhone {ip}:8080 no respondió a la orden de {labels.get(path,path)} ({error}).')
        threading.Thread(target=worker,daemon=True).start()

    def connect(self): self.phone_request('connect')
    def toggle_phone(self):
        if self.busy:
            self.status.set('Cancela o espera la reconstrucción antes de iniciar otra captura.'); return
        self.phone_request('start')

    def cancel_processing(self):
        self.cancel.set(); self.status.set('Cancelando… la sesión original y los resultados ya completados se conservan.')

    def preview_worker(self):
        last_status=0.
        while not self.preview_stop.wait(.2):
            ip=self.preview_ip
            if not ip: continue
            if time.monotonic()-last_status>=1:
                last_status=time.monotonic()
                try:
                    with urllib.request.urlopen(f'http://{ip}:8080/status',timeout=1.5) as response:
                        status=json.loads(response.read(65536))
                    self.post('phone_status',status)
                except Exception as error: self.post('phone_lost',str(error))
            try:
                with urllib.request.urlopen(f'http://{ip}:8080/frame.jpg',timeout=1) as response:
                    import io
                    data=response.read(2*1024*1024)
                    if data:
                        image=Image.open(io.BytesIO(data)).copy(); image.thumbnail((340,220))
                        self.post('preview',image)
            except Exception: pass

    def phone_state(self,value):
        self.last_phone_ok=time.monotonic()
        self.phone_active,self.phone_busy=bool(value.get('isScanning')),bool(value.get('busy'))
        phase=value.get('phase','')
        label=PHASES.get(phase,phase or ('Capturando' if self.phone_active else 'Listo'))
        self.phone_phase.set(f'iPhone: {label} · {value.get("frames",0)} fotogramas · {value.get("dropped",0)} omitidos · {value.get("recordedMB",0):.0f} MB sesión')
        error=value.get('captureError','')
        self.phone_phase_label.configure(foreground='#dc2626' if error or phase in ('error','blockedCache') else ('#d97706' if phase in ('pausedCache','waitingSensor') else ''))
        extra=[value.get('status','')]
        if value.get('stopReason'): extra.append('Última parada: '+value['stopReason'])
        if value.get('sensor'): extra.append(value['sensor'])
        self.phone_message.set('\n'.join(item for item in extra if item))
        self.transfer_message.set(f"Transferencia (iPhone): {value.get('transferMessage','')} · Caché del iPhone: {value.get('pendingMB',0):.0f}/{value.get('cacheLimitMB',180):.0f} MB en {value.get('pendingSessions',0)} sesión(es). Bloques recibidos aquí: {self.blocks_received}.")
        if error: self.status.set('iPhone: '+error)
        self.capture_button.configure(state='disabled' if self.phone_active or self.phone_busy or self.busy else 'normal')
        self.stop_button.configure(state='normal' if self.phone_active and not self.phone_busy else 'disabled')
        if self.phone_active and self.busy:
            self.cancel.set(); self.status.set('Captura iniciada en el iPhone: cancelando la reconstrucción para priorizar la captura.')

    def poll(self):
        if self.closed: return
        self.preview_ip=self.ip.get().strip()
        for _ in range(60):
            try: kind,value=self.events.get_nowait()
            except queue.Empty: break
            if kind=='preview':
                photo=ImageTk.PhotoImage(value); self.video.configure(image=photo,text='',height=0); self.video.image=photo
            elif kind=='paired':
                if self.ip.get().strip()!=value: self.ip.set(value); self.status.set(f'iPhone detectado automáticamente en {value}.')
            elif kind=='phone_status': self.phone_state(value)
            elif kind=='phone_lost':
                if time.monotonic()-self.last_phone_ok>4:
                    self.phone_phase.set(f'iPhone: desconectado ({self.ip.get().strip()}:8080 no responde)')
                    self.phone_phase_label.configure(foreground='#dc2626')
            elif kind=='block':
                self.blocks_received+=1
            elif kind=='transfer_error':
                session,error=value
                self.transfer_message.set(f'Transferencia: {error} (sesión {str(session or "?")[:8]}). El iPhone conserva el bloque y reintenta.')
            elif kind=='received':
                self.results={}; self.result_combo['values']=[]; self.result_choice.set(''); self.mesh=None
                self.viewer.clear('Sesión nueva verificada. Pulsa Reconstruir.')
                for button in (self.export_stl,self.export_ply,self.swap_button): button.configure(state='disabled')
                self.session,receipt=value
                text=(f'Sesión verificada en Windows: {receipt["frames"]} fotogramas, {receipt["dropped"]} omitidos durante la captura'
                      +(' (captura interrumpida: se conservan los fotogramas válidos)' if receipt.get('interrupted') else '')+'.')
                self.session_message.set(text+f' Archivo: {Path(self.session).name}')
                self.status.set(text+' Pulsa Reconstruir.')
            elif kind=='progress':
                current,total,accepted,rejected,device,fitness,rmse=value; self.progress['value']=current/max(1,total)*100
                self.status.set(f'{getattr(self,"variant_label","")}{device} · {current}/{total} · {accepted} aceptados · {rejected} rechazados · RMSE {rmse*1000:.2f} mm')
            elif kind=='variant': self.variant_label=f'[{value}] '
            elif kind=='comparison':
                self.busy=False; self.results=value; self.result_combo['values']=list(value)
                self.result_choice.set(next(iter(value))); self.select_result(keep_camera=False)
                self.swap_button.configure(state='normal' if len(value)>1 else 'disabled')
            elif kind=='reconstruction_error':
                self.busy=False; self.status.set(str(value))
            elif kind=='error': self.status.set(str(value))
            else: self.status.set(str(value))
        self.reconstruct_button.configure(state='disabled' if self.busy else 'normal')
        self.cancel_button.configure(state='normal' if self.busy else 'disabled')
        self.update_steps()
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
            except Exception as error: self.post('error',f'Sesión no válida: {error}')
        threading.Thread(target=worker,daemon=True).start()

    def start_reconstruction(self):
        if self.busy: return
        if self.phone_active or self.phone_busy or (hasattr(self,'server') and self.server.upload_lock.locked()):
            self.status.set('Espera a que finalice la captura y la transferencia.'); return
        if self.session is None: self.status.set('Primero finaliza y transfiere una captura, o abre una sesión.'); return
        try: settings=self.read_settings()
        except (ValueError,tk.TclError) as error: self.status.set(str(error)); return
        mode=self.reconstruction_mode.get()
        self.results={}; self.mesh=None; self.result_combo['values']=[]; self.result_choice.set('')
        for button in (self.export_stl,self.export_ply,self.swap_button): button.configure(state='disabled')
        self.busy=True; self.cancel.clear(); path=self.session; self.status.set('Preparando reconstrucción en la RTX…')
        def worker():
            try:
                from dataclasses import replace
                variants=['RTX sola','RTX + IA'] if mode=='Comparar ambas' else [mode]
                results={}
                for variant in variants:
                    ai=variant=='RTX + IA'
                    # Same original capture for both; neural options only in RTX + IA.
                    config=replace(settings,device='CUDA',neural_denoising=ai,neural_alignment=settings.neural_alignment if ai else False)
                    self.post('variant',variant)
                    mesh,report=reconstruct(path,config,lambda *args:self.post('progress',args),self.cancel)
                    report['variant']=variant
                    Path(path).with_suffix('.'+('rtx-ai' if ai else 'rtx')+'.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
                    results[variant]=(mesh,report)
                self.post('comparison',results)
            except InterruptedError as error: self.post('reconstruction_error',f'{error}')
            except Exception as error: self.post('reconstruction_error',f'Reconstrucción detenida: {type(error).__name__}: {error}')
        threading.Thread(target=worker,daemon=True).start()

    def export(self,extension):
        if self.mesh is None: return
        name=self.result_choice.get() or 'resultado'
        path=filedialog.asksaveasfilename(defaultextension='.'+extension,initialfile=f'Bichito3D-{name.replace(" ","").replace("+","-")}.{extension}',
                                          filetypes=[(extension.upper(),'*.'+extension)])
        if not path: return
        try:
            verified=export_selected(self.mesh,path)
            self.status.set(f'Exportado en milímetros: {path} · {verified:,} triángulos = resultado «{name}».')
            self.step_labels[5].configure(bg='#166534',fg='#dcfce7')
        except Exception as error: self.status.set(f'Exportación fallida: {error}')

    def close(self):
        self.closed=True; self.cancel.set(); self.preview_stop.set()
        if hasattr(self,'beacon'): self.beacon.stop()
        if hasattr(self,'blocks'): threading.Thread(target=self.blocks.close,daemon=True).start()
        if hasattr(self,'server'): self.server.server_close()
        if hasattr(self,'pair'): self.pair.server_close()
        self.root.destroy()


def export_selected(mesh,path):
    """Writes the selected result in mm and re-reads the file to confirm it."""
    path=Path(path)
    if hasattr(mesh,'export_mm'): mesh.export_mm(path); expected=len(mesh.triangles)
    else:
        import copy,open3d as o3d
        scaled=copy.deepcopy(mesh); scaled.scale(1000,center=(0,0,0)); expected=len(mesh.triangles)
        if not o3d.io.write_triangle_mesh(str(path),scaled,write_ascii=False): raise ValueError('Open3D no pudo escribir el archivo')
    if path.suffix.lower()=='.stl':
        with open(path,'rb') as file:
            header=file.read(84)
        count=struct.unpack_from('<I',header,80)[0]
        if count!=expected or path.stat().st_size!=84+50*count: raise ValueError(f'STL inconsistente: {count} triángulos, se esperaban {expected}')
        return count
    return expected


def main():
    root=tk.Tk(); OfflineStudio(root); root.mainloop()
