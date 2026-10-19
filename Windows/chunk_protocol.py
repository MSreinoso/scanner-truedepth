"""Durable v3 binary WebSocket blocks. ACK means fsync, not an in-memory enqueue."""
from pathlib import Path
import hashlib
import io
import json
import os
import struct
import threading
import uuid
import numpy as np
from PIL import Image
import zstandard as zstd

MAGIC_PNG = 0xB1C01703
MAGIC_REPEAT = 0xB1C01704
MAX_BLOCK = 8 * 1024 * 1024


def decode_lossless_packet(packet, previous=None):
    if not 100 <= len(packet) <= MAX_BLOCK: raise ValueError('Bloque fuera de límites')
    magic=struct.unpack_from('<I',packet)[0]
    dw,dh,depth_size=struct.unpack_from('<HHI',packet,80)
    if not 0<dw*dh<=2_000_000 or depth_size>MAX_BLOCK: raise ValueError('Dimensiones de profundidad inválidas')
    rgb_header=88+depth_size
    if rgb_header+12>len(packet): raise ValueError('Bloque incompleto')
    rw,rh,png_size=struct.unpack_from('<HHI',packet,rgb_header)
    if not 0<rw*rh<=2_000_000 or rgb_header+8+png_size+4>len(packet): raise ValueError('Dimensiones RGB inválidas')
    meta_start=rgb_header+8+png_size
    meta_size=struct.unpack_from('<I',packet,meta_start)[0]
    if meta_size>262144 or meta_start+4+meta_size!=len(packet): raise ValueError('Calibración incompleta')
    if magic==MAGIC_PNG:
        raw=zstd.ZstdDecompressor().decompress(packet[88:rgb_header],max_output_size=dw*dh*4)
        if len(raw)!=dw*dh*4: raise ValueError('Profundidad Zstd inválida')
        depth=np.frombuffer(raw,np.uint8).reshape(4,-1).T.copy().tobytes()
        with Image.open(io.BytesIO(packet[rgb_header+8:meta_start])) as image:
            if image.size!=(rw,rh) or image.format!='PNG' or image.mode not in ('RGB','RGBA'): raise ValueError('PNG inválido')
            rgba=np.asarray(image.convert('RGBA'))
            rgb=rgba[:,:,[2,1,0,3]].copy().tobytes()
    elif magic==MAGIC_REPEAT:
        if previous is None or depth_size!=0 or png_size!=0: raise ValueError('Referencia redundante inválida')
        prev_depth=struct.unpack_from('<I',previous,84)[0]
        prev_rgb=88+prev_depth
        depth=previous[88:prev_rgb]
        size=struct.unpack_from('<I',previous,prev_rgb+4)[0]
        rgb=previous[prev_rgb+8:prev_rgb+8+size]
        if len(depth)!=dw*dh*4 or len(rgb)!=rw*rh*4: raise ValueError('Referencia incompatible')
    else: raise ValueError('Codec de bloque desconocido')
    header=bytearray(packet[:88])
    struct.pack_into('<I',header,0,0xB1C01702); struct.pack_into('<I',header,84,len(depth))
    return bytes(header)+depth+struct.pack('<HHI',rw,rh,len(rgb))+rgb+packet[meta_start:]


def stream_records(path):
    from offline_pipeline import decode_frame
    path=Path(path); meta=json.loads(path.read_text(encoding='utf-8'))
    session=str(uuid.UUID(meta['session']))
    folder=path.parent/'Chunks'/session
    if meta.get('version')!=3 or not isinstance(meta.get('frames'),int) or not 0<meta['frames']<=1_000_000: raise ValueError('Manifiesto v3 inválido')
    if not isinstance(meta.get('dropped'),int) or meta['dropped']<0: raise ValueError('Contador inválido')
    previous=None; previous_time=-float('inf')
    for seq in range(meta['frames']):
        packet=(folder/f'{seq:09d}.b3dc').read_bytes()
        previous=decode_lossless_packet(packet,previous)
        frame=decode_frame(previous)
        if frame.frame_id!=seq or frame.timestamp<=previous_time: raise ValueError('Orden temporal inválido')
        previous_time=frame.timestamp
        yield frame
    return meta


class BlockStore:
    def __init__(self,directory):
        self.directory=Path(directory); self.directory.mkdir(parents=True,exist_ok=True)
        self.lock=threading.Lock()

    def folder(self,session):
        return self.directory/'Chunks'/str(uuid.UUID(session))

    def _previous(self,folder,seq):
        # Exact repeats share pixels with the nearest preceding full block.
        while seq>=0:
            packet=(folder/f'{seq:09d}.b3dc').read_bytes()
            if struct.unpack_from('<I',packet)[0]==MAGIC_PNG:
                return decode_lossless_packet(packet)
            seq-=1
        raise ValueError('Falta keyframe de referencia')

    def accept(self,session,message):
        from offline_pipeline import decode_frame
        if not 136<=len(message)<=MAX_BLOCK+36: raise ValueError('Mensaje fuera de límites')
        seq=struct.unpack_from('<I',message)[0]
        if seq>=1_000_000: raise ValueError('Secuencia fuera de límites')
        digest=message[4:36].hex(); packet=message[36:]
        if hashlib.sha256(packet).hexdigest()!=digest: raise ValueError('Checksum incorrecto')
        folder=self.folder(session); folder.mkdir(parents=True,exist_ok=True)
        with self.lock:
            target=folder/f'{seq:09d}.b3dc'
            if target.exists():
                if hashlib.sha256(target.read_bytes()).hexdigest()!=digest: raise ValueError('Conflicto de secuencia')
            else:
                previous=self._previous(folder,seq-1) if struct.unpack_from('<I',packet)[0]==MAGIC_REPEAT else None
                raw=decode_lossless_packet(packet,previous)
                if decode_frame(raw).frame_id!=seq: raise ValueError('ID de frame incorrecto')
                partial=target.with_suffix('.partial')
                try:
                    with open(partial,'wb') as file: file.write(packet); file.flush(); os.fsync(file.fileno())
                    os.replace(partial,target)
                finally:
                    if partial.exists(): partial.unlink()
        return {'op':'ack','session':str(uuid.UUID(session)),'seq':seq,'sha256':digest}

    def finish(self,session,frames,dropped,interrupted=False):
        from offline_pipeline import validate_session
        session=str(uuid.UUID(session))
        if not isinstance(frames,int) or not 0<frames<=1_000_000 or not isinstance(dropped,int) or dropped<0: raise ValueError('Manifiesto incorrecto')
        target=self.directory/(session+'.b3dstream')
        partial=self.directory/(session+'.partial.b3dstream')
        value={'version':3,'session':session,'frames':frames,'dropped':dropped,'interrupted':bool(interrupted)}
        with self.lock:
            try:
                with open(partial,'w',encoding='utf-8') as file: json.dump(value,file); file.flush(); os.fsync(file.fileno())
                validate_session(partial)
                if target.exists():
                    if json.loads(target.read_text())!=value: raise ValueError('Sesión finalizada con otro manifiesto')
                    partial.unlink()
                else: os.replace(partial,target)
            finally:
                if partial.exists(): partial.unlink()
        return target,{'op':'committed',**value}


class BlockServer:
    def __init__(self,directory,callback,host='0.0.0.0',port=50508):
        from websockets.sync.server import serve
        self.store=BlockStore(directory); self.callback=callback
        self.server=serve(self.handle,host,port,max_size=MAX_BLOCK+36,compression=None,open_timeout=10,close_timeout=5)

    def handle(self,socket):
        try:
            hello=json.loads(socket.recv(timeout=15))
            if hello.get('op')!='open' or hello.get('version')!=3: raise ValueError('Protocolo incompatible')
            session=str(uuid.UUID(hello['session']))
            socket.send(json.dumps({'op':'opened','session':session}))
            for message in socket:
                if isinstance(message,bytes): socket.send(json.dumps(self.store.accept(session,message)))
                else:
                    value=json.loads(message)
                    if value.get('op')!='finish' or value.get('session')!=session: raise ValueError('Comando inválido')
                    path,receipt=self.store.finish(session,value['frames'],value['dropped'],value.get('interrupted',False))
                    socket.send(json.dumps(receipt)); self.callback(path,receipt); return
        except Exception as error:
            try: socket.send(json.dumps({'error':str(error)}))
            except Exception: pass

    def start(self): threading.Thread(target=self.server.serve_forever,daemon=True).start()
    def close(self): self.server.shutdown()
