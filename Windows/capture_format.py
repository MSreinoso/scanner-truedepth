"""Capture/session file formats shared by the UI, CUDA backend and protocol.

No GUI or GPU dependencies: imported by CI on macOS and by the block server.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import math
import os
import re
import shutil
import struct
import uuid
import zlib
import numpy as np

HEADER = b'B3DREC02'
RAW_MAGIC = 0xB1C01702
MAX_RECORD = 8 * 1024 * 1024
MAX_SESSION = 32 * 1024**3



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
