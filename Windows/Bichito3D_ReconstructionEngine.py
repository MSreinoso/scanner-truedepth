"""
================================================================================
 Bichito 3D Pro v2.0 — PC Reconstruction Engine (EinStar Architecture)
 Real-Time Hybrid ICP Alignment + Frame Rejection + TSDF Volumetric Fusion
================================================================================
 Architecture:
   iPhone (Streamer) ──TCP:50506──▶ PC (This Engine)
   - Receives binary frame packets (Depth + RGB + IMU) at 30 FPS
   - Runs Point-to-Plane ICP with IMU priors for spatial alignment
   - Rejects bad frames (fast motion / tracking lost) via RMSE threshold
   - Integrates accepted frames into Open3D TSDF voxel volume
   - Exports watertight Poisson mesh → STL / PLY
================================================================================
"""

import sys
import os
import time
import socket
import struct
import threading
import io
import zlib
import json
import queue

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk

import numpy as np

# Optional heavy imports with graceful fallback
try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    print("[WARN] open3d not installed. Install with: pip install open3d")

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("[WARN] opencv-python not installed. Install with: pip install opencv-python")


# =============================================================================
# Constants & Wire Protocol
# =============================================================================
MAGIC = 0xB1C01700
TCP_PORT = 50506
UDP_DISCOVERY_PORT = 50505
MAX_PAYLOAD_BYTES = 8 * 1024 * 1024

# ICP Alignment Thresholds
ICP_MAX_CORRESPONDENCE_DISTANCE = 0.015  # 15 mm
ICP_ACCEPT_FITNESS = 0.35                # Minimum overlap fraction to accept
ICP_ACCEPT_RMSE = 0.008                  # Maximum RMSE (meters) to accept
ICP_REJECT_RMSE = 0.015                  # Above this = instant reject

# TSDF Volume Parameters
TSDF_VOXEL_LENGTH = 0.002    # 2 mm voxels
TSDF_SDF_TRUNC = 0.010       # 10 mm truncation distance
TSDF_VOLUME_BOUNDS = 0.5     # 500 mm cube centered at origin


# =============================================================================
# Binary Frame Packet Deserializer
# =============================================================================
class FramePacket:
    """Deserialized binary frame from the iPhone streamer."""

    __slots__ = [
        'frame_id', 'timestamp', 'rotation_matrix', 'user_acceleration',
        'intrinsics', 'depth_width', 'depth_height', 'depth_map',
        'rgb_width', 'rgb_height', 'rgb_image'
    ]

    def __init__(self):
        self.frame_id = 0
        self.timestamp = 0.0
        self.rotation_matrix = np.eye(3, dtype=np.float32)
        self.user_acceleration = np.zeros(3, dtype=np.float32)
        self.intrinsics = np.array([500.0, 500.0, 320.0, 240.0], dtype=np.float32)
        self.depth_width = 0
        self.depth_height = 0
        self.depth_map = None      # np.float32 (H, W) in meters
        self.rgb_width = 0
        self.rgb_height = 0
        self.rgb_image = None      # np.uint8 (H, W, 3) BGR

    @staticmethod
    def deserialize(raw_data: bytes) -> 'FramePacket':
        """Parse a complete binary frame packet from raw TCP bytes."""
        pkt = FramePacket()
        offset = 0

        # Magic (4 bytes)
        magic = struct.unpack_from('<I', raw_data, offset)[0]
        offset += 4
        if magic != MAGIC:
            raise ValueError(f"Invalid magic: 0x{magic:08X}, expected 0x{MAGIC:08X}")

        # Frame ID (4 bytes)
        pkt.frame_id = struct.unpack_from('<I', raw_data, offset)[0]
        offset += 4

        # Timestamp (8 bytes)
        pkt.timestamp = struct.unpack_from('<d', raw_data, offset)[0]
        offset += 8

        # IMU Rotation Matrix (9 x float32 = 36 bytes)
        rot_flat = struct.unpack_from('<9f', raw_data, offset)
        pkt.rotation_matrix = np.array(rot_flat, dtype=np.float32).reshape(3, 3)
        offset += 36

        # IMU User Acceleration (3 x float32 = 12 bytes)
        acc = struct.unpack_from('<3f', raw_data, offset)
        pkt.user_acceleration = np.array(acc, dtype=np.float32)
        offset += 12

        # Camera Intrinsics (4 x float32 = 16 bytes)
        intr = struct.unpack_from('<4f', raw_data, offset)
        pkt.intrinsics = np.array(intr, dtype=np.float32)
        offset += 16

        # Depth dimensions (2 x uint16 = 4 bytes)
        pkt.depth_width, pkt.depth_height = struct.unpack_from('<HH', raw_data, offset)
        offset += 4

        # Depth payload size (uint32 = 4 bytes)
        depth_payload_size = struct.unpack_from('<I', raw_data, offset)[0]
        offset += 4

        # Depth payload (zlib-compressed Float16)
        depth_compressed = raw_data[offset:offset + depth_payload_size]
        offset += depth_payload_size
        # Apple's Compression framework emits raw DEFLATE (RFC 1951).
        expected_size = pkt.depth_width * pkt.depth_height * 2
        if not 0 < expected_size <= MAX_PAYLOAD_BYTES:
            raise ValueError("Invalid depth dimensions")
        def inflate(wbits):
            decoder = zlib.decompressobj(wbits)
            decoded = decoder.decompress(depth_compressed, expected_size + 1)
            if len(decoded) != expected_size or not decoder.eof:
                raise ValueError("Invalid depth payload size")
            return decoded
        try:
            depth_raw = inflate(-zlib.MAX_WBITS)
        except (zlib.error, ValueError):
            depth_raw = inflate(zlib.MAX_WBITS)  # Older wrapped senders
        depth_f16 = np.frombuffer(depth_raw, dtype=np.float16).reshape(pkt.depth_height, pkt.depth_width)
        pkt.depth_map = depth_f16.astype(np.float32)

        # RGB dimensions (2 x uint16 = 4 bytes)
        pkt.rgb_width, pkt.rgb_height = struct.unpack_from('<HH', raw_data, offset)
        offset += 4

        # RGB payload size (uint32 = 4 bytes)
        rgb_payload_size = struct.unpack_from('<I', raw_data, offset)[0]
        offset += 4

        # RGB payload (JPEG)
        rgb_jpeg = raw_data[offset:offset + rgb_payload_size]
        offset += rgb_payload_size

        if HAS_CV2:
            rgb_arr = np.frombuffer(rgb_jpeg, dtype=np.uint8)
            pkt.rgb_image = cv2.imdecode(rgb_arr, cv2.IMREAD_COLOR)
        else:
            img = Image.open(io.BytesIO(rgb_jpeg))
            pkt.rgb_image = np.array(img)[:, :, ::-1]  # RGB -> BGR

        return pkt

    def to_rgbd_image(self):
        """Convert to Open3D RGBDImage for TSDF integration."""
        if not HAS_OPEN3D:
            return None

        # Scale depth from meters to Open3D convention (meters, float)
        depth_o3d = o3d.geometry.Image(self.depth_map.copy())

        # Convert BGR to RGB for Open3D
        if self.rgb_image is not None:
            rgb = cv2.resize(self.rgb_image, (self.depth_width, self.depth_height))
            rgb_rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            color_o3d = o3d.geometry.Image(rgb_rgb.astype(np.uint8))
        else:
            color_o3d = o3d.geometry.Image(
                np.full((self.depth_height, self.depth_width, 3), 180, dtype=np.uint8)
            )

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d,
            depth_scale=1.0,         # Already in meters
            depth_trunc=0.6,         # Max 600mm
            convert_rgb_to_intensity=False
        )
        return rgbd

    def get_o3d_intrinsic(self):
        """Build Open3D PinholeCameraIntrinsic from frame intrinsics."""
        if not HAS_OPEN3D:
            return None
        fx, fy, cx, cy = self.intrinsics
        return o3d.camera.PinholeCameraIntrinsic(
            self.depth_width, self.depth_height,
            float(fx), float(fy), float(cx), float(cy)
        )


# =============================================================================
# Hybrid ICP Alignment Engine
# =============================================================================
class AlignmentEngine:
    """
    Performs frame-to-model registration using hybrid ICP with IMU priors.
    Implements frame rejection and tracking loss detection.
    """

    class TrackingState:
        IDLE = "IDLE"
        TRACKING = "TRACKING"
        LOST = "TRACKING_LOST"

    def __init__(self):
        self.state = self.TrackingState.IDLE
        self.global_transform = np.eye(4, dtype=np.float64)
        self.prev_pcd = None
        self.keyframe_buffer = []   # Last N accepted point clouds for relocalization
        self.max_keyframes = 10
        self.accepted_frames = 0
        self.rejected_frames = 0
        self.last_fitness = 0.0
        self.last_rmse = 0.0

    def reset(self):
        self.state = self.TrackingState.IDLE
        self.global_transform = np.eye(4, dtype=np.float64)
        self.prev_pcd = None
        self.keyframe_buffer.clear()
        self.accepted_frames = 0
        self.rejected_frames = 0
        self.last_fitness = 0.0
        self.last_rmse = 0.0

    def build_imu_prior(self, rotation_matrix: np.ndarray) -> np.ndarray:
        """
        Build a 4x4 rigid transformation matrix from IMU rotation data.
        Used as the initial guess for ICP to dramatically speed up convergence.
        """
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = rotation_matrix.astype(np.float64)
        return T

    def align_frame(self, current_pcd, imu_rotation: np.ndarray):
        """
        Align incoming point cloud against previous frame using hybrid ICP.

        Returns:
            (accepted: bool, transform_4x4: np.ndarray, fitness: float, rmse: float)
        """
        if not HAS_OPEN3D:
            return True, np.eye(4), 1.0, 0.0

        # First frame → accept unconditionally
        if self.prev_pcd is None or self.state == self.TrackingState.IDLE:
            self.prev_pcd = current_pcd
            self.state = self.TrackingState.TRACKING
            self.accepted_frames = 1
            self.last_fitness = 1.0
            self.last_rmse = 0.0
            self._add_keyframe(current_pcd)
            return True, np.eye(4), 1.0, 0.0

        # Estimate normals for Point-to-Plane ICP
        current_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30)
        )
        self.prev_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30)
        )

        # ICP with IMU prior as initial guess
        imu_prior = self.build_imu_prior(imu_rotation)

        # Stage 1: Coarse Point-to-Plane ICP
        result = o3d.pipelines.registration.registration_icp(
            current_pcd, self.prev_pcd,
            max_correspondence_distance=ICP_MAX_CORRESPONDENCE_DISTANCE,
            init=imu_prior,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-6,
                relative_rmse=1e-6,
                max_iteration=50
            )
        )

        fitness = result.fitness
        rmse = result.inlier_rmse
        transform = result.transformation

        self.last_fitness = fitness
        self.last_rmse = rmse

        # Frame Rejection Decision
        accepted = False
        if fitness >= ICP_ACCEPT_FITNESS and rmse <= ICP_ACCEPT_RMSE:
            # Good alignment → accept
            accepted = True
            self.global_transform = transform @ self.global_transform
            self.prev_pcd = current_pcd
            self.accepted_frames += 1
            self.state = self.TrackingState.TRACKING
            self._add_keyframe(current_pcd)
        elif rmse > ICP_REJECT_RMSE or fitness < 0.15:
            # Bad alignment → reject and enter tracking lost
            self.rejected_frames += 1
            self.state = self.TrackingState.LOST

            # Attempt relocalization against keyframe buffer
            relocalized = self._try_relocalize(current_pcd)
            if relocalized:
                accepted = True
                self.state = self.TrackingState.TRACKING
        else:
            # Marginal → reject but stay in tracking mode
            self.rejected_frames += 1

        return accepted, self.global_transform.copy(), fitness, rmse

    def _add_keyframe(self, pcd):
        """Store downsampled keyframe for relocalization."""
        kf = pcd.voxel_down_sample(voxel_size=0.005)
        self.keyframe_buffer.append(kf)
        if len(self.keyframe_buffer) > self.max_keyframes:
            self.keyframe_buffer.pop(0)

    def _try_relocalize(self, current_pcd):
        """Attempt to match current frame against stored keyframes."""
        if not self.keyframe_buffer:
            return False

        current_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30)
        )

        for kf in reversed(self.keyframe_buffer):
            kf.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30)
            )
            result = o3d.pipelines.registration.registration_icp(
                current_pcd, kf,
                max_correspondence_distance=ICP_MAX_CORRESPONDENCE_DISTANCE * 2,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=80)
            )
            if result.fitness >= ICP_ACCEPT_FITNESS and result.inlier_rmse <= ICP_ACCEPT_RMSE * 1.5:
                self.global_transform = result.transformation @ self.global_transform
                self.prev_pcd = current_pcd
                self._add_keyframe(current_pcd)
                return True
        return False


# =============================================================================
# TSDF Volume Manager
# =============================================================================
class VolumeIntegrator:
    """Manages the Open3D TSDF Volume for continuous depth fusion."""

    def __init__(self):
        self.volume = None
        self.integrated_frames = 0
        self._create_volume()

    def _create_volume(self):
        if not HAS_OPEN3D:
            return
        self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=TSDF_VOXEL_LENGTH,
            sdf_trunc=TSDF_SDF_TRUNC,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )

    def reset(self):
        self.integrated_frames = 0
        self._create_volume()

    def integrate(self, rgbd_image, intrinsic, extrinsic_4x4: np.ndarray):
        """Integrate an RGBD frame into the TSDF volume."""
        if self.volume is None or rgbd_image is None:
            return
        self.volume.integrate(rgbd_image, intrinsic, np.linalg.inv(extrinsic_4x4))
        self.integrated_frames += 1

    def extract_mesh(self):
        """Extract triangle mesh from the TSDF volume."""
        if self.volume is None:
            return None
        mesh = self.volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        return mesh

    def extract_point_cloud(self):
        """Extract colored point cloud from the TSDF volume."""
        if self.volume is None:
            return None
        return self.volume.extract_point_cloud()

    def extract_poisson_mesh(self, depth=9):
        """Generates watertight CAD manifold mesh via Screened Poisson Surface Reconstruction."""
        if not HAS_OPEN3D or self.volume is None:
            return None
        pcd = self.volume.extract_point_cloud()
        if len(pcd.points) < 100:
            return self.extract_mesh()
        
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.015, max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(k=15)
        
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
        densities_arr = np.asarray(densities)
        if len(densities_arr) > 0:
            cutoff = np.quantile(densities_arr, 0.05)
            vertices_to_remove = densities_arr < cutoff
            mesh.remove_vertices_by_mask(vertices_to_remove)
            
        mesh.compute_vertex_normals()
        return mesh


# =============================================================================
# TCP Binary Frame Receiver
# =============================================================================
class TCPFrameReceiver:
    """
    Persistent TCP server that receives binary frame packets from the iPhone.
    Runs in a dedicated background thread.
    """

    def __init__(self, frame_queue: queue.Queue, port: int = TCP_PORT):
        self.port = port
        self.frame_queue = frame_queue
        self.running = False
        self.server_socket = None
        self.client_addr = None
        self.frames_received = 0
        self.on_client_connected = None
        self.on_client_disconnected = None

    def start(self):
        self.running = True
        t = threading.Thread(target=self._server_loop, daemon=True)
        t.start()

    def stop(self):
        self.running = False
        if self.server_socket:
            try:
                self.server_socket.close()
            except Exception:
                pass

    def _server_loop(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.settimeout(2.0)
        try:
            self.server_socket.bind(('0.0.0.0', self.port))
        except OSError as e:
            print(f"[ERROR] Cannot bind TCP port {self.port}: {e}")
            return
        self.server_socket.listen(1)
        print(f"[TCP] Listening for iPhone on port {self.port}...")

        while self.running:
            try:
                conn, addr = self.server_socket.accept()
                self.client_addr = addr
                print(f"[TCP] iPhone connected from {addr}")
                if self.on_client_connected:
                    self.on_client_connected(addr)
                self._receive_loop(conn)
                if self.on_client_disconnected:
                    self.on_client_disconnected()
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"[TCP] Accept error: {e}")
                    time.sleep(1)

    def _receive_loop(self, conn: socket.socket):
        """Read frames from persistent TCP connection."""
        conn.settimeout(5.0)
        buffer = bytearray()

        while self.running:
            try:
                chunk = conn.recv(262144)  # 256 KB read buffer
                if not chunk:
                    print("[TCP] Client disconnected (EOF)")
                    break
                buffer.extend(chunk)

                # Try to parse complete frames from buffer
                while len(buffer) >= 88:  # Minimum header size before depth payload
                    # Peek at magic
                    magic = struct.unpack_from('<I', buffer, 0)[0]
                    if magic != MAGIC:
                        # Scan for next magic byte
                        idx = buffer.find(struct.pack('<I', MAGIC), 1)
                        if idx > 0:
                            buffer = buffer[idx:]
                        else:
                            buffer = buffer[-3:]  # Magic may span TCP reads.
                        break

                    # We need to peek ahead to determine total packet size
                    # Fixed header before depth payload = 88 bytes
                    if len(buffer) < 88:
                        break

                    # Read depth payload size at offset 84
                    depth_payload_size = struct.unpack_from('<I', buffer, 84)[0]
                    if not 0 < depth_payload_size <= MAX_PAYLOAD_BYTES:
                        raise ValueError("Invalid depth payload length")
                    depth_end = 88 + depth_payload_size

                    # After depth payload: rgb_w(2) + rgb_h(2) + rgb_size(4) + rgb_payload
                    rgb_header_start = depth_end
                    if len(buffer) < rgb_header_start + 8:
                        break  # Need more data

                    rgb_payload_size = struct.unpack_from('<I', buffer, rgb_header_start + 4)[0]
                    if rgb_payload_size > MAX_PAYLOAD_BYTES:
                        raise ValueError("Invalid RGB payload length")
                    total_packet_size = rgb_header_start + 8 + rgb_payload_size

                    if len(buffer) < total_packet_size:
                        break  # Need more data

                    # Extract complete packet
                    packet_bytes = bytes(buffer[:total_packet_size])
                    buffer = buffer[total_packet_size:]

                    try:
                        frame = FramePacket.deserialize(packet_bytes)
                        self.frames_received += 1

                        # Non-blocking enqueue (drop if processing can't keep up)
                        try:
                            self.frame_queue.put_nowait(frame)
                        except queue.Full:
                            pass  # Drop frame (backpressure)
                    except Exception as e:
                        print(f"[TCP] Frame parse error: {e}")

            except socket.timeout:
                continue
            except ConnectionResetError:
                print("[TCP] Connection reset by iPhone")
                break
            except Exception as e:
                print(f"[TCP] Receive error: {e}")
                break

        try:
            conn.close()
        except Exception:
            pass


# =============================================================================
# UDP Server Discovery Beacon
# =============================================================================
class UDPServerBeacon:
    """Broadcasts server availability so the iPhone can auto-discover the PC."""

    def __init__(self, tcp_port: int = TCP_PORT):
        self.tcp_port = tcp_port
        self.running = False
        self._local_ip = self._get_local_ip()

    def start(self):
        self.running = True
        t = threading.Thread(target=self._beacon_loop, daemon=True)
        t.start()

    def stop(self):
        self.running = False

    def _beacon_loop(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.settimeout(1.0)

        while self.running:
            try:
                msg = f"BICHITO3D_SERVER|IP:{self._local_ip}|PORT:{self.tcp_port}"
                sock.sendto(msg.encode('utf-8'), ('<broadcast>', UDP_DISCOVERY_PORT))
            except Exception:
                pass
            time.sleep(1.5)

    @staticmethod
    def _get_local_ip() -> str:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"


# =============================================================================
# STL / PLY Exporter (From Open3D Mesh)
# =============================================================================
class MeshExporter:
    """Exports Open3D triangle meshes to CAD-standard binary STL and PLY."""

    @staticmethod
    def export_stl(mesh, filepath: str):
        """Export to binary STL (millimeters, CAD-compliant header)."""
        if mesh is None:
            return False

        vertices = np.asarray(mesh.vertices) * 1000.0  # meters → mm
        triangles = np.asarray(mesh.triangles)
        normals = np.asarray(mesh.triangle_normals) if mesh.has_triangle_normals() else None

        num_tris = len(triangles)
        header = b"TrueDepth-3D-Scanner [Binary CAD Metric mm] EinStar-v2.0".ljust(80, b'\0')

        with open(filepath, 'wb') as f:
            f.write(header)
            f.write(struct.pack('<I', num_tris))

            for i in range(num_tris):
                idx = triangles[i]
                v1, v2, v3 = vertices[idx[0]], vertices[idx[1]], vertices[idx[2]]

                if normals is not None and i < len(normals):
                    n = normals[i]
                else:
                    edge1 = v2 - v1
                    edge2 = v3 - v1
                    n = np.cross(edge1, edge2)
                    norm_len = np.linalg.norm(n)
                    if norm_len > 1e-6:
                        n = n / norm_len
                    else:
                        n = np.array([0, 0, 1], dtype=np.float32)

                record = struct.pack('<12fH',
                    n[0], n[1], n[2],
                    v1[0], v1[1], v1[2],
                    v2[0], v2[1], v2[2],
                    v3[0], v3[1], v3[2],
                    0
                )
                f.write(record)
        return True

    @staticmethod
    def export_ply(mesh, filepath: str):
        """Export to PLY with vertex colors."""
        if mesh is None or not HAS_OPEN3D:
            return False
        o3d.io.write_triangle_mesh(filepath, mesh, write_vertex_normals=True)
        return True


# =============================================================================
# Main Application: Tkinter UI + Processing Pipeline
# =============================================================================
class ReconstructionStudioApp:
    """
    Bichito 3D Pro v2.0 — EinStar-Grade Reconstruction Studio
    """

    def __init__(self, root):
        self.root = root
        self.root.title("Bichito 3D Pro v2.0 — Reconstruction Engine")
        self.root.geometry("1400x900")
        self.root.configure(bg="#0a0e17")

        # Processing pipeline
        self.frame_queue = queue.Queue(maxsize=5)
        self.alignment = AlignmentEngine()
        self.volume = VolumeIntegrator()
        self.receiver = TCPFrameReceiver(self.frame_queue)
        self.beacon = UDPServerBeacon()

        # State
        self.is_scanning = False
        self.total_frames_received = 0
        self.total_frames_accepted = 0
        self.total_frames_rejected = 0
        self.last_fps = 0.0
        self.fps_timer = time.time()
        self.fps_count = 0
        self.latest_rgb_tk = None
        self.processing_thread = None
        self.final_mesh = None

        # Also maintain backward-compatible HTTP-polling receiver for old iPhone app
        self.legacy_iphone_ip = None
        self.legacy_stream_running = False

        # Load app icon
        icon_path = os.path.join(os.path.dirname(__file__), "bichito_pro_icon_1024.png")
        if os.path.exists(icon_path):
            try:
                self.app_icon = ImageTk.PhotoImage(Image.open(icon_path).resize((64, 64)))
                self.root.iconphoto(False, self.app_icon)
            except Exception:
                pass

        # Build UI
        self._setup_styles()
        self._build_ui()

        # Start services
        self.beacon.start()
        self.receiver.on_client_connected = self._on_iphone_connected
        self.receiver.on_client_disconnected = self._on_iphone_disconnected
        self.receiver.start()

        # Also start legacy UDP listener for backward compat
        self._start_legacy_udp_discovery()

        # Start processing loop
        self._start_processing_pipeline()

    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(".", background="#0a0e17", foreground="#f1f5f9", font=("Segoe UI", 9))
        style.configure("TFrame", background="#0a0e17")
        style.configure("Card.TFrame", background="#111827", relief="solid", borderwidth=1)
        style.configure("TLabel", background="#0a0e17", foreground="#f1f5f9")
        style.configure("Header.TLabel", font=("Segoe UI", 13, "bold"), foreground="#38bdf8")
        style.configure("Status.TLabel", font=("Segoe UI", 10, "bold"), foreground="#10b981")
        style.configure("Alert.TLabel", font=("Segoe UI", 11, "bold"), foreground="#ef4444", background="#111827")
        style.configure("Metric.TLabel", font=("Consolas", 28, "bold"), foreground="#38bdf8", background="#111827")
        style.configure("MetricSm.TLabel", font=("Consolas", 14), foreground="#94a3b8", background="#111827")
        style.configure("Action.TButton", background="#0284c7", foreground="white", font=("Segoe UI", 10, "bold"))
        style.map("Action.TButton", background=[("active", "#0369a1")])
        style.configure("Stop.TButton", background="#ef4444", foreground="white", font=("Segoe UI", 10, "bold"))
        style.map("Stop.TButton", background=[("active", "#dc2626")])
        style.configure("Success.TButton", background="#16a34a", foreground="white", font=("Segoe UI", 10, "bold"))
        style.map("Success.TButton", background=[("active", "#15803d")])

    def _build_ui(self):
        # Top bar
        top_bar = ttk.Frame(self.root, style="Card.TFrame", padding=10)
        top_bar.pack(side=tk.TOP, fill=tk.X, padx=10, pady=6)

        ttk.Label(top_bar, text="Bichito 3D Pro v2.0 — EinStar Reconstruction Engine",
                  style="Header.TLabel").pack(side=tk.LEFT, padx=6)

        self.lbl_status = ttk.Label(top_bar, text="Esperando iPhone (TCP:50506)...",
                                     style="Status.TLabel")
        self.lbl_status.pack(side=tk.LEFT, padx=20)

        self.lbl_fps = ttk.Label(top_bar, text="0 FPS", foreground="#fbbf24",
                                  font=("Consolas", 10, "bold"))
        self.lbl_fps.pack(side=tk.RIGHT, padx=10)

        frame_ip = ttk.Frame(top_bar)
        frame_ip.pack(side=tk.RIGHT, padx=6)
        ttk.Label(frame_ip, text="IP iPhone:").pack(side=tk.LEFT, padx=4)
        self.entry_ip = ttk.Entry(frame_ip, width=16)
        self.entry_ip.insert(0, "192.168.1.X")
        self.entry_ip.pack(side=tk.LEFT, padx=4)
        ttk.Button(frame_ip, text="Conectar Manual",
                   command=self._manual_connect_legacy).pack(side=tk.LEFT, padx=4)

        ttk.Label(self.root,
                  text=f"IP de este PC: {self.beacon._local_ip}  |  TCP {TCP_PORT}  |  Introduce esta IP en el iPhone para conectar.",
                  foreground="#38bdf8").pack(anchor=tk.W, padx=16)

        # Main area: left (video + tracking), right (3D + metrics)
        main_panes = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_panes.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)

        # === LEFT PANEL: Live Video Feed + Tracking State ===
        left_frame = ttk.Frame(main_panes, style="Card.TFrame", padding=10)
        main_panes.add(left_frame, weight=1)

        ttk.Label(left_frame, text="Transmisión TrueDepth RGB",
                  font=("Segoe UI", 10, "bold")).pack(anchor=tk.W, pady=4)

        self.lbl_video = tk.Label(left_frame, bg="#030712",
                                   text="Esperando video del iPhone...")
        self.lbl_video.pack(fill=tk.BOTH, expand=True, pady=6)

        # Tracking State Indicator
        tracking_frame = ttk.Frame(left_frame, style="Card.TFrame", padding=8)
        tracking_frame.pack(fill=tk.X, pady=4)

        self.lbl_tracking_state = ttk.Label(tracking_frame, text="⬤ IDLE",
                                             font=("Segoe UI", 14, "bold"),
                                             foreground="#64748b")
        self.lbl_tracking_state.pack(side=tk.LEFT, padx=10)

        metrics_mini = ttk.Frame(tracking_frame)
        metrics_mini.pack(side=tk.RIGHT, padx=10)

        self.lbl_fitness = ttk.Label(metrics_mini, text="Fitness: —",
                                      font=("Consolas", 9), foreground="#94a3b8")
        self.lbl_fitness.pack(anchor=tk.E)

        self.lbl_rmse = ttk.Label(metrics_mini, text="RMSE: —",
                                   font=("Consolas", 9), foreground="#94a3b8")
        self.lbl_rmse.pack(anchor=tk.E)

        # Scan controls
        scan_controls = ttk.Frame(left_frame)
        scan_controls.pack(fill=tk.X, pady=6)

        self.btn_toggle = ttk.Button(scan_controls, text="INICIAR ESCANEO CONTINUO",
                                      style="Action.TButton",
                                      command=self._toggle_scan)
        self.btn_toggle.pack(fill=tk.X, ipady=8, pady=2)

        btn_row = ttk.Frame(scan_controls)
        btn_row.pack(fill=tk.X, pady=2)

        ttk.Button(btn_row, text="Limpiar", command=self._clear_scan).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        ttk.Button(btn_row, text="Exportar STL", style="Success.TButton",
                   command=self._export_stl).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        ttk.Button(btn_row, text="Exportar PLY",
                   command=self._export_ply).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        # === RIGHT PANEL: 3D Viewport + Metrics ===
        right_frame = ttk.Frame(main_panes, style="Card.TFrame", padding=10)
        main_panes.add(right_frame, weight=2)

        ttk.Label(right_frame, text="Reconstrucción 3D en Tiempo Real",
                  font=("Segoe UI", 10, "bold")).pack(anchor=tk.W, pady=4)

        # Metrics dashboard
        metrics_row = ttk.Frame(right_frame, style="Card.TFrame", padding=8)
        metrics_row.pack(fill=tk.X, pady=4)

        m1 = ttk.Frame(metrics_row, style="Card.TFrame")
        m1.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        ttk.Label(m1, text="FRAMES ACEPTADOS", font=("Segoe UI", 8),
                  foreground="#64748b").pack()
        self.lbl_accepted = ttk.Label(m1, text="0", style="Metric.TLabel")
        self.lbl_accepted.pack()

        m2 = ttk.Frame(metrics_row, style="Card.TFrame")
        m2.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        ttk.Label(m2, text="FRAMES RECHAZADOS", font=("Segoe UI", 8),
                  foreground="#64748b").pack()
        self.lbl_rejected = ttk.Label(m2, text="0", style="Metric.TLabel")
        self.lbl_rejected.pack()

        m3 = ttk.Frame(metrics_row, style="Card.TFrame")
        m3.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        ttk.Label(m3, text="TSDF INTEGRADOS", font=("Segoe UI", 8),
                  foreground="#64748b").pack()
        self.lbl_integrated = ttk.Label(m3, text="0", style="Metric.TLabel")
        self.lbl_integrated.pack()

        # 3D viewport (Open3D visualization or matplotlib fallback)
        self.lbl_3d_placeholder = tk.Label(right_frame, bg="#0a0e17",
            text="La reconstrucción 3D se mostrará aquí.\nInicia un escaneo desde el iPhone.",
            fg="#475569", font=("Segoe UI", 12), justify=tk.CENTER)
        self.lbl_3d_placeholder.pack(fill=tk.BOTH, expand=True, pady=10)

        # Bottom bar: connection info
        bottom_bar = ttk.Frame(self.root, style="Card.TFrame", padding=8)
        bottom_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=6)

        self.lbl_bottom = ttk.Label(bottom_bar,
            text=f"Servidor TCP activo en puerto {TCP_PORT} • Beacon UDP en puerto {UDP_DISCOVERY_PORT}",
            font=("Segoe UI", 9), foreground="#64748b")
        self.lbl_bottom.pack(side=tk.LEFT, padx=6)

        self.lbl_received = ttk.Label(bottom_bar, text="Recibidos: 0",
                                       font=("Consolas", 9), foreground="#64748b")
        self.lbl_received.pack(side=tk.RIGHT, padx=10)

    # =========================================================================
    # Connection Handlers
    # =========================================================================
    def _on_iphone_connected(self, addr):
        self.root.after(0, self._on_legacy_device_discovered, addr[0])
        self.root.after(0, lambda: self.lbl_status.config(
            text=f"¡iPhone conectado! {addr[0]}:{addr[1]} (TCP Binary Stream)",
            foreground="#10b981"
        ))

    def _on_iphone_disconnected(self):
        self.root.after(0, lambda: self.lbl_status.config(
            text="iPhone desconectado. Esperando reconexión...",
            foreground="#fbbf24"
        ))

    # =========================================================================
    # Processing Pipeline (Background Thread)
    # =========================================================================
    def _start_processing_pipeline(self):
        def pipeline_worker():
            while True:
                try:
                    frame = self.frame_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                self.total_frames_received += 1

                if not self.is_scanning:
                    # Not scanning — just show video preview
                    self._update_video_preview(frame)
                    continue

                # === ALIGNMENT PHASE ===
                if HAS_OPEN3D:
                    # Build point cloud from depth
                    pcd = self._depth_to_pointcloud(frame)
                    if pcd is None or len(pcd.points) < 100:
                        continue

                    # Downsample for ICP speed
                    pcd_down = pcd.voxel_down_sample(voxel_size=0.003)

                    # Run hybrid ICP alignment
                    accepted, transform, fitness, rmse = self.alignment.align_frame(
                        pcd_down, frame.rotation_matrix
                    )

                    if accepted:
                        self.total_frames_accepted += 1

                        # === TSDF INTEGRATION ===
                        rgbd = frame.to_rgbd_image()
                        intrinsic = frame.get_o3d_intrinsic()
                        if rgbd and intrinsic:
                            self.volume.integrate(rgbd, intrinsic, transform)

                    else:
                        self.total_frames_rejected += 1

                    # Update UI (throttled)
                    self.root.after(0, self._update_tracking_ui,
                                   self.alignment.state, fitness, rmse)
                else:
                    # Fallback: accept all frames without alignment
                    self.total_frames_accepted += 1

                # Update video preview
                self._update_video_preview(frame)

                # FPS counter
                self.fps_count += 1
                elapsed = time.time() - self.fps_timer
                if elapsed >= 1.0:
                    self.last_fps = self.fps_count / elapsed
                    self.fps_count = 0
                    self.fps_timer = time.time()
                    self.root.after(0, self._update_metrics_ui)

        self.processing_thread = threading.Thread(target=pipeline_worker, daemon=True)
        self.processing_thread.start()

    def _depth_to_pointcloud(self, frame: FramePacket):
        """Convert depth frame to Open3D point cloud using pinhole projection."""
        if not HAS_OPEN3D:
            return None

        depth = frame.depth_map
        fx, fy, cx, cy = frame.intrinsics
        h, w = depth.shape

        # Create mesh grid
        u = np.arange(w, dtype=np.float32)
        v = np.arange(h, dtype=np.float32)
        u, v = np.meshgrid(u, v)

        z = depth
        valid = (z > 0.10) & (z < 0.60) & (~np.isnan(z)) & (~np.isinf(z))

        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        points = np.stack([x[valid], -y[valid], -z[valid]], axis=-1)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))

        # Add colors if available
        if frame.rgb_image is not None:
            rgb_resized = cv2.resize(frame.rgb_image, (w, h)) if HAS_CV2 else None
            if rgb_resized is not None:
                colors = rgb_resized[valid][:, ::-1].astype(np.float64) / 255.0  # BGR→RGB, normalize
                pcd.colors = o3d.utility.Vector3dVector(colors)

        return pcd

    def _update_video_preview(self, frame: FramePacket):
        """Show the latest RGB frame in the UI."""
        if frame.rgb_image is None:
            return
        try:
            rgb = frame.rgb_image[:, :, ::-1] if HAS_CV2 else frame.rgb_image  # BGR→RGB
            img = Image.fromarray(rgb)
            img.thumbnail((420, 320))
            tk_img = ImageTk.PhotoImage(img)
            self.root.after(0, self._set_video_image, tk_img)
        except Exception:
            pass

    def _set_video_image(self, tk_img):
        self.latest_rgb_tk = tk_img
        self.lbl_video.config(image=tk_img, text="")
        self.lbl_video.image = tk_img

    def _update_tracking_ui(self, state, fitness, rmse):
        state_colors = {
            AlignmentEngine.TrackingState.IDLE: ("#64748b", "⬤ IDLE"),
            AlignmentEngine.TrackingState.TRACKING: ("#10b981", "⬤ TRACKING"),
            AlignmentEngine.TrackingState.LOST: ("#ef4444", "⚠ TRACKING LOST — Regresa a la posición anterior"),
        }
        color, text = state_colors.get(state, ("#64748b", "⬤ UNKNOWN"))
        self.lbl_tracking_state.config(text=text, foreground=color)
        self.lbl_fitness.config(text=f"Fitness: {fitness:.4f}")
        self.lbl_rmse.config(text=f"RMSE: {rmse*1000:.2f} mm")

    def _update_metrics_ui(self):
        self.lbl_fps.config(text=f"{self.last_fps:.0f} FPS")
        self.lbl_accepted.config(text=str(self.total_frames_accepted))
        self.lbl_rejected.config(text=str(self.total_frames_rejected))
        self.lbl_integrated.config(text=str(self.volume.integrated_frames))
        self.lbl_received.config(text=f"Recibidos: {self.receiver.frames_received}")

    # =========================================================================
    # Scan Control
    # =========================================================================
    def _toggle_scan(self):
        if self.is_scanning:
            self._stop_scan()
        else:
            self._start_scan()

    def _start_scan(self):
        self.is_scanning = True
        self.alignment.reset()
        self.volume.reset()
        self.total_frames_accepted = 0
        self.total_frames_rejected = 0
        self.fps_count = 0
        self.fps_timer = time.time()

        self.btn_toggle.config(text="DETENER ESCANEO Y GENERAR MALLA",
                               style="Stop.TButton")
        self.lbl_status.config(text="● GRABANDO ESCANEO 3D CONTINUO...",
                               foreground="#ef4444")

        # Also toggle on iPhone via legacy HTTP if connected
        self._legacy_toggle_scan()

    def _stop_scan(self):
        self.is_scanning = False
        self.btn_toggle.config(text="INICIAR ESCANEO CONTINUO",
                               style="Action.TButton")
        self.lbl_status.config(text="Generando malla de superficie...",
                               foreground="#fbbf24")

        # Also toggle on iPhone via legacy HTTP
        self._legacy_toggle_scan()

        # Extract final mesh in background
        def finalize():
            if HAS_OPEN3D and self.volume.integrated_frames > 0:
                self.root.after(0, lambda: self.lbl_status.config(
                    text="Ejecutando reconstrucción de superficie Poisson CAD...",
                    foreground="#38bdf8"
                ))
                self.final_mesh = self.volume.extract_poisson_mesh(depth=9)
                if self.final_mesh is None or len(self.final_mesh.triangles) == 0:
                    self.final_mesh = self.volume.extract_mesh()
                
                tri_count = len(self.final_mesh.triangles) if self.final_mesh else 0
                vert_count = len(self.final_mesh.vertices) if self.final_mesh else 0
                self.root.after(0, lambda: self.lbl_status.config(
                    text=f"Malla Poisson CAD generada: {vert_count:,} vértices, {tri_count:,} triángulos.",
                    foreground="#10b981"
                ))
            else:
                self.root.after(0, lambda: self.lbl_status.config(
                    text="Escaneo finalizado (sin Open3D → sin malla volumétrica).",
                    foreground="#fbbf24"
                ))

        threading.Thread(target=finalize, daemon=True).start()

    def _clear_scan(self):
        self.is_scanning = False
        self.alignment.reset()
        self.volume.reset()
        self.final_mesh = None
        self.total_frames_accepted = 0
        self.total_frames_rejected = 0

        self.lbl_accepted.config(text="0")
        self.lbl_rejected.config(text="0")
        self.lbl_integrated.config(text="0")
        self.lbl_tracking_state.config(text="⬤ IDLE", foreground="#64748b")
        self.lbl_status.config(text="Escaneo limpiado. Listo.", foreground="#38bdf8")
        self.btn_toggle.config(text="INICIAR ESCANEO CONTINUO", style="Action.TButton")

        # Also clear on iPhone
        self._legacy_clear_scan()

    # =========================================================================
    # Export
    # =========================================================================
    def _export_stl(self):
        if self.final_mesh is None:
            messagebox.showwarning("Aviso", "No hay malla reconstruida. Realiza un escaneo primero.")
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".stl",
            filetypes=[("Binary CAD STL", "*.stl")],
            initialfile="Bichito3D_EinStar_Scan.stl"
        )
        if not path:
            return

        if MeshExporter.export_stl(self.final_mesh, path):
            messagebox.showinfo("Exportación Exitosa",
                f"STL binario CAD guardado:\n{os.path.basename(path)}\n\n"
                f"Listo para Cura, PrusaSlicer, Meshmixer o Fusion 360.")

    def _export_ply(self):
        if self.final_mesh is None:
            messagebox.showwarning("Aviso", "No hay malla reconstruida.")
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".ply",
            filetypes=[("Stanford PLY", "*.ply")],
            initialfile="Bichito3D_EinStar_Color.ply"
        )
        if not path:
            return

        if MeshExporter.export_ply(self.final_mesh, path):
            messagebox.showinfo("PLY Guardado", f"Archivo PLY con color guardado en:\n{path}")

    # =========================================================================
    # Legacy HTTP Compatibility (for current iPhone app version)
    # =========================================================================
    def _start_legacy_udp_discovery(self):
        import urllib.request

        def udp_listener():
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("", UDP_DISCOVERY_PORT))
            except Exception:
                return

            while True:
                try:
                    data, addr = sock.recvfrom(1024)
                    text = data.decode("utf-8", errors="ignore")
                    if text.startswith("TRUEDEPTH_3D_BEACON"):
                        ip = addr[0]
                        for part in text.split("|"):
                            if part.startswith("IP:"):
                                ip = part.replace("IP:", "").strip()
                        self.root.after(0, self._on_legacy_device_discovered, ip)
                except Exception:
                    time.sleep(1)

        threading.Thread(target=udp_listener, daemon=True).start()

    def _on_legacy_device_discovered(self, ip):
        if not self.legacy_iphone_ip or self.legacy_iphone_ip != ip:
            self.legacy_iphone_ip = ip
            self.entry_ip.delete(0, tk.END)
            self.entry_ip.insert(0, ip)
            if not self.legacy_stream_running:
                self._start_legacy_video_stream()
                self._start_legacy_telemetry()

    def _manual_connect_legacy(self):
        ip = self.entry_ip.get().strip()
        if ip:
            self._on_legacy_device_discovered(ip)
            # Ask the phone to stream back to the HTTP request's source IP.
            # This also works when UDP broadcast is blocked by iOS or the router.
            def connect_stream():
                import urllib.request
                try:
                    with urllib.request.urlopen(f"http://{ip}:8080/connect", timeout=5) as resp:
                        resp.read()
                except Exception as error:
                    self.root.after(0, self.lbl_status.config,
                                    {"text": f"Conexión manual: {error}", "foreground": "#fbbf24"})
            threading.Thread(target=connect_stream, daemon=True).start()

    def _start_legacy_video_stream(self):
        """Poll JPEG frames from the current iPhone app's HTTP server."""
        import urllib.request
        self.legacy_stream_running = True

        def worker():
            while self.legacy_stream_running:
                if not self.legacy_iphone_ip:
                    time.sleep(0.5)
                    continue
                try:
                    url = f"http://{self.legacy_iphone_ip}:8080/frame.jpg?t={time.time()}"
                    req = urllib.request.Request(url, headers={"User-Agent": "BichitoStudioPC"})
                    with urllib.request.urlopen(req, timeout=1.0) as resp:
                        img_bytes = resp.read()
                        img = Image.open(io.BytesIO(img_bytes))
                        img.thumbnail((420, 320))
                        tk_img = ImageTk.PhotoImage(img)
                        self.root.after(0, self._set_video_image, tk_img)
                    time.sleep(0.04)
                except Exception:
                    time.sleep(0.2)

        threading.Thread(target=worker, daemon=True).start()

    def _start_legacy_telemetry(self):
        """Poll /status and /live_points from current iPhone app."""
        import urllib.request

        def worker():
            while True:
                if not self.legacy_iphone_ip:
                    time.sleep(0.8)
                    continue
                try:
                    url = f"http://{self.legacy_iphone_ip}:8080/status"
                    with urllib.request.urlopen(urllib.request.Request(url), timeout=1.0) as resp:
                        data = json.loads(resp.read().decode("utf-8"))
                        pts = data.get("points", 0)
                        tris = data.get("triangles", 0)
                        is_scan = data.get("isScanning", False)
                        self.root.after(0, self._update_legacy_state, is_scan, pts, tris)
                except Exception:
                    pass
                time.sleep(0.35)

        threading.Thread(target=worker, daemon=True).start()

    def _update_legacy_state(self, is_scan, pts, tris):
        if is_scan and not self.is_scanning:
            self.is_scanning = True
            self.btn_toggle.config(text="DETENER ESCANEO Y GENERAR MALLA",
                                   style="Stop.TButton")
        elif not is_scan and self.is_scanning:
            self.is_scanning = False
            self.btn_toggle.config(text="INICIAR ESCANEO CONTINUO",
                                   style="Action.TButton")
        self.lbl_accepted.config(text=str(pts))
        self.lbl_integrated.config(text=str(tris))

    def _legacy_toggle_scan(self):
        import urllib.request
        if not self.legacy_iphone_ip:
            return
        def worker():
            try:
                url = f"http://{self.legacy_iphone_ip}:8080/toggle"
                urllib.request.urlopen(urllib.request.Request(url, method="POST"), timeout=3)
            except Exception:
                pass
        threading.Thread(target=worker, daemon=True).start()

    def _legacy_clear_scan(self):
        import urllib.request
        if not self.legacy_iphone_ip:
            return
        def worker():
            try:
                url = f"http://{self.legacy_iphone_ip}:8080/clear"
                urllib.request.urlopen(urllib.request.Request(url, method="POST"), timeout=3)
            except Exception:
                pass
        threading.Thread(target=worker, daemon=True).start()


# =============================================================================
# Entry Point
# =============================================================================
if __name__ == "__main__":
    import sys
    if len(sys.argv)==3 and sys.argv[1]=='--diagnostic':
        from bundle_checks import run
        raise SystemExit(run(sys.argv[2]))
    from offline_pipeline import main
    main()
