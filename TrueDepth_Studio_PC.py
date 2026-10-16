"""
================================================================================
 TrueDepth 3D Scanner - Windows Desktop Studio (Pro Edition)
 Arquitectura de Software para Escaneo 3D Sub-milimétrico de Grado Industrial
================================================================================
 Características:
 1. Autodescubrimiento UDP en segundo plano (Puerto 50505).
 2. Transmisión de vídeo en directo con baja latencia.
 3. Recorte dinámico de plano de referencia (Base Plane Clipping) en tiempo real.
 4. Detección automática del plano de mesa mediante RANSAC 3D.
 5. Exportación directa a binario CAD STL (.stl) y nube de puntos coloreada (.ply).
================================================================================
"""

import sys
import os
import time
import socket
import threading
import struct
import io
import urllib.request
import urllib.parse
import json

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class TrueDepthStudioApp:
    def __init__(self, root):
        self.root = root
        self.root.title("TrueDepth 3D Studio - Grado Industrial")
        self.root.geometry("1300x820")
        self.root.configure(bg="#0b0f19")
        
        # Estado de red
        self.iphone_ip = None
        self.iphone_port = 8080
        self.is_connected = False
        self.stream_running = False
        
        # Datos geométricos
        self.raw_triangles = None  # numpy array (N, 3, 3) en mm
        self.raw_normals = None    # numpy array (N, 3)
        self.clipped_triangles = None
        
        # Parámetros de recorte
        self.plane_offset_mm = 0.0
        self.ransac_plane = None  # (a, b, c, d)
        
        # Inicializar interfaz y sockets
        self._setup_styles()
        self._build_ui()
        self._start_udp_discovery()

    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(".", background="#0b0f19", foreground="#f1f5f9", font=("Segoe UI", 9))
        style.configure("TFrame", background="#0b0f19")
        style.configure("Card.TFrame", background="#111827", relief="solid", borderwidth=1)
        style.configure("TLabel", background="#0b0f19", foreground="#f1f5f9")
        style.configure("Header.TLabel", font=("Segoe UI", 12, "bold"), foreground="#38bdf8")
        style.configure("Status.TLabel", font=("Segoe UI", 9, "bold"), foreground="#10b981")
        style.configure("Action.TButton", background="#0284c7", foreground="white", font=("Segoe UI", 10, "bold"))
        style.map("Action.TButton", background=[("active", "#0369a1")])
        style.configure("Success.TButton", background="#16a34a", foreground="white", font=("Segoe UI", 9, "bold"))
        style.map("Success.TButton", background=[("active", "#15803d")])

    def _build_ui(self):
        # 1. Barra de Herramientas Superior (Telemetría de Red)
        top_bar = ttk.Frame(self.root, style="Card.TFrame", padding=10)
        top_bar.pack(side=tk.TOP, fill=tk.X, padx=10, pady=6)
        
        ttk.Label(top_bar, text="TrueDepth Industrial 3D Studio", style="Header.TLabel").pack(side=tk.LEFT, padx=6)
        
        self.lbl_status = ttk.Label(top_bar, text="Buscando iPhone vía UDP (50505)...", style="Status.TLabel")
        self.lbl_status.pack(side=tk.LEFT, padx=16)
        
        # Entrada manual de IP por si el broadcast Wi-Fi está bloqueado
        frame_ip = ttk.Frame(top_bar)
        frame_ip.pack(side=tk.RIGHT, padx=6)
        ttk.Label(frame_ip, text="IP iPhone:").pack(side=tk.LEFT, padx=4)
        self.entry_ip = ttk.Entry(frame_ip, width=16)
        self.entry_ip.insert(0, "192.168.1.X")
        self.entry_ip.pack(side=tk.LEFT, padx=4)
        ttk.Button(frame_ip, text="Conectar Manual", command=self._manual_connect).pack(side=tk.LEFT, padx=4)

        # 2. Área de Trabajo Principal (2 Paneles: Video en Vivo & Visualizador 3D)
        main_panes = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_panes.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)
        
        # Panel Izquierdo: Monitor de Vídeo y Calibración
        left_frame = ttk.Frame(main_panes, style="Card.TFrame", padding=10)
        main_panes.add(left_frame, weight=1)
        
        ttk.Label(left_frame, text="1. Transmisión TrueDepth Infrarrojo/RGB", font=("Segoe UI", 10, "bold")).pack(anchor=tk.W, pady=4)
        self.lbl_video = tk.Label(left_frame, bg="#030712", text="Esperando transmisión de cámara...")
        self.lbl_video.pack(fill=tk.BOTH, expand=True, pady=6)
        
        # Controles de Disparo en Panel Izquierdo
        scan_controls = ttk.Frame(left_frame)
        scan_controls.pack(fill=tk.X, pady=6)
        
        self.btn_scan = ttk.Button(scan_controls, text="Capturar Malla 3D (Filtro SOR Activo)", style="Action.TButton", command=self._trigger_scan)
        self.btn_scan.pack(fill=tk.X, ipady=6)
        
        # Panel Derecho: Visualizador 3D y Recorte de Plano
        right_frame = ttk.Frame(main_panes, style="Card.TFrame", padding=10)
        main_panes.add(right_frame, weight=2)
        
        header_3d = ttk.Frame(right_frame)
        header_3d.pack(fill=tk.X)
        ttk.Label(header_3d, text="2. Malla Geométrica 3D (Sub-milimétrica)", font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT)
        self.lbl_tri_count = ttk.Label(header_3d, text="0 Facetas CAD", foreground="#38bdf8")
        self.lbl_tri_count.pack(side=tk.RIGHT)
        
        # Canvas Matplotlib 3D
        self.fig = Figure(figsize=(6, 5), dpi=90, facecolor="#0b0f19")
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_facecolor("#0b0f19")
        self._format_3d_axes()
        
        self.canvas_3d = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas_3d.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=6)
        
        # 3. Barra de Parámetros Inferior (Base Plane Clipping & Export)
        bottom_frame = ttk.Frame(self.root, style="Card.TFrame", padding=10)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=6)
        
        # Slider de Recorte Dinámico de Base (Plano de Mesa)
        slider_frame = ttk.Frame(bottom_frame)
        slider_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        
        self.lbl_plane_val = ttk.Label(slider_frame, text="Recorte Plano Base (Mesa): 0.0 mm (Desactivado)")
        self.lbl_plane_val.pack(anchor=tk.W)
        
        self.slider_plane = ttk.Scale(slider_frame, from_=-50.0, to=50.0, orient=tk.HORIZONTAL, command=self._on_slider_plane_change)
        self.slider_plane.set(0.0)
        self.slider_plane.pack(fill=tk.X, pady=4)
        
        # Botones de Acción
        action_frame = ttk.Frame(bottom_frame)
        action_frame.pack(side=tk.RIGHT, padx=10)
        
        ttk.Button(action_frame, text="Detectar Mesa con RANSAC", command=self._auto_detect_table_plane).pack(side=tk.LEFT, padx=4)
        ttk.Button(action_frame, text="Exportar STL Binario CAD", style="Success.TButton", command=self._export_cad_stl).pack(side=tk.LEFT, padx=4)
        ttk.Button(action_frame, text="Exportar PLY Color", command=self._export_color_ply).pack(side=tk.LEFT, padx=4)

    def _format_3d_axes(self):
        self.ax.tick_params(colors="#94a3b8", labelsize=8)
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False
        self.ax.xaxis.pane.set_edgecolor("#1e293b")
        self.ax.yaxis.pane.set_edgecolor("#1e293b")
        self.ax.zaxis.pane.set_edgecolor("#1e293b")
        self.ax.set_xlabel("X (mm)", color="#94a3b8", fontsize=8)
        self.ax.set_ylabel("Y (mm)", color="#94a3b8", fontsize=8)
        self.ax.set_zlabel("Z (mm)", color="#94a3b8", fontsize=8)

    # =========================================================================
    # Red y Autodescubrimiento UDP
    # =========================================================================
    def _start_udp_discovery(self):
        def udp_listener():
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("", 50505))
            except Exception as e:
                print(f"Error binding UDP: {e}")
                return
            
            while True:
                try:
                    data, addr = sock.recvfrom(1024)
                    text = data.decode("utf-8", errors="ignore")
                    if text.startswith("TRUEDEPTH_3D_BEACON"):
                        # Protocolo: TRUEDEPTH_3D_BEACON|IP:<ip>|PORT:<port>|READY
                        parts = text.split("|")
                        ip_found = addr[0]
                        for p in parts:
                            if p.startswith("IP:"):
                                ip_found = p.replace("IP:", "").strip()
                        self.root.after(0, self._on_device_discovered, ip_found)
                except Exception as e:
                    time.sleep(1)

        t = threading.Thread(target=udp_listener, daemon=True)
        t.start()

    def _on_device_discovered(self, ip):
        if not self.is_connected or self.iphone_ip != ip:
            self.iphone_ip = ip
            self.is_connected = True
            self.lbl_status.config(text=f"¡iPhone conectado!: http://{ip}:{self.iphone_port}", foreground="#10b981")
            self.entry_ip.delete(0, tk.END)
            self.entry_ip.insert(0, ip)
            if not self.stream_running:
                self._start_video_stream()

    def _manual_connect(self):
        ip = self.entry_ip.get().strip()
        if ip:
            self._on_device_discovered(ip)

    # =========================================================================
    # Transmisión de Vídeo en Directo
    # =========================================================================
    def _start_video_stream(self):
        self.stream_running = True
        def worker():
            while self.stream_running:
                if not self.iphone_ip:
                    time.sleep(0.5)
                    continue
                try:
                    url = f"http://{self.iphone_ip}:{self.iphone_port}/frame.jpg?t={time.time()}"
                    req = urllib.request.Request(url, headers={"User-Agent": "TrueDepthStudioPC"})
                    with urllib.request.urlopen(req, timeout=1.0) as resp:
                        img_bytes = resp.read()
                        img = Image.open(io.BytesIO(img_bytes))
                        # Redimensionar manteniendo proporción
                        img.thumbnail((420, 320))
                        tk_img = ImageTk.PhotoImage(img)
                        self.root.after(0, self._update_video_label, tk_img)
                    time.sleep(0.04) # ~25 FPS
                except Exception:
                    time.sleep(0.2)
        
        t = threading.Thread(target=worker, daemon=True)
        t.start()

    def _update_video_label(self, tk_img):
        self.lbl_video.config(image=tk_img, text="")
        self.lbl_video.image = tk_img

    # =========================================================================
    # Adquisición de Malla y Procesamiento 3D
    # =========================================================================
    def _trigger_scan(self):
        if not self.iphone_ip:
            messagebox.showwarning("Sin Conexión", "Conecta el iPhone a la misma red Wi-Fi.")
            return
        
        self.lbl_status.config(text="Procesando escaneo en iPhone (Filtro SOR)...", foreground="#38bdf8")
        self.btn_scan.config(state=tk.DISABLED)
        
        def worker():
            try:
                # 1. Solicitar escaneo con SOR activado
                scan_url = f"http://{self.iphone_ip}:{self.iphone_port}/scan?max=0.55&sor=1"
                req = urllib.request.Request(scan_url)
                with urllib.request.urlopen(req, timeout=10.0) as resp:
                    res_json = json.loads(resp.read().decode("utf-8"))
                    tri_count = res_json.get("triangles", 0)
                
                if tri_count == 0:
                    self.root.after(0, self._on_scan_failed, "No se encontraron suficientes puntos en el rango.")
                    return
                
                # 2. Descargar STL binario
                stl_url = f"http://{self.iphone_ip}:{self.iphone_port}/model.stl?t={time.time()}"
                with urllib.request.urlopen(stl_url, timeout=10.0) as resp:
                    stl_bytes = resp.read()
                
                # 3. Parsear facetas binarias
                triangles, normals = self._parse_binary_stl(stl_bytes)
                self.root.after(0, self._on_scan_success, triangles, normals)
            except Exception as e:
                self.root.after(0, self._on_scan_failed, str(e))

        threading.Thread(target=worker, daemon=True).start()

    def _on_scan_failed(self, msg):
        self.btn_scan.config(state=tk.NORMAL)
        self.lbl_status.config(text=f"Fallo en escaneo: {msg}", foreground="#ef4444")
        messagebox.showerror("Error de Escaneo", msg)

    def _on_scan_success(self, triangles, normals):
        self.btn_scan.config(state=tk.NORMAL)
        self.raw_triangles = triangles
        self.raw_normals = normals
        self.lbl_status.config(text="¡Malla 3D adquirida con éxito!", foreground="#10b981")
        self.lbl_tri_count.config(text=f"{len(triangles):,} Facetas CAD")
        
        # Ajustar slider según rango Z de la geometría
        z_vals = triangles[:, :, 2]
        z_min, z_max = np.min(z_vals), np.max(z_vals)
        self.slider_plane.config(from_=float(z_min), to=float(z_max))
        self.slider_plane.set(float(z_min))
        
        self._apply_plane_clipping()

    def _parse_binary_stl(self, stl_bytes):
        """Decodifica el estándar STL binario CAD (80 bytes header + UInt32 count + 50 bytes/triángulo)."""
        if len(stl_bytes) < 84:
            raise ValueError("Buffer STL incompleto.")
        
        num_triangles = struct.unpack("<I", stl_bytes[80:84])[0]
        expected_size = 84 + num_triangles * 50
        if len(stl_bytes) != expected_size:
            print(f"Aviso: Tamaño de STL {len(stl_bytes)} difiere de esperado {expected_size}")
        
        triangles = np.zeros((num_triangles, 3, 3), dtype=np.float32)
        normals = np.zeros((num_triangles, 3), dtype=np.float32)
        
        offset = 84
        for i in range(num_triangles):
            record = struct.unpack("<12fH", stl_bytes[offset:offset + 50])
            normals[i] = record[0:3]
            triangles[i, 0] = record[3:6]
            triangles[i, 1] = record[6:9]
            triangles[i, 2] = record[9:12]
            offset += 50
            
        return triangles, normals

    # =========================================================================
    # Calibración y Recorte Dinámico de Plano de Base (Mesa)
    # =========================================================================
    def _on_slider_plane_change(self, val):
        self.plane_offset_mm = float(val)
        self.lbl_plane_val.config(text=f"Recorte Plano Base (Z): {self.plane_offset_mm:.2f} mm")
        self._apply_plane_clipping()

    def _apply_plane_clipping(self):
        if self.raw_triangles is None or len(self.raw_triangles) == 0:
            return
        
        cutoff_z = self.plane_offset_mm
        # Conservar solo triángulos cuyos 3 vértices estén por encima del plano de corte
        mask = np.all(self.raw_triangles[:, :, 2] > cutoff_z, axis=1)
        self.clipped_triangles = self.raw_triangles[mask]
        
        self.lbl_tri_count.config(text=f"{len(self.clipped_triangles):,} Facetas (Aisladas)")
        self._render_3d_mesh()

    def _auto_detect_table_plane(self):
        """Ajusta un plano de referencia usando RANSAC para aislar la mesa o base plana."""
        if self.raw_triangles is None or len(self.raw_triangles) == 0:
            messagebox.showinfo("Aviso", "Primero captura un modelo 3D.")
            return
        
        # Extraer nube de puntos de los vértices
        pts = self.raw_triangles.reshape(-1, 3)
        if len(pts) < 100:
            return
        
        # RANSAC simple en Python
        best_inliers = 0
        best_plane = None
        threshold = 1.0 # 1 mm de tolerancia
        
        sample_indices = np.random.choice(len(pts), size=(100, 3))
        for idx in sample_indices:
            p1, p2, p3 = pts[idx[0]], pts[idx[1]], pts[idx[2]]
            v1 = p2 - p1
            v2 = p3 - p1
            n = np.cross(v1, v2)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-6:
                continue
            n = n / n_norm
            d = -np.dot(n, p1)
            
            distances = np.abs(np.dot(pts, n) + d)
            inliers = np.sum(distances < threshold)
            if inliers > best_inliers:
                best_inliers = inliers
                best_plane = (n, d)
        
        if best_plane:
            n, d = best_plane
            # Calcular la cota Z promedio de la mesa detectada
            table_z = -d / (n[2] if abs(n[2]) > 1e-4 else 1.0)
            self.slider_plane.set(table_z + 0.5) # +0.5 mm por encima de la mesa
            self._apply_plane_clipping()
            messagebox.showinfo("RANSAC", f"Plano de mesa detectado en Z = {table_z:.2f} mm.\nAislado con margen de +0.5 mm.")

    def _render_3d_mesh(self):
        self.ax.clear()
        self._format_3d_axes()
        
        if self.clipped_triangles is not None and len(self.clipped_triangles) > 0:
            # Submuestreo para fluidez en la interfaz (máximo 8,000 facetas en vista previa interactiva)
            mesh_data = self.clipped_triangles
            if len(mesh_data) > 8000:
                step = len(mesh_data) // 8000
                mesh_data = mesh_data[::step]
            
            collection = Poly3DCollection(mesh_data, facecolors="#0ea5e9", edgecolors="#0369a1", linewidths=0.2, alpha=0.9)
            self.ax.add_collection3d(collection)
            
            # Ajustar límites de visualización
            all_pts = mesh_data.reshape(-1, 3)
            min_lims = np.min(all_pts, axis=0)
            max_lims = np.max(all_pts, axis=0)
            center = (min_lims + max_lims) / 2.0
            max_range = np.max(max_lims - min_lims) / 2.0
            
            if max_range > 0:
                self.ax.set_xlim(center[0] - max_range, center[0] + max_range)
                self.ax.set_ylim(center[1] - max_range, center[1] + max_range)
                self.ax.set_zlim(center[2] - max_range, center[2] + max_range)
        
        self.canvas_3d.draw()

    # =========================================================================
    # Exportación CAD
    # =========================================================================
    def _export_cad_stl(self):
        triangles_to_export = self.clipped_triangles if self.clipped_triangles is not None else self.raw_triangles
        if triangles_to_export is None or len(triangles_to_export) == 0:
            messagebox.showwarning("Aviso", "No hay geometría para exportar.")
            return
        
        path = filedialog.asksaveasfilename(defaultextension=".stl", filetypes=[("Binary CAD STL", "*.stl")], initialfile="Escaneo_TrueDepth_CAD.stl")
        if not path:
            return
        
        # Guardar en estándar binario CAD estricto
        num_triangles = len(triangles_to_export)
        header = b"TrueDepth-3D-Scanner [Binary CAD Metric mm] Manifold-v2.0".ljust(80, b"\0")
        
        with open(path, "wb") as f:
            f.write(header)
            f.write(struct.pack("<I", num_triangles))
            
            for tri in triangles_to_export:
                v1, v2, v3 = tri[0], tri[1], tri[2]
                normal = np.cross(v2 - v1, v3 - v1)
                norm_len = np.linalg.norm(normal)
                if norm_len > 1e-6:
                    normal = normal / norm_len
                else:
                    normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
                
                record = struct.pack("<12fH",
                    normal[0], normal[1], normal[2],
                    v1[0], v1[1], v1[2],
                    v2[0], v2[1], v2[2],
                    v3[0], v3[1], v3[2],
                    0
                )
                f.write(record)
        
        messagebox.showinfo("Exportación Exitosa", f"Archivo STL generado con éxito:\n{os.path.basename(path)}\nListo para abrir en Cura, PrusaSlicer o Fusion 360.")

    def _export_color_ply(self):
        if not self.iphone_ip:
            messagebox.showwarning("Aviso", "Conecta el iPhone para descargar el archivo PLY coloreado.")
            return
        
        path = filedialog.asksaveasfilename(defaultextension=".ply", filetypes=[("Stanford Polygon Format", "*.ply")], initialfile="Escaneo_TrueDepth_Color.ply")
        if not path:
            return
        
        try:
            ply_url = f"http://{self.iphone_ip}:{self.iphone_port}/model.ply?t={time.time()}"
            urllib.request.urlretrieve(ply_url, path)
            messagebox.showinfo("PLY Guardado", f"Archivo PLY con nubes de puntos y color guardado en:\n{path}")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo descargar el PLY: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = TrueDepthStudioApp(root)
    root.mainloop()
