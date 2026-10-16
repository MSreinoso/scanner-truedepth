import Foundation
import Network
import UIKit

/// High-throughput HTTP & UDP server running natively on iOS.
/// Provides real-time camera streaming, telemetry, CAD STL/PLY endpoints,
/// and automated UDP beaconing (port 50505) for zero-config PC pairing.
class WebStreamServer: ObservableObject {
    @Published var serverURL: String = "Obteniendo IP Wi-Fi..."
    private var listener: NWListener?
    private let port: UInt16 = 8080
    private let udpBeaconPort: UInt16 = 50505
    private var udpTimer: Timer?
    
    var getLatestJPEG: (() -> Data?)?
    var getLatestSTL: (() -> Data?)?
    var getLatestPLY: (() -> Data?)?
    var onTriggerScan: ((Float, Bool, Float?, @escaping (Int) -> Void) -> Void)?
    var onTriggerShareOnPhone: (() -> Void)?
    
    func start() {
        do {
            let parameters = NWParameters.tcp
            listener = try NWListener(using: parameters, on: NWEndpoint.Port(rawValue: port)!)
            
            listener?.newConnectionHandler = { [weak self] connection in
                self?.handleConnection(connection)
            }
            
            listener?.start(queue: .global(qos: .userInteractive))
            
            DispatchQueue.main.async {
                if let ip = self.getWiFiAddress() {
                    self.serverURL = "http://\(ip):\(self.port)"
                    self.startUDPBeacon(ip: ip)
                } else {
                    self.serverURL = "Conecta el iPhone a la misma red Wi-Fi"
                }
            }
        } catch {
            print("Error iniciando servidor: \(error)")
        }
    }
    
    // MARK: - UDP Auto-Discovery Beacon (Port 50505)
    private func startUDPBeacon(ip: String) {
        udpTimer?.invalidate()
        let beaconMessage = "TRUEDEPTH_3D_BEACON|IP:\(ip)|PORT:\(port)|READY"
        
        udpTimer = Timer.scheduledTimer(withTimeInterval: 2.0, repeats: true) { [weak self] _ in
            guard let self = self, let beaconData = beaconMessage.data(using: .utf8) else { return }
            
            let host = NWEndpoint.Host("255.255.255.255")
            let beaconPort = NWEndpoint.Port(rawValue: self.udpBeaconPort)!
            let connection = NWConnection(host: host, port: beaconPort, using: .udp)
            
            connection.start(queue: .global(qos: .background))
            connection.send(content: beaconData, completion: .contentProcessed { _ in
                connection.cancel()
            })
        }
    }
    
    // MARK: - TCP HTTP Connection Handler
    private func handleConnection(_ connection: NWConnection) {
        connection.start(queue: .global(qos: .userInteractive))
        connection.receive(minimumIncompleteLength: 1, maximumLength: 65536) { [weak self] data, _, _, _ in
            guard let self = self,
                  let data = data,
                  let request = String(data: data, encoding: .utf8) else {
                connection.cancel()
                return
            }
            
            let firstLine = request.components(separatedBy: "\r\n").first ?? ""
            let parts = firstLine.components(separatedBy: " ")
            let path = parts.count > 1 ? parts[1] : "/"
            
            if path.hasPrefix("/frame.jpg") {
                if let jpeg = self.getLatestJPEG?() {
                    self.sendResponse(connection: connection, contentType: "image/jpeg", body: jpeg)
                } else {
                    self.sendResponse(connection: connection, status: "204 No Content", contentType: "text/plain", body: Data())
                }
            } else if path.hasPrefix("/scan") {
                var maxDist: Float = 0.50
                var enableSOR: Bool = true
                var planeCutoff: Float? = nil
                
                // Parse Query Parameters
                if let range = path.range(of: "max=") {
                    let valStr = String(path[range.upperBound...]).components(separatedBy: "&").first ?? "0.50"
                    maxDist = Float(valStr) ?? 0.50
                }
                if let range = path.range(of: "sor=") {
                    let valStr = String(path[range.upperBound...]).components(separatedBy: "&").first ?? "1"
                    enableSOR = (valStr == "1" || valStr.lowercased() == "true")
                }
                if let range = path.range(of: "plane=") {
                    let valStr = String(path[range.upperBound...]).components(separatedBy: "&").first ?? ""
                    if let f = Float(valStr), f > 0 {
                        planeCutoff = f
                    }
                }
                
                self.onTriggerScan?(maxDist, enableSOR, planeCutoff) { triangleCount in
                    let json = "{\"status\": \"success\", \"triangles\": \(triangleCount)}".data(using: .utf8) ?? Data()
                    self.sendResponse(connection: connection, contentType: "application/json", body: json)
                }
            } else if path.hasPrefix("/model.stl") {
                if let stlData = self.getLatestSTL?() {
                    let headers = "Content-Disposition: attachment; filename=\"TrueDepth_Scan_CAD.stl\"\r\n"
                    self.sendResponse(connection: connection, contentType: "application/octet-stream", extraHeaders: headers, body: stlData)
                } else {
                    self.sendResponse(connection: connection, status: "404 Not Found", contentType: "text/plain", body: Data())
                }
            } else if path.hasPrefix("/model.ply") {
                if let plyData = self.getLatestPLY?() {
                    let headers = "Content-Disposition: attachment; filename=\"TrueDepth_ColorScan.ply\"\r\n"
                    self.sendResponse(connection: connection, contentType: "application/octet-stream", extraHeaders: headers, body: plyData)
                } else {
                    self.sendResponse(connection: connection, status: "404 Not Found", contentType: "text/plain", body: Data())
                }
            } else if path.hasPrefix("/share") {
                DispatchQueue.main.async {
                    self.onTriggerShareOnPhone?()
                }
                let json = "{\"shared\": true}".data(using: .utf8) ?? Data()
                self.sendResponse(connection: connection, contentType: "application/json", body: json)
            } else {
                let html = self.dashboardHTML().data(using: .utf8) ?? Data()
                self.sendResponse(connection: connection, contentType: "text/html; charset=utf-8", body: html)
            }
        }
    }
    
    private func sendResponse(connection: NWConnection, status: String = "200 OK", contentType: String, extraHeaders: String = "", body: Data) {
        var header = "HTTP/1.1 \(status)\r\n"
        header += "Content-Type: \(contentType)\r\n"
        header += "Content-Length: \(body.count)\r\n"
        header += "Cache-Control: no-cache, no-store, must-revalidate\r\n"
        header += "Access-Control-Allow-Origin: *\r\n"
        header += extraHeaders
        header += "Connection: close\r\n\r\n"
        
        var responseData = header.data(using: .utf8) ?? Data()
        responseData.append(body)
        
        connection.send(content: responseData, completion: .contentProcessed { _ in
            connection.cancel()
        })
    }
    
    private func getWiFiAddress() -> String? {
        var address: String?
        var ifaddr: UnsafeMutablePointer<ifaddrs>?
        if getifaddrs(&ifaddr) == 0 {
            var ptr = ifaddr
            while ptr != nil {
                defer { ptr = ptr?.pointee.ifa_next }
                guard let interface = ptr?.pointee else { continue }
                let addrFamily = interface.ifa_addr.pointee.sa_family
                if addrFamily == UInt8(AF_INET) {
                    let name = String(cString: interface.ifa_name)
                    if name == "en0" {
                        var hostname = [CChar](repeating: 0, count: Int(NI_MAXHOST))
                        getnameinfo(interface.ifa_addr, socklen_t(interface.ifa_addr.pointee.sa_len),
                                     &hostname, socklen_t(hostname.count),
                                     nil, socklen_t(0), NI_NUMERICHOST)
                        address = String(cString: hostname)
                    }
                }
            }
            freeifaddrs(ifaddr)
        }
        return address
    }
    
    private func dashboardHTML() -> String {
        return """
        <!DOCTYPE html>
        <html lang="es">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>TrueDepth 3D Scanner - Grado Industrial</title>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
            <style>
                body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0b0f19; color: #f1f5f9; display: flex; flex-direction: column; height: 100vh; }
                header { padding: 12px 24px; background: #111827; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #1f2937; }
                .main { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; padding: 16px; flex: 1; min-height: 0; }
                .card { background: #111827; border-radius: 12px; overflow: hidden; display: flex; flex-direction: column; border: 1px solid #1f2937; }
                .card-title { padding: 10px 16px; background: #1e293b; font-weight: 600; font-size: 13px; display: flex; justify-content: space-between; border-bottom: 1px solid #334155; }
                #liveView { width: 100%; height: 100%; object-fit: contain; background: #000; flex: 1; }
                #viewer3d { width: 100%; height: 100%; flex: 1; background: #030712; }
                .controls { padding: 14px 24px; background: #111827; border-top: 1px solid #1f2937; display: flex; gap: 14px; align-items: center; flex-wrap: wrap; }
                button, .btn { border: none; padding: 12px 20px; border-radius: 8px; font-weight: 600; font-size: 14px; cursor: pointer; text-decoration: none; display: inline-flex; align-items: center; gap: 8px; }
                .btn-scan { background: linear-gradient(90deg, #2563eb, #06b6d4); color: white; flex: 1; justify-content: center; }
                .btn-stl { background: #16a34a; color: white; }
                .btn-ply { background: #9333ea; color: white; }
                .slider-box { display: flex; flex-direction: column; gap: 4px; min-width: 180px; font-size: 13px; }
            </style>
        </head>
        <body>
            <header>
                <div><strong>TrueDepth Industrial Scanner</strong> • Sincronización Sub-milimétrica</div>
                <div id="status" style="color:#38bdf8; font-size:13px;">Sensor Activo</div>
            </header>
            <div class="main">
                <div class="card">
                    <div class="card-title"><span>Video Sincronizado TrueDepth</span><span>30 FPS</span></div>
                    <img id="liveView" alt="Cargando stream..." />
                </div>
                <div class="card">
                    <div class="card-title"><span>Visualizador CAD 3D</span><span id="triCount">0 Facetas</span></div>
                    <div id="viewer3d"></div>
                </div>
            </div>
            <div class="controls">
                <div class="slider-box">
                    <label>Fondo Z Máximo: <strong id="distVal">50 cm</strong></label>
                    <input type="range" id="maxDist" min="0.20" max="0.70" step="0.05" value="0.50" oninput="document.getElementById('distVal').innerText = Math.round(this.value*100)+' cm'">
                </div>
                <button class="btn-scan" onclick="triggerScan()">Disparar Escaneo 3D</button>
                <a id="stlBtn" class="btn btn-stl" href="/model.stl" download="TrueDepth_CAD.stl" style="display:none;">Descargar STL CAD</a>
                <a id="plyBtn" class="btn btn-ply" href="/model.ply" download="TrueDepth_Color.ply" style="display:none;">Descargar PLY Color</a>
            </div>
            <script>
                const img = document.getElementById('liveView');
                function updateFrame() {
                    const next = new Image();
                    next.onload = () => { img.src = next.src; setTimeout(updateFrame, 50); };
                    next.onerror = () => { setTimeout(updateFrame, 250); };
                    next.src = '/frame.jpg?t=' + Date.now();
                }
                updateFrame();

                const container = document.getElementById('viewer3d');
                const scene = new THREE.Scene();
                const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 10);
                camera.position.set(0, 0, 0.40);
                const renderer = new THREE.WebGLRenderer({ antialias: true });
                container.appendChild(renderer.domElement);
                scene.add(new THREE.AmbientLight(0xffffff, 0.8));
                const dirLight = new THREE.DirectionalLight(0xffffff, 1.0);
                dirLight.position.set(1, 2, 3);
                scene.add(dirLight);

                let meshObj = null, isDragging = false, prevX = 0, prevY = 0;
                container.addEventListener('mousedown', e => { isDragging = true; prevX = e.clientX; prevY = e.clientY; });
                window.addEventListener('mouseup', () => isDragging = false);
                window.addEventListener('mousemove', e => {
                    if (!isDragging || !meshObj) return;
                    meshObj.rotation.y += (e.clientX - prevX) * 0.01;
                    meshObj.rotation.x += (e.clientY - prevY) * 0.01;
                    prevX = e.clientX; prevY = e.clientY;
                });
                container.addEventListener('wheel', e => {
                    camera.position.z = Math.max(0.1, Math.min(2.0, camera.position.z + e.deltaY * 0.0005));
                });

                function animate() {
                    requestAnimationFrame(animate);
                    const w = container.clientWidth, h = container.clientHeight;
                    if (renderer.domElement.width !== w || renderer.domElement.height !== h) {
                        renderer.setSize(w, h, false);
                        camera.aspect = w / h;
                        camera.updateProjectionMatrix();
                    }
                    renderer.render(scene, camera);
                }
                animate();

                async function triggerScan() {
                    const max = document.getElementById('maxDist').value;
                    document.getElementById('status').innerText = 'Reconstruyendo con filtro SOR...';
                    const res = await fetch('/scan?max=' + max + '&sor=1');
                    const data = await res.json();
                    document.getElementById('triCount').innerText = data.triangles + ' facetas';
                    if (data.triangles > 0) {
                        document.getElementById('status').innerText = 'Geometría lista';
                        document.getElementById('stlBtn').style.display = 'inline-flex';
                        document.getElementById('plyBtn').style.display = 'inline-flex';
                        loadSTL();
                    }
                }

                async function loadSTL() {
                    const res = await fetch('/model.stl?t=' + Date.now());
                    const buf = await res.arrayBuffer();
                    const view = new DataView(buf);
                    const numTriangles = view.getUint32(80, true);
                    const positions = new Float32Array(numTriangles * 9);
                    let offset = 84;
                    for (let i = 0; i < numTriangles; i++) {
                        offset += 12;
                        for (let j = 0; j < 9; j++) {
                            positions[i * 9 + j] = view.getFloat32(offset, true) / 1000.0;
                            offset += 4;
                        }
                        offset += 2;
                    }
                    const geo = new THREE.BufferGeometry();
                    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
                    geo.computeVertexNormals();
                    geo.center();
                    if (meshObj) scene.remove(meshObj);
                    const mat = new THREE.MeshStandardMaterial({ color: 0x38bdf8, roughness: 0.4, metalness: 0.2, side: THREE.DoubleSide });
                    meshObj = new THREE.Mesh(geo, mat);
                    scene.add(meshObj);
                }
            </script>
        </body>
        </html>
        """
    }
}
