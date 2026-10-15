import Foundation
import Network
import UIKit

class WebStreamServer: ObservableObject {
    @Published var serverURL: String = "Obteniendo IP Wi-Fi..."
    private var listener: NWListener?
    private let port: UInt16 = 8080
    
    var getLatestJPEG: (() -> Data?)?
    var getLatestSTL: (() -> Data?)?
    var onTriggerScan: ((Float, @escaping (Int) -> Void) -> Void)?
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
                } else {
                    self.serverURL = "Conecta el iPhone al Wi-Fi"
                }
            }
        } catch {
            print("Error iniciando servidor: \(error)")
        }
    }
    
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
                if let range = path.range(of: "max=") {
                    let valStr = String(path[range.upperBound...]).components(separatedBy: "&").first ?? "0.50"
                    maxDist = Float(valStr) ?? 0.50
                }
                
                self.onTriggerScan?(maxDist) { triangleCount in
                    let json = "{\"triangles\": \(triangleCount)}".data(using: .utf8) ?? Data()
                    self.sendResponse(connection: connection, contentType: "application/json", body: json)
                }
            } else if path.hasPrefix("/model.stl") {
                if let stlData = self.getLatestSTL?() {
                    let headers = "Content-Disposition: attachment; filename=\"Escaneo_TrueDepth.stl\"\r\n"
                    self.sendResponse(connection: connection, contentType: "application/octet-stream", extraHeaders: headers, body: stlData)
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
            <title>TrueDepth 3D Scanner - Monitor Inalámbrico</title>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
            <style>
                body { margin: 0; font-family: system-ui, sans-serif; background: #0f172a; color: #f8fafc; display: flex; flex-direction: column; height: 100vh; }
                header { padding: 14px 24px; background: #1e293b; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #334155; }
                .main { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; padding: 16px; flex: 1; min-height: 0; }
                .card { background: #1e293b; border-radius: 14px; overflow: hidden; display: flex; flex-direction: column; border: 1px solid #334155; position: relative; }
                .card-title { padding: 10px 16px; background: #0f172a80; font-weight: 600; font-size: 14px; display: flex; justify-content: space-between; }
                #liveView { width: 100%; height: 100%; object-fit: contain; background: #000; flex: 1; }
                #viewer3d { width: 100%; height: 100%; flex: 1; background: #090d16; }
                .controls { padding: 16px 24px; background: #1e293b; border-top: 1px solid #334155; display: flex; gap: 16px; align-items: center; flex-wrap: wrap; }
                button, .btn { border: none; padding: 14px 22px; border-radius: 10px; font-weight: 700; font-size: 15px; cursor: pointer; text-decoration: none; display: inline-flex; align-items: center; gap: 8px; }
                .btn-scan { background: linear-gradient(90deg, #0284c7, #06b6d4); color: white; flex: 1; justify-content: center; }
                .btn-dl { background: #16a34a; color: white; }
                .btn-wa { background: #2563eb; color: white; }
                .slider-box { display: flex; flex-direction: column; gap: 4px; min-width: 220px; }
            </style>
        </head>
        <body>
            <header>
                <div><strong>TrueDepth 3D Scanner</strong> • Monitor Inalámbrico en Tiempo Real</div>
                <div id="status" style="color:#38bdf8; font-size:14px;">Conectado al sensor Face ID</div>
            </header>
            <div class="main">
                <div class="card">
                    <div class="card-title"><span>1. Transmisión en Vivo (Cámara TrueDepth)</span><span>20 - 50 cm</span></div>
                    <img id="liveView" alt="Esperando transmisión..." />
                </div>
                <div class="card">
                    <div class="card-title"><span>2. Vista Previa del Modelo 3D (.STL)</span><span id="triCount">0 triángulos</span></div>
                    <div id="viewer3d"></div>
                </div>
            </div>
            <div class="controls">
                <div class="slider-box">
                    <label style="font-size:13px;">Recorte de fondo: <strong id="distVal">50 cm</strong></label>
                    <input type="range" id="maxDist" min="0.25" max="0.80" step="0.05" value="0.50" oninput="document.getElementById('distVal').innerText = Math.round(this.value*100)+' cm'">
                </div>
                <button class="btn-scan" onclick="triggerScan()">Iniciar escaneo 3D TrueDepth</button>
                <a id="dlBtn" class="btn btn-dl" href="/model.stl" download="Escaneo_TrueDepth.stl" style="display:none;">Descargar .STL en PC</a>
                <button id="waBtn" class="btn btn-wa" onclick="shareOnPhone()" style="display:none;">Compartir en iPhone (WhatsApp / Correo)</button>
            </div>
            <script>
                const img = document.getElementById('liveView');
                function updateFrame() {
                    const next = new Image();
                    next.onload = () => { img.src = next.src; setTimeout(updateFrame, 60); };
                    next.onerror = () => { setTimeout(updateFrame, 300); };
                    next.src = '/frame.jpg?t=' + Date.now();
                }
                updateFrame();

                const container = document.getElementById('viewer3d');
                const scene = new THREE.Scene();
                const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 10);
                camera.position.set(0, 0, 0.45);
                const renderer = new THREE.WebGLRenderer({ antialias: true });
                container.appendChild(renderer.domElement);

                const light1 = new THREE.DirectionalLight(0xffffff, 1);
                light1.position.set(1, 1, 2);
                scene.add(light1);
                scene.add(new THREE.AmbientLight(0x404040, 1.5));

                let meshObj = null;
                let isDragging = false, prevX = 0, prevY = 0;
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
                    document.getElementById('status').innerText = 'Escaneando y generando STL...';
                    const res = await fetch('/scan?max=' + max);
                    const data = await res.json();
                    document.getElementById('triCount').innerText = data.triangles + ' triángulos';
                    if (data.triangles > 0) {
                        document.getElementById('status').innerText = '¡Modelo 3D generado!';
                        document.getElementById('dlBtn').style.display = 'inline-flex';
                        document.getElementById('waBtn').style.display = 'inline-flex';
                        loadSTLPreview();
                    } else {
                        document.getElementById('status').innerText = 'Sin puntos en rango. Acerca más el objeto.';
                    }
                }

                async function loadSTLPreview() {
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
                    const mat = new THREE.MeshPhongMaterial({ color: 0x06b6d4, side: THREE.DoubleSide, flatShading: true });
                    meshObj = new THREE.Mesh(geo, mat);
                    scene.add(meshObj);
                }

                async function shareOnPhone() {
                    await fetch('/share');
                    alert('Se abrió el menú de WhatsApp / Correo en la pantalla de tu iPhone.');
                }
            </script>
        </body>
        </html>
        """
    }
}
