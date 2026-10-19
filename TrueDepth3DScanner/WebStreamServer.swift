import Foundation
import Combine
import Network
import UIKit

/// HTTP preview, exports and manual TCP pairing running natively on iOS.
/// Handles real-time video streaming, continuous 3D telemetry, CAD STL/PLY model downloads,
/// and a /connect endpoint that pairs with the requesting PC.
class WebStreamServer: ObservableObject {
    @Published var serverURL: String = "Obteniendo IP Wi-Fi..."
    private var listener: NWListener?
    private let port: UInt16 = 8080
    
    var getLatestJPEG: (() -> Data?)?
    var getLatestSTL: (() -> Data?)?
    var getLatestPLY: (() -> Data?)?
    var getLatestNewPoints: (() -> Data?)?
    var getStatusJSON: (() -> String)?
    var onToggleScan: (() -> Void)?
    var onStartScan: (() -> Void)?
    var onStopScan: (() -> Void)?
    var onConnectToPC: ((String) -> Void)?
    var onClearScan: (() -> Void)?
    var onTriggerShareOnPhone: (() -> Void)?
    
    func start() {
        guard listener == nil else { return }
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
                    self.serverURL = "Conecta el iPhone a la misma red Wi-Fi"
                }
            }
        } catch {
            print("Error iniciando servidor WebStreamServer: \(error)")
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
            
            if path == "/connect" {
                guard case let .hostPort(host, _) = connection.endpoint else {
                    connection.cancel()
                    return
                }
                self.onConnectToPC?("\(host)")
                self.sendResponse(connection: connection, status: "202 Accepted",
                                  contentType: "application/json",
                                  body: Data("{\"status\":\"connecting\"}".utf8))
            } else if path.hasPrefix("/frame.jpg") {
                if let jpeg = self.getLatestJPEG?() {
                    self.sendResponse(connection: connection, contentType: "image/jpeg", body: jpeg)
                } else {
                    self.sendResponse(connection: connection, status: "204 No Content", contentType: "text/plain", body: Data())
                }
            } else if path == "/start" || path == "/stop" {
                DispatchQueue.main.async {
                    if path == "/start" { self.onStartScan?() } else { self.onStopScan?() }
                }
                self.sendResponse(connection: connection, status: "202 Accepted", contentType: "application/json", body: Data("{\"accepted\":true}".utf8))
            } else if path.hasPrefix("/toggle") {
                DispatchQueue.main.async {
                    self.onToggleScan?()
                }
                let resp = self.getStatusJSON?().data(using: .utf8) ?? "{\"status\": \"toggled\"}".data(using: .utf8)!
                self.sendResponse(connection: connection, contentType: "application/json", body: resp)
            } else if path.hasPrefix("/clear") {
                DispatchQueue.main.async {
                    self.onClearScan?()
                }
                let resp = "{\"status\": \"cleared\"}".data(using: .utf8)!
                self.sendResponse(connection: connection, contentType: "application/json", body: resp)
            } else if path.hasPrefix("/status") {
                let status = self.getStatusJSON?() ?? "{\"isScanning\": false, \"points\": 0}"
                let resp = status.data(using: .utf8) ?? Data()
                self.sendResponse(connection: connection, contentType: "application/json", body: resp)
            } else if path.hasPrefix("/live_points") {
                if let ptsData = self.getLatestNewPoints?() {
                    self.sendResponse(connection: connection, contentType: "application/json", body: ptsData)
                } else {
                    self.sendResponse(connection: connection, contentType: "application/json", body: "[]".data(using: .utf8)!)
                }
            } else if path.hasPrefix("/model.stl") || path.hasPrefix("/download.stl") {
                if let stlData = self.getLatestSTL?() {
                    let headers = "Content-Disposition: attachment; filename=\"TrueDepth_Scan_CAD.stl\"\r\n"
                    self.sendResponse(connection: connection, contentType: "application/sla", extraHeaders: headers, body: stlData)
                } else {
                    self.sendResponse(connection: connection, status: "404 Not Found", contentType: "text/plain", body: "No hay modelo STL generado aún. Realiza un escaneo.".data(using: .utf8)!)
                }
            } else if path.hasPrefix("/model.ply") || path.hasPrefix("/download.ply") {
                if let plyData = self.getLatestPLY?() {
                    let headers = "Content-Disposition: attachment; filename=\"TrueDepth_ColorScan.ply\"\r\n"
                    self.sendResponse(connection: connection, contentType: "application/octet-stream", extraHeaders: headers, body: plyData)
                } else {
                    self.sendResponse(connection: connection, status: "404 Not Found", contentType: "text/plain", body: "No hay modelo PLY generado aún.".data(using: .utf8)!)
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
                guard let addr = interface.ifa_addr else { continue }
                let addrFamily = addr.pointee.sa_family
                if addrFamily == UInt8(AF_INET) {
                    let name = String(cString: interface.ifa_name)
                    if name == "en0" || name == "bridge100" {
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
            <title>Bichito 3D Pro - Monitor Web</title>
            <style>
                body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0b0f19; color: #f1f5f9; display: flex; flex-direction: column; height: 100vh; }
                header { padding: 12px 24px; background: #111827; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #1f2937; }
                .main { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; padding: 16px; flex: 1; min-height: 0; }
                .card { background: #111827; border-radius: 12px; overflow: hidden; display: flex; flex-direction: column; border: 1px solid #1f2937; }
                .card-title { padding: 10px 16px; background: #1e293b; font-weight: 600; font-size: 13px; display: flex; justify-content: space-between; border-bottom: 1px solid #334155; }
                #liveView { width: 100%; height: 100%; object-fit: contain; background: #000; flex: 1; }
                .stats-box { padding: 24px; display: flex; flex-direction: column; justify-content: center; align-items: center; gap: 14px; flex: 1; }
                .metric { font-size: 42px; font-weight: 800; color: #38bdf8; font-family: monospace; }
                .metric-label { font-size: 14px; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; }
                .controls { padding: 14px 24px; background: #111827; border-top: 1px solid #1f2937; display: flex; gap: 14px; align-items: center; }
                button, .btn { border: none; padding: 12px 24px; border-radius: 8px; font-weight: 700; font-size: 15px; cursor: pointer; text-decoration: none; display: inline-flex; align-items: center; gap: 8px; }
                .btn-toggle { background: #0284c7; color: white; flex: 1; justify-content: center; }
                .btn-clear { background: #374151; color: white; }
                .btn-stl { background: #16a34a; color: white; }
            </style>
        </head>
        <body>
            <header>
                <div><strong>Bichito 3D Pro</strong> • Escáner Continuo TrueDepth</div>
                <div id="statusText" style="color:#10b981; font-weight: bold;">Sensor Activo</div>
            </header>
            <div class="main">
                <div class="card">
                    <div class="card-title"><span>Video TrueDepth en Vivo</span><span>30 FPS</span></div>
                    <img id="liveView" alt="Cargando vídeo..." />
                </div>
                <div class="card">
                    <div class="card-title"><span>Telemetría de Vértices Acumulados</span><span id="scanStatus">En Espera</span></div>
                    <div class="stats-box">
                        <div class="metric" id="ptsCount">0</div>
                        <div class="metric-label">Vértices 3D Registrados en Malla</div>
                        <div style="font-size: 13px; color: #64748b; text-align: center;">Pulsa el botón o haz doble clic en volumen para Iniciar / Detener el escaneo continuo.</div>
                    </div>
                </div>
            </div>
            <div class="controls">
                <button id="toggleBtn" class="btn-toggle" onclick="toggleScan()">Iniciar Escaneo Continuo</button>
                <button class="btn-clear" onclick="clearScan()">Limpiar</button>
                <a id="stlBtn" class="btn btn-stl" href="/model.stl" download="TrueDepth_CAD.stl">Descargar STL</a>
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

                async function pollStatus() {
                    try {
                        const res = await fetch('/status');
                        const data = await res.json();
                        document.getElementById('ptsCount').innerText = data.points.toLocaleString();
                        const isScanning = data.isScanning;
                        const btn = document.getElementById('toggleBtn');
                        const statusLabel = document.getElementById('scanStatus');
                        if (isScanning) {
                            btn.innerText = 'DETENER ESCANEO Y GENERAR STL';
                            btn.style.background = '#ef4444';
                            statusLabel.innerText = '● Grabando en vivo';
                            statusLabel.style.color = '#ef4444';
                        } else {
                            btn.innerText = 'INICIAR ESCANEO CONTINUO';
                            btn.style.background = '#0284c7';
                            statusLabel.innerText = data.points > 0 ? 'Malla Lista' : 'En Espera';
                            statusLabel.style.color = '#10b981';
                        }
                    } catch(e) {}
                    setTimeout(pollStatus, 500);
                }
                pollStatus();

                async function toggleScan() {
                    await fetch('/toggle');
                }
                async function clearScan() {
                    await fetch('/clear');
                }
            </script>
        </body>
        </html>
        """
    }
}
