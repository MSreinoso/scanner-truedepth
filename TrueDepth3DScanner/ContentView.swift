import SwiftUI
import AVFoundation

struct CameraPreviewView: UIViewRepresentable {
    let session: AVCaptureSession
    
    func makeUIView(context: Context) -> UIView {
        let view = UIView(frame: .zero)
        let previewLayer = AVCaptureVideoPreviewLayer(session: session)
        previewLayer.videoGravity = .resizeAspectFill
        view.layer.addSublayer(previewLayer)
        return view
    }
    
    func updateUIView(_ uiView: UIView, context: Context) {
        if let layer = uiView.layer.sublayers?.first as? AVCaptureVideoPreviewLayer {
            DispatchQueue.main.async {
                layer.frame = uiView.bounds
            }
        }
    }
}

struct ContentView: View {
    @StateObject private var captureManager = DepthCaptureManager()
    @StateObject private var webServer = WebStreamServer()
    @StateObject private var volumeTrigger = VolumeButtonTriggerManager()
    
    @State private var showPreviewMode = false
    @State private var showWireframe = false
    @State private var showShareSheet = false
    @State private var shareURL: URL?
    
    var body: some View {
        NavigationView {
            VStack(spacing: 0) {
                // Telemetry Header
                HStack(spacing: 8) {
                    Circle()
                        .fill(captureManager.isScanning ? Color.red : (captureManager.isSessionRunning ? Color.green : Color.gray))
                        .frame(width: 10, height: 10)
                        .scaleEffect(captureManager.isScanning ? 1.25 : 1.0)
                        .animation(captureManager.isScanning ? Animation.easeInOut(duration: 0.6).repeatForever() : .default, value: captureManager.isScanning)
                    
                    Text(captureManager.isScanning ? "GRABANDO ESCANEO 3D" : "PC Studio: \(webServer.serverURL)")
                        .font(.caption.monospaced().bold())
                        .foregroundColor(captureManager.isScanning ? .red : .white)
                    
                    Spacer()
                    
                    Text("Auto UDP: 50505")
                        .font(.caption2.monospaced())
                        .foregroundColor(.gray)
                }
                .padding(.horizontal)
                .padding(.vertical, 8)
                .background(Color.black.opacity(0.92))
                .foregroundColor(.white)
                
                // Viewport Mode Switcher
                Picker("Modo", selection: $showPreviewMode) {
                    Text("1. Vista Cámara").tag(false)
                    Text("2. Malla 3D Final").tag(true)
                }
                .pickerStyle(SegmentedPickerStyle())
                .padding(.horizontal)
                .padding(.vertical, 6)
                .background(Color(UIColor.secondarySystemBackground))
                
                // Viewport Area
                ZStack {
                    if !showPreviewMode {
                        CameraPreviewView(session: captureManager.captureSession)
                            .edgesIgnoringSafeArea(.horizontal)
                        
                        // Reticle Overlay
                        RoundedRectangle(cornerRadius: 24)
                            .strokeBorder(
                                style: StrokeStyle(lineWidth: captureManager.isScanning ? 3 : 2, dash: captureManager.isScanning ? [] : [10])
                            )
                            .foregroundColor(captureManager.isScanning ? .red : .cyan.opacity(0.85))
                            .padding(36)
                        
                        // Top Scanning Stats HUD
                        VStack {
                            if captureManager.isScanning || captureManager.pointCount > 0 {
                                HStack(spacing: 16) {
                                    VStack(alignment: .leading, spacing: 2) {
                                        Text("VÉRTICES ACUMULADOS")
                                            .font(.system(size: 9, weight: .bold))
                                            .foregroundColor(.gray)
                                        Text("\(captureManager.pointCount.formatted())")
                                            .font(.system(size: 20, weight: .heavy, design: .monospaced))
                                            .foregroundColor(.cyan)
                                    }
                                    
                                    Divider()
                                        .frame(height: 24)
                                        .background(Color.white.opacity(0.3))
                                    
                                    VStack(alignment: .leading, spacing: 2) {
                                        Text("FACETAS CAD")
                                            .font(.system(size: 9, weight: .bold))
                                            .foregroundColor(.gray)
                                        Text("\(captureManager.triangleCount.formatted())")
                                            .font(.system(size: 20, weight: .heavy, design: .monospaced))
                                            .foregroundColor(.green)
                                    }
                                }
                                .padding(.horizontal, 16)
                                .padding(.vertical, 8)
                                .background(.ultraThinMaterial)
                                .cornerRadius(14)
                                .padding(.top, 14)
                            }
                            
                            Spacer()
                            
                            // Bottom Guidance Badge
                            HStack(spacing: 8) {
                                Image(systemName: "speaker.wave.3.fill")
                                    .foregroundColor(.yellow)
                                Text("Doble clic en Botón de Volumen para \(captureManager.isScanning ? "DETENER" : "INICIAR") Escaneo")
                                    .font(.caption.bold())
                            }
                            .padding(.horizontal, 14)
                            .padding(.vertical, 8)
                            .background(.ultraThinMaterial)
                            .cornerRadius(20)
                            .padding(.bottom, 16)
                        }
                    } else {
                        SceneKit3DView(mesh: captureManager.latestMesh, showWireframe: showWireframe)
                            .edgesIgnoringSafeArea(.horizontal)
                        
                        if let mesh = captureManager.latestMesh {
                            VStack {
                                HStack {
                                    Label("\(mesh.triangles.count) triángulos", systemImage: "cube.fill")
                                        .font(.caption.bold())
                                        .padding(8)
                                        .background(.ultraThinMaterial)
                                        .cornerRadius(8)
                                    
                                    Spacer()
                                    
                                    Button(action: { showWireframe.toggle() }) {
                                        Label(showWireframe ? "Sólido" : "Alámbrico", systemImage: showWireframe ? "cube" : "square.grid.3x3")
                                            .font(.caption.bold())
                                            .padding(8)
                                            .background(.ultraThinMaterial)
                                            .cornerRadius(8)
                                    }
                                }
                                .padding()
                                
                                Spacer()
                                
                                Text("Girar con 1 dedo • Zoom con pellizco")
                                    .font(.caption2)
                                    .padding(.horizontal, 12)
                                    .padding(.vertical, 6)
                                    .background(.ultraThinMaterial)
                                    .cornerRadius(12)
                                    .padding(.bottom, 12)
                            }
                        } else {
                            VStack(spacing: 12) {
                                Image(systemName: "cube.transparent")
                                    .font(.system(size: 56))
                                    .foregroundColor(.gray)
                                Text("Aún no has completado un escaneo.\nInicia el escaneo continuo y rodéalo con la cámara.")
                                    .font(.subheadline)
                                    .foregroundColor(.gray)
                                    .multilineTextAlignment(.center)
                                    .padding(.horizontal, 32)
                            }
                        }
                    }
                }
                .frame(maxHeight: .infinity)
                
                // Bottom Control Panel
                VStack(spacing: 12) {
                    // Parameter Sliders
                    HStack {
                        Toggle(isOn: $captureManager.enableSOR) {
                            Text("Filtro SOR")
                                .font(.caption.bold())
                        }
                        .toggleStyle(SwitchToggleStyle(tint: .cyan))
                        
                        Spacer()
                        
                        Text("Rango Máximo: \(Int(captureManager.maxDepthMeters * 100)) cm")
                            .font(.caption.bold())
                            .foregroundColor(.cyan)
                    }
                    
                    Slider(value: $captureManager.maxDepthMeters, in: 0.20...0.70, step: 0.05)
                        .accentColor(.cyan)
                    
                    // Main Action Trigger Button (Continuous Scanner)
                    Button(action: {
                        captureManager.toggleScan()
                    }) {
                        HStack(spacing: 12) {
                            Image(systemName: captureManager.isScanning ? "stop.circle.fill" : "play.circle.fill")
                                .font(.title2)
                            Text(captureManager.isScanning ? "DETENER ESCANEO Y GENERAR STL" : "INICIAR ESCANEO CONTINUO (O 2x VOL+)")
                                .font(.headline.bold())
                        }
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 15)
                        .background(
                            captureManager.isScanning ?
                            LinearGradient(colors: [Color.red, Color.orange], startPoint: .leading, endPoint: .trailing) :
                            LinearGradient(colors: [Color.blue, Color.cyan], startPoint: .leading, endPoint: .trailing)
                        )
                        .foregroundColor(.white)
                        .cornerRadius(14)
                    }
                    
                    // Secondary Actions
                    HStack(spacing: 10) {
                        Button(action: {
                            captureManager.clearScan()
                            showPreviewMode = false
                        }) {
                            Label("Limpiar", systemImage: "trash")
                                .font(.subheadline.bold())
                                .frame(maxWidth: .infinity)
                                .padding(.vertical, 10)
                                .background(Color(UIColor.tertiarySystemFill))
                                .foregroundColor(.primary)
                                .cornerRadius(10)
                        }
                        
                        if let stlUrl = captureManager.latestSTLURL {
                            Button(action: {
                                self.shareURL = stlUrl
                                self.showShareSheet = true
                            }) {
                                Label("Compartir .STL", systemImage: "square.and.arrow.up.fill")
                                    .font(.subheadline.bold())
                                    .frame(maxWidth: .infinity)
                                    .padding(.vertical, 10)
                                    .background(Color.green)
                                    .foregroundColor(.white)
                                    .cornerRadius(10)
                            }
                        }
                        
                        if let plyUrl = captureManager.latestPLYURL {
                            Button(action: {
                                self.shareURL = plyUrl
                                self.showShareSheet = true
                            }) {
                                Label("PLY Color", systemImage: "paintpalette.fill")
                                    .font(.subheadline.bold())
                                    .frame(maxWidth: .infinity)
                                    .padding(.vertical, 10)
                                    .background(Color.purple)
                                    .foregroundColor(.white)
                                    .cornerRadius(10)
                            }
                        }
                    }
                }
                .padding()
                .background(Color(UIColor.secondarySystemBackground))
            }
            .navigationTitle("Bichito 3D Pro")
            .navigationBarTitleDisplayMode(.inline)
            .sheet(isPresented: $showShareSheet) {
                if let url = shareURL {
                    ShareSheet(activityItems: [url])
                }
            }
            .onAppear {
                UIApplication.shared.isIdleTimerDisabled = true
                setupNetworking()
                volumeTrigger.onTriggerAction = {
                    captureManager.toggleScan()
                }
                webServer.start()
            }
        }
    }
    
    private func setupNetworking() {
        webServer.getLatestJPEG = { captureManager.latestJPEGData }
        webServer.getLatestSTL = { captureManager.latestSTLData }
        webServer.getLatestPLY = { captureManager.latestPLYData }
        webServer.getLatestNewPoints = { captureManager.latestNewPointsJSON }
        webServer.getStatusJSON = {
            let status = captureManager.statusMessage.replacingOccurrences(of: "\"", with: "'")
            return "{\"isScanning\": \(captureManager.isScanning), \"points\": \(captureManager.pointCount), \"triangles\": \(captureManager.triangleCount), \"status\": \"\(status)\"}"
        }
        webServer.onToggleScan = {
            captureManager.toggleScan()
        }
        webServer.onClearScan = {
            captureManager.clearScan()
        }
        webServer.onTriggerShareOnPhone = {
            if let stlUrl = captureManager.latestSTLURL {
                self.shareURL = stlUrl
                self.showShareSheet = true
            }
        }
    }
}

struct ShareSheet: UIViewControllerRepresentable {
    var activityItems: [Any]
    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: activityItems, applicationActivities: nil)
    }
    func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) {}
}
