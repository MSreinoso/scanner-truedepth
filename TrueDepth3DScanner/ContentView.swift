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
    
    @State private var capturedMesh: Mesh3D?
    @State private var exportedSTLURL: URL?
    @State private var exportedPLYURL: URL?
    @State private var latestSTLData: Data?
    @State private var latestPLYData: Data?
    
    @State private var showShareSheet = false
    @State private var isProcessing = false
    @State private var showPreviewMode = false
    @State private var showWireframe = false
    @State private var enableSORFilter = true
    @State private var errorMessage: String?
    
    @State private var minDistance: Float = 0.15
    @State private var maxDistance: Float = 0.50
    @State private var basePlaneCutoff: Float = 0.0 // 0 = disabled
    @State private var meshResolution: Int = 2
    
    var body: some View {
        NavigationView {
            VStack(spacing: 0) {
                // Telemetry Header
                HStack {
                    Circle()
                        .fill(captureManager.isSessionRunning ? Color.green : Color.red)
                        .frame(width: 8, height: 8)
                    Text("PC Studio: \(webServer.serverURL)")
                        .font(.caption.monospaced().bold())
                    Spacer()
                    Text("Auto UDP: 50505")
                        .font(.caption2.monospaced())
                        .foregroundColor(.gray)
                }
                .padding(.horizontal)
                .padding(.vertical, 8)
                .background(Color.black.opacity(0.90))
                .foregroundColor(.white)
                
                // Mode Selector
                Picker("Modo", selection: $showPreviewMode) {
                    Text("1. Vista Cámara").tag(false)
                    Text("2. Malla CAD 3D").tag(true)
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
                            .strokeBorder(style: StrokeStyle(lineWidth: 2, dash: [10]))
                            .foregroundColor(.cyan.opacity(0.85))
                            .padding(40)
                        
                        VStack {
                            Spacer()
                            HStack(spacing: 8) {
                                Image(systemName: "speaker.wave.3.fill")
                                    .foregroundColor(.yellow)
                                Text("Doble clic en Botón de Volumen para Disparar")
                                    .font(.caption.bold())
                            }
                            .padding(.horizontal, 14)
                            .padding(.vertical, 8)
                            .background(.ultraThinMaterial)
                            .cornerRadius(20)
                            .padding(.bottom, 16)
                        }
                    } else {
                        SceneKit3DView(mesh: capturedMesh, showWireframe: showWireframe)
                            .edgesIgnoringSafeArea(.horizontal)
                        
                        if let mesh = capturedMesh {
                            VStack {
                                HStack {
                                    Label("\(mesh.triangles.count) facetas", systemImage: "shippingbox.fill")
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
                                Text("Sin escaneo activo.\nPresiona el botón de abajo o doble clic en Subir Volumen.")
                                    .font(.subheadline)
                                    .foregroundColor(.gray)
                                    .multilineTextAlignment(.center)
                                    .padding(.horizontal, 32)
                            }
                        }
                    }
                    
                    if isProcessing {
                        VStack(spacing: 14) {
                            ProgressView()
                                .scaleEffect(1.4)
                            Text("Filtrado SOR & Reconstrucción...")
                                .font(.subheadline.bold())
                        }
                        .padding(26)
                        .background(.ultraThinMaterial)
                        .cornerRadius(18)
                    }
                }
                .frame(maxHeight: .infinity)
                
                // Bottom Parameter Controls
                VStack(spacing: 12) {
                    HStack {
                        Toggle(isOn: $enableSORFilter) {
                            Text("Filtro Estadístico SOR")
                                .font(.caption.bold())
                        }
                        .toggleStyle(SwitchToggleStyle(tint: .cyan))
                        
                        Spacer()
                        
                        Text("Profundidad: \(Int(maxDistance * 100)) cm")
                            .font(.caption.bold())
                            .foregroundColor(.cyan)
                    }
                    
                    Slider(value: $maxDistance, in: 0.20...0.70, step: 0.05)
                        .accentColor(.cyan)
                    
                    Button(action: startTrueDepthScan) {
                        HStack(spacing: 10) {
                            Image(systemName: "camera.metering.matrix")
                                .font(.title3)
                            Text("Capturar Modelo 3D Sub-milimétrico")
                                .font(.headline.bold())
                        }
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 14)
                        .background(
                            LinearGradient(
                                colors: [Color.blue, Color.cyan],
                                startPoint: .leading,
                                endPoint: .trailing
                            )
                        )
                        .foregroundColor(.white)
                        .cornerRadius(12)
                    }
                    .disabled(isProcessing || !captureManager.isSessionRunning)
                    
                    if capturedMesh != nil {
                        HStack(spacing: 10) {
                            Button(action: { showPreviewMode = false }) {
                                Label("Repetir", systemImage: "arrow.counterclockwise")
                                    .font(.caption.bold())
                                    .frame(maxWidth: .infinity)
                                    .padding(.vertical, 10)
                                    .background(Color(UIColor.tertiarySystemFill))
                                    .foregroundColor(.primary)
                                    .cornerRadius(10)
                            }
                            
                            Button(action: generateAndShareSTL) {
                                Label("STL CAD", systemImage: "square.and.arrow.up")
                                    .font(.caption.bold())
                                    .frame(maxWidth: .infinity)
                                    .padding(.vertical, 10)
                                    .background(Color.green)
                                    .foregroundColor(.white)
                                    .cornerRadius(10)
                            }
                            
                            Button(action: generateAndSharePLY) {
                                Label("PLY Color", systemImage: "paintpalette.fill")
                                    .font(.caption.bold())
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
            .navigationTitle("Escáner 3D TrueDepth")
            .navigationBarTitleDisplayMode(.inline)
            .sheet(isPresented: $showShareSheet) {
                if let url = exportedSTLURL {
                    ShareSheet(activityItems: [url])
                }
            }
            .alert(item: Binding<AlertItem?>(
                get: { errorMessage.map { AlertItem(message: $0) } },
                set: { _ in errorMessage = nil }
            )) { item in
                Alert(title: Text("Aviso de Sistema"), message: Text(item.message), dismissButton: .default(Text("OK")))
            }
            .onAppear {
                setupNetworkingCallbacks()
                volumeTrigger.onTriggerAction = {
                    if !isProcessing && captureManager.isSessionRunning {
                        startTrueDepthScan()
                    }
                }
                webServer.start()
            }
        }
    }
    
    private func setupNetworkingCallbacks() {
        webServer.getLatestJPEG = { captureManager.latestJPEGData }
        webServer.getLatestSTL = { latestSTLData }
        webServer.getLatestPLY = { latestPLYData }
        webServer.onTriggerScan = { maxDist, sorEnabled, planeCut, completion in
            self.maxDistance = maxDist
            guard let depthData = captureManager.lastCapturedDepthData else {
                completion(0)
                return
            }
            
            DispatchQueue.global(qos: .userInitiated).async {
                let colorBuf = captureManager.latestColorPixelBuffer
                if let mesh = MeshGenerator.generateMesh(
                    from: depthData,
                    colorPixelBuffer: colorBuf,
                    minDepth: self.minDistance,
                    maxDepth: maxDist,
                    stepSize: self.meshResolution,
                    enableSOR: sorEnabled,
                    basePlaneCutoffZ: planeCut
                ), !mesh.triangles.isEmpty {
                    let stlUrl = try? STLExporter.exportToBinarySTL(mesh: mesh)
                    let plyUrl = try? PLYExporter.exportToBinaryPLY(mesh: mesh)
                    let stlBytes = stlUrl != nil ? try? Data(contentsOf: stlUrl!) : nil
                    let plyBytes = plyUrl != nil ? try? Data(contentsOf: plyUrl!) : nil
                    
                    DispatchQueue.main.async {
                        self.capturedMesh = mesh
                        self.exportedSTLURL = stlUrl
                        self.exportedPLYURL = plyUrl
                        self.latestSTLData = stlBytes
                        self.latestPLYData = plyBytes
                        self.showPreviewMode = true
                        completion(mesh.triangles.count)
                    }
                } else {
                    DispatchQueue.main.async { completion(0) }
                }
            }
        }
        webServer.onTriggerShareOnPhone = { self.generateAndShareSTL() }
    }
    
    private func startTrueDepthScan() {
        guard let depthData = captureManager.lastCapturedDepthData else {
            errorMessage = "Apunta el sensor TrueDepth hacia el objeto (20 a 50 cm)."
            return
        }
        
        let generator = UIImpactFeedbackGenerator(style: .medium)
        generator.impactOccurred()
        
        isProcessing = true
        let colorBuf = captureManager.latestColorPixelBuffer
        
        DispatchQueue.global(qos: .userInitiated).async {
            let mesh = MeshGenerator.generateMesh(
                from: depthData,
                colorPixelBuffer: colorBuf,
                minDepth: self.minDistance,
                maxDepth: self.maxDistance,
                stepSize: self.meshResolution,
                enableSOR: self.enableSORFilter,
                basePlaneCutoffZ: self.basePlaneCutoff > 0 ? self.basePlaneCutoff : nil
            )
            
            DispatchQueue.main.async {
                self.isProcessing = false
                if let mesh = mesh, !mesh.triangles.isEmpty {
                    self.capturedMesh = mesh
                    self.showPreviewMode = true
                    
                    if let url = try? STLExporter.exportToBinarySTL(mesh: mesh) {
                        self.exportedSTLURL = url
                        self.latestSTLData = try? Data(contentsOf: url)
                    }
                    if let plyUrl = try? PLYExporter.exportToBinaryPLY(mesh: mesh) {
                        self.exportedPLYURL = plyUrl
                        self.latestPLYData = try? Data(contentsOf: plyUrl)
                    }
                    
                    let successFeedback = UINotificationFeedbackGenerator()
                    successFeedback.notificationOccurred(.success)
                } else {
                    self.errorMessage = "No se encontraron suficientes puntos en el rango geométrico configurado."
                }
            }
        }
    }
    
    private func generateAndShareSTL() {
        guard let mesh = capturedMesh else { return }
        do {
            let formatter = DateFormatter()
            formatter.dateFormat = "yyyyMMdd_HHmmss"
            let filename = "TrueDepth_Scan_\(formatter.string(from: Date())).stl"
            let url = try STLExporter.exportToBinarySTL(mesh: mesh, filename: filename)
            self.exportedSTLURL = url
            self.latestSTLData = try? Data(contentsOf: url)
            self.showShareSheet = true
        } catch {
            self.errorMessage = error.localizedDescription
        }
    }
    
    private func generateAndSharePLY() {
        guard let mesh = capturedMesh else { return }
        do {
            let formatter = DateFormatter()
            formatter.dateFormat = "yyyyMMdd_HHmmss"
            let filename = "TrueDepth_ColorScan_\(formatter.string(from: Date())).ply"
            let url = try PLYExporter.exportToBinaryPLY(mesh: mesh, filename: filename)
            self.exportedPLYURL = url
            self.latestPLYData = try? Data(contentsOf: url)
            self.showShareSheet = true
        } catch {
            self.errorMessage = error.localizedDescription
        }
    }
}

struct AlertItem: Identifiable {
    var id = UUID()
    var message: String
}

struct ShareSheet: UIViewControllerRepresentable {
    var activityItems: [Any]
    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: activityItems, applicationActivities: nil)
    }
    func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) {}
}
