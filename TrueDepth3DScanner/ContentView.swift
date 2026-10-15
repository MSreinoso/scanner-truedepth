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
    @State private var capturedMesh: Mesh3D?
    @State private var exportedSTLURL: URL?
    @State private var latestSTLData: Data?
    @State private var showShareSheet = false
    @State private var isProcessing = false
    @State private var showPreviewMode = false
    @State private var showWireframe = false
    @State private var errorMessage: String?
    
    @State private var minDistance: Float = 0.15
    @State private var maxDistance: Float = 0.50
    @State private var meshResolution: Int = 2
    
    var body: some View {
        NavigationView {
            VStack(spacing: 0) {
                HStack {
                    Image(systemName: "wifi")
                        .foregroundColor(.green)
                    Text("Monitor PC: \(webServer.serverURL)")
                        .font(.caption.monospaced().bold())
                    Spacer()
                }
                .padding(.horizontal)
                .padding(.vertical, 8)
                .background(Color.black.opacity(0.85))
                .foregroundColor(.white)
                
                Picker("Modo", selection: $showPreviewMode) {
                    Text("1. Cámara TrueDepth").tag(false)
                    Text("2. Vista Previa 3D").tag(true)
                }
                .pickerStyle(SegmentedPickerStyle())
                .padding()
                .background(Color(UIColor.secondarySystemBackground))
                
                ZStack {
                    if !showPreviewMode {
                        CameraPreviewView(session: captureManager.captureSession)
                            .edgesIgnoringSafeArea(.horizontal)
                        
                        RoundedRectangle(cornerRadius: 24)
                            .strokeBorder(style: StrokeStyle(lineWidth: 2, dash: [10]))
                            .foregroundColor(.cyan.opacity(0.8))
                            .padding(40)
                        
                        VStack {
                            Spacer()
                            Text("Coloca el objeto a 20 - 50 cm de la cámara frontal")
                                .font(.caption.bold())
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
                                    Label("\(mesh.triangles.count) triángulos", systemImage: "cube.fill")
                                        .font(.caption.bold())
                                        .padding(8)
                                        .background(.ultraThinMaterial)
                                        .cornerRadius(8)
                                    
                                    Spacer()
                                    
                                    Button(action: { showWireframe.toggle() }) {
                                        Label(showWireframe ? "Sólido" : "Malla", systemImage: showWireframe ? "cube" : "square.grid.3x3")
                                            .font(.caption.bold())
                                            .padding(8)
                                            .background(.ultraThinMaterial)
                                            .cornerRadius(8)
                                    }
                                }
                                .padding()
                                
                                Spacer()
                                
                                Text("Arrastra con 1 dedo para rotar • Pellizca para hacer zoom")
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
                                Text("Aún no has capturado ningún modelo.\nPulsa 'Iniciar escaneo 3D TrueDepth'.")
                                    .font(.subheadline)
                                    .foregroundColor(.gray)
                                    .multilineTextAlignment(.center)
                                    .padding(.horizontal, 32)
                            }
                        }
                    }
                    
                    if isProcessing {
                        VStack(spacing: 12) {
                            ProgressView()
                                .scaleEffect(1.3)
                            Text("Reconstruyendo geometría 3D...")
                                .font(.subheadline.bold())
                        }
                        .padding(24)
                        .background(.ultraThinMaterial)
                        .cornerRadius(16)
                    }
                }
                .frame(maxHeight: .infinity)
                
                VStack(spacing: 14) {
                    VStack(alignment: .leading, spacing: 4) {
                        HStack {
                            Text("Recortar fondo a partir de:")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            Spacer()
                            Text("\(Int(maxDistance * 100)) cm")
                                .font(.caption.bold())
                                .foregroundColor(.cyan)
                        }
                        Slider(value: $maxDistance, in: 0.25...0.80, step: 0.05)
                            .accentColor(.cyan)
                    }
                    
                    Button(action: startTrueDepthScan) {
                        HStack(spacing: 10) {
                            Image(systemName: "viewfinder.circle.fill")
                                .font(.title2)
                            Text("Iniciar escaneo 3D TrueDepth")
                                .font(.headline.bold())
                        }
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 15)
                        .background(
                            LinearGradient(
                                colors: [Color.blue, Color.cyan],
                                startPoint: .leading,
                                endPoint: .trailing
                            )
                        )
                        .foregroundColor(.white)
                        .cornerRadius(14)
                    }
                    .disabled(isProcessing || !captureManager.isSessionRunning)
                    
                    if capturedMesh != nil {
                        HStack(spacing: 12) {
                            Button(action: {
                                showPreviewMode = false
                            }) {
                                Label("Repetir toma", systemImage: "arrow.counterclockwise")
                                    .font(.subheadline.bold())
                                    .frame(maxWidth: .infinity)
                                    .padding(.vertical, 12)
                                    .background(Color(UIColor.tertiarySystemFill))
                                    .foregroundColor(.primary)
                                    .cornerRadius(12)
                            }
                            
                            Button(action: generateAndShareSTL) {
                                Label("Compartir .STL (WhatsApp / Correo)", systemImage: "square.and.arrow.up.fill")
                                    .font(.subheadline.bold())
                                    .frame(maxWidth: .infinity)
                                    .padding(.vertical, 12)
                                    .background(Color.green)
                                    .foregroundColor(.white)
                                    .cornerRadius(12)
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
                Alert(title: Text("Atención"), message: Text(item.message), dismissButton: .default(Text("Entendido")))
            }
            .onAppear {
                webServer.getLatestJPEG = {
                    return captureManager.latestJPEGData
                }
                webServer.getLatestSTL = {
                    return latestSTLData
                }
                webServer.onTriggerScan = { maxDist, completion in
                    self.maxDistance = maxDist
                    guard let depthData = captureManager.lastCapturedDepthData else {
                        completion(0)
                        return
                    }
                    DispatchQueue.global(qos: .userInitiated).async {
                        if let mesh = MeshGenerator.generateMesh(from: depthData, minDepth: self.minDistance, maxDepth: maxDist, stepSize: self.meshResolution),
                           !mesh.triangles.isEmpty,
                           let url = try? STLExporter.exportToBinarySTL(mesh: mesh),
                           let stlBytes = try? Data(contentsOf: url) {
                            DispatchQueue.main.async {
                                self.capturedMesh = mesh
                                self.exportedSTLURL = url
                                self.latestSTLData = stlBytes
                                self.showPreviewMode = true
                                completion(mesh.triangles.count)
                            }
                        } else {
                            DispatchQueue.main.async {
                                completion(0)
                            }
                        }
                    }
                }
                webServer.onTriggerShareOnPhone = {
                    self.generateAndShareSTL()
                }
                webServer.start()
            }
        }
    }
    
    private func startTrueDepthScan() {
        guard let depthData = captureManager.lastCapturedDepthData else {
            errorMessage = "El sensor TrueDepth aún no detecta profundidad. Apunta a un objeto entre 20 y 50 cm."
            return
        }
        
        let generator = UIImpactFeedbackGenerator(style: .medium)
        generator.impactOccurred()
        
        isProcessing = true
        DispatchQueue.global(qos: .userInitiated).async {
            let mesh = MeshGenerator.generateMesh(
                from: depthData,
                minDepth: self.minDistance,
                maxDepth: self.maxDistance,
                stepSize: self.meshResolution
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
                    let successFeedback = UINotificationFeedbackGenerator()
                    successFeedback.notificationOccurred(.success)
                } else {
                    self.errorMessage = "No se encontraron puntos suficientes en el rango configurado."
                }
            }
        }
    }
    
    private func generateAndShareSTL() {
        guard let mesh = capturedMesh else { return }
        do {
            let formatter = DateFormatter()
            formatter.dateFormat = "yyyyMMdd_HHmmss"
            let filename = "Modelo3D_TrueDepth_\(formatter.string(from: Date())).stl"
            let fileURL = try STLExporter.exportToBinarySTL(mesh: mesh, filename: filename)
            self.exportedSTLURL = fileURL
            self.latestSTLData = try? Data(contentsOf: fileURL)
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
