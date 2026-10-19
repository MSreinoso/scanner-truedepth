import SwiftUI
import AVFoundation

private final class PreviewSurface: UIView {
    override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
    override func layoutSubviews() { super.layoutSubviews(); layer.frame = bounds }
}
struct CameraPreviewView: UIViewRepresentable {
    let session: AVCaptureSession
    func makeUIView(context: Context) -> UIView {
        let view = PreviewSurface()
        let layer = view.layer as! AVCaptureVideoPreviewLayer
        layer.session = session
        layer.videoGravity = .resizeAspectFill
        return view
    }
    func updateUIView(_ uiView: UIView, context: Context) {}
}

struct ContentView: View {
    @StateObject private var capture = DepthCaptureManager()
    @StateObject private var server = WebStreamServer()
    @StateObject private var volume = VolumeButtonTriggerManager()
    @AppStorage("manualPCAddress") private var pcIP = ""

    var body: some View {
        NavigationView {
            VStack(spacing: 10) {
                Text(server.serverURL).font(.caption.monospaced()).textSelection(.enabled)
                ConnectionStatus(client: capture.streamClient)
                HStack {
                    TextField("IP de Windows", text: $pcIP)
                        .keyboardType(.decimalPad).textFieldStyle(.roundedBorder)
                    Button("Conectar PC") { capture.streamClient.connect(to: pcIP) }
                        .disabled(pcIP.isEmpty)
                }
                CameraPreviewView(session: capture.captureSession)
                    .overlay(alignment: .topLeading) {
                        Text(capture.isScanning ? "● CAPTURANDO EN DISCO" : "VISTA PREVIA")
                            .font(.caption.bold()).padding(8)
                            .background(.ultraThinMaterial).foregroundColor(capture.isScanning ? .red : .primary)
                    }
                Text("\(capture.capturedFrames) frames • \(capture.recordedMB, specifier: "%.0f") MB • \(capture.droppedFrames) omitidos")
                    .font(.caption.monospaced())
                Text(capture.statusMessage).font(.caption).frame(maxWidth: .infinity, alignment: .leading)
                if capture.isBusy { ProgressView() }
                Button(capture.isScanning ? "FINALIZAR CAPTURA" : "INICIAR CAPTURA") { capture.toggleScan() }
                    .buttonStyle(.borderedProminent).tint(capture.isScanning ? .red : .blue)
                    .disabled(capture.isBusy || !capture.isSessionRunning)
                Button("Reintentar envío pendiente") { capture.retryTransfer() }
                    .disabled(capture.isBusy || capture.isScanning)
                Text("PNG + profundidad Float32 sin pérdida. Caché limitada; se libera solo tras confirmar recepción en Windows. No se reconstruye en el teléfono.")
                    .font(.caption2).foregroundColor(.secondary)
            }
            .padding()
            .navigationTitle("Bichito 3D • Captura")
            .navigationBarTitleDisplayMode(.inline)
            .onAppear {
                UIApplication.shared.isIdleTimerDisabled = true
                server.onConnectToPC = { capture.streamClient.connect(to: $0) }
                server.getLatestJPEG = { capture.latestJPEGData }
                server.getStatusJSON = {
                    DispatchQueue.main.sync {
                        let value: [String: Any] = ["isScanning": capture.isScanning,
                            "busy": capture.isBusy, "frames": capture.capturedFrames,
                            "dropped": capture.droppedFrames, "status": capture.statusMessage]
                        let data = (try? JSONSerialization.data(withJSONObject: value)) ?? Data("{}".utf8)
                        return String(data: data, encoding: .utf8) ?? "{}"
                    }
                }
                server.onToggleScan = { capture.toggleScan() }
                volume.onTriggerAction = { capture.toggleScan() }
                server.start()
            }
        }
    }
}

private struct ConnectionStatus: View {
    @ObservedObject var client: BinaryStreamClient
    var body: some View {
        Text("PC: \(client.connectionState)").font(.caption)
            .foregroundColor(client.isConnected ? .green : .orange)
    }
}
