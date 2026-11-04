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
    @AppStorage("volumeTrigger") private var volumeTrigger = true
    @State private var confirmDiscard = false

    var body: some View {
        NavigationView {
            ScrollView {
                VStack(alignment: .leading, spacing: 10) {
                    connectionCard
                    CameraPreviewView(session: capture.captureSession)
                        .frame(height: 300)
                        .clipShape(RoundedRectangle(cornerRadius: 10))
                        .overlay(alignment: .topLeading) {
                            Text(capture.isScanning ? "● CAPTURANDO" : "VISTA PREVIA")
                                .font(.caption.bold()).padding(8)
                                .background(.ultraThinMaterial).foregroundColor(capture.isScanning ? .red : .primary)
                        }
                    Text("\(capture.capturedFrames) fotogramas • \(capture.recordedMB, specifier: "%.0f") MB sesión • \(capture.droppedFrames) omitidos")
                        .font(.caption.monospaced())
                    Text(capture.statusMessage).font(.callout)
                        .foregroundColor(capture.captureError.isEmpty ? .primary : .red)
                    if !capture.stopReason.isEmpty {
                        Text("Última parada: \(capture.stopReason)").font(.caption).foregroundColor(.secondary)
                    }
                    if capture.isBusy { ProgressView() }
                    Button(capture.isScanning ? "FINALIZAR CAPTURA" : "INICIAR CAPTURA") {
                        if capture.isScanning { capture.stopScan(source: .phone) } else { capture.startScan(source: .phone) }
                    }
                    .frame(maxWidth: .infinity)
                    .buttonStyle(.borderedProminent).tint(capture.isScanning ? .red : .blue)
                    .disabled(capture.isBusy || !capture.isSessionRunning)
                    transferCard
                    Toggle("Doble Vol+ inicia/finaliza", isOn: $volumeTrigger).font(.caption)
                    Text(capture.sensorFormat).font(.caption2).foregroundColor(.secondary)
                    Text("Profundidad Float32 + RGB sin pérdida. La caché del iPhone (≈\(Int(CaptureRecorder.cacheLimit / 1_000_000)) MB) solo guarda bloques aún no confirmados; la sesión completa en Windows puede ser mayor. No se reconstruye en el teléfono.")
                        .font(.caption2).foregroundColor(.secondary)
                }
                .padding()
            }
            .navigationTitle("Bichito 3D • Captura")
            .navigationBarTitleDisplayMode(.inline)
            .confirmationDialog("¿Descartar sesiones no confirmadas?", isPresented: $confirmDiscard, titleVisibility: .visible) {
                Button("Descartar \(Int(capture.pendingMB + capture.legacyMB)) MB", role: .destructive) { capture.discardPendingSessions() }
                Button("Cancelar", role: .cancel) {}
            } message: {
                Text("Windows aún no ha confirmado estas capturas. Si las descartas no podrán reconstruirse.")
            }
            .onAppear {
                UIApplication.shared.isIdleTimerDisabled = true
                server.onConnectToPC = { capture.streamClient.connect(to: $0) }
                server.getLatestJPEG = { capture.latestJPEGData }
                server.getStatusJSON = {
                    DispatchQueue.main.sync {
                        var value = capture.statusDictionary()
                        value["phoneURL"] = server.serverURL
                        let data = (try? JSONSerialization.data(withJSONObject: value)) ?? Data("{}".utf8)
                        return String(data: data, encoding: .utf8) ?? "{}"
                    }
                }
                server.onToggleScan = { capture.toggleScan(source: $0) }
                server.onStartScan = { capture.startScan(source: $0) }
                server.onStopScan = { capture.stopScan(source: $0) }
                volume.onTriggerAction = {
                    if UserDefaults.standard.object(forKey: "volumeTrigger") as? Bool ?? true { capture.toggleScan(source: .volume) }
                }
                server.start()
            }
        }
        .navigationViewStyle(.stack)
    }

    private var connectionCard: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("IP de este iPhone").font(.caption).foregroundColor(.secondary)
            Text(server.serverURL.replacingOccurrences(of: "http://", with: "").replacingOccurrences(of: ":8080", with: ""))
                .font(.title3.monospaced().bold()).textSelection(.enabled)
            Text("Escríbela en Windows (paso 1 · Conectar) si el descubrimiento automático no la encuentra.")
                .font(.caption2).foregroundColor(.secondary)
            ConnectionStatus(client: capture.streamClient)
            HStack {
                TextField("IP de Windows (manual)", text: $pcIP)
                    .keyboardType(.decimalPad).textFieldStyle(.roundedBorder)
                Button("Conectar") { capture.streamClient.connect(to: pcIP) }
                    .disabled(pcIP.isEmpty)
            }
        }
        .padding(10)
        .background(RoundedRectangle(cornerRadius: 10).fill(Color(.secondarySystemBackground)))
    }

    private var transferCard: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Transferencia a Windows").font(.caption.bold())
            Text(capture.transferMessage).font(.caption)
                .foregroundColor(capture.transferKind == "networkError" || capture.transferKind == "rejected" ? .orange : .primary)
            Text(String(format: "Pendiente en el iPhone: %.0f MB en %ld sesión(es)", capture.pendingMB, capture.pendingSessions) + (capture.rejectedSessions > 0 ? " · \(capture.rejectedSessions) apartada(s)" : ""))
                .font(.caption.monospaced())
            if capture.legacyMB >= 1 {
                Text("Archivos de versiones antiguas (v2) que ya no se envían: \(capture.legacyMB, specifier: "%.0f") MB").font(.caption2)
            }
            HStack {
                Button("Reintentar envío") { capture.retryTransfer() }
                    .disabled(capture.isScanning)
                Spacer()
                Button("Sesiones pendientes…", role: .destructive) { confirmDiscard = true }
                    .disabled(capture.isScanning || capture.isBusy || (capture.pendingMB + capture.legacyMB) < 0.01)
            }.font(.caption)
        }
        .padding(10)
        .background(RoundedRectangle(cornerRadius: 10).fill(Color(.secondarySystemBackground)))
    }
}

private struct ConnectionStatus: View {
    @ObservedObject var client: BinaryStreamClient
    var body: some View {
        Text("Windows: \(client.connectionState)" + (client.serverIP.isEmpty ? "" : " (\(client.serverIP))")).font(.caption)
            .foregroundColor(client.isConnected ? .green : .orange)
    }
}
