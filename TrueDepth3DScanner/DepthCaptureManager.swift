import Foundation
import Combine
import AVFoundation
import CoreImage
import UIKit

/// Who or what started/stopped a capture. Shown verbatim on iPhone and Windows.
enum CaptureSource: String {
    case phone = "botón del iPhone"
    case windows = "orden de Windows"
    case volume = "doble pulsación de volumen"
    case web = "panel web del iPhone"
    case storage = "fallo de almacenamiento"
    case sensor = "sensor TrueDepth sin datos válidos"
    case network = "red insuficiente"
    case thermal = "temperatura crítica"
    case interruption = "interrupción de la cámara"
}

/// Capture only: raw depth + BGRA on disk, lightweight preview, progressive upload.
final class DepthCaptureManager: NSObject, ObservableObject, AVCaptureDataOutputSynchronizerDelegate {
    @Published var isSessionRunning = false
    @Published var isScanning = false
    @Published var isBusy = false
    @Published var capturedFrames = 0
    @Published var droppedFrames = 0
    @Published var recordedMB = 0.0
    @Published var statusMessage = "Iniciando TrueDepth…"
    @Published var captureError = ""
    /// preparing, idle, starting, waitingSensor, capturing, pausedCache, finishing, blockedCache, error
    @Published var phase = "preparing"
    @Published var stopReason = ""
    @Published var transferKind = "idle"
    @Published var transferMessage = "Sin envíos pendientes"
    @Published var pendingMB = 0.0
    @Published var pendingSessions = 0
    @Published var rejectedSessions = 0
    @Published var legacyMB = 0.0
    @Published var sensorFormat = ""
    @Published var sensorDiagnostics = ""

    let captureSession = AVCaptureSession()
    let streamClient = BinaryStreamClient()
    private let sessionQueue = DispatchQueue(label: "com.bichito.session")
    private let synchronizerQueue = DispatchQueue(label: "com.bichito.sync")
    private let processingQueue = DispatchQueue(label: "com.bichito.capture")
    private let transferQueue = DispatchQueue(label: "com.bichito.transfer", qos: .utility)
    private let videoDataOutput = AVCaptureVideoDataOutput()
    private let depthDataOutput = AVCaptureDepthDataOutput()
    private var dataOutputSynchronizer: AVCaptureDataOutputSynchronizer?
    private let ciContext = CIContext(options: [.cacheIntermediates: false])
    private let motionTracker = MotionTracker()
    private let recorder = CaptureRecorder()
    private lazy var uploader = ChunkStreamer(address: { [weak self] in self?.streamClient.targetAddress ?? "" })
    private var scanActive = false // processingQueue only
    private var frameID: UInt32 = 0
    private var lastPreview = -Double.infinity

    private struct Stats {
        var callbacks = 0, syncDropped = 0, busy = 0, uncalibrated = 0, cacheSkipped = 0, recorded = 0
        var dropped: Int { syncDropped + busy + uncalibrated + cacheSkipped }
        var text: String {
            "\(callbacks) muestras sincronizadas, \(recorded) grabadas, \(syncDropped) descartadas por sincronización, \(uncalibrated) sin calibración, \(busy) por carga, \(cacheSkipped) por caché llena"
        }
    }
    private let lock = NSLock()
    private var processing = false
    private var stats = Stats()
    private var captureGate = false
    private var pauseStart: Date?
    private var jpeg: Data?
    // Main thread only.
    private var lastTransition = Date.distantPast
    private var pendingStop: (CaptureSource, String?)?
    private var startedAt = Date()
    private var refreshTimer: Timer?

    var latestJPEGData: Data? {
        lock.lock(); defer { lock.unlock() }
        return jpeg
    }

    override init() {
        super.init()
        streamClient.startDiscovery()
        CaptureRecorder.sealInterruptedSessions()
        uploader.onStatus = { [weak self] kind, text in
            DispatchQueue.main.async {
                self?.transferKind = kind.rawValue
                self?.transferMessage = text
            }
        }
        uploader.start()
        let center = NotificationCenter.default
        center.addObserver(self, selector: #selector(sessionInterrupted(_:)), name: .AVCaptureSessionWasInterrupted, object: captureSession)
        center.addObserver(self, selector: #selector(sessionInterruptionEnded(_:)), name: .AVCaptureSessionInterruptionEnded, object: captureSession)
        center.addObserver(self, selector: #selector(sessionRuntimeError(_:)), name: .AVCaptureSessionRuntimeError, object: captureSession)
        refreshTimer = Timer.scheduledTimer(withTimeInterval: 2, repeats: true) { [weak self] _ in self?.refreshPending() }
        refreshPending()
        checkPermissionsAndConfigure()
    }

    deinit { NotificationCenter.default.removeObserver(self); refreshTimer?.invalidate() }

    func refreshPending() {
        transferQueue.async { [weak self] in
            let summary = CaptureRecorder.summary()
            DispatchQueue.main.async {
                self?.pendingMB = Double(summary.bytes) / 1_000_000
                self?.pendingSessions = summary.sessions
                self?.rejectedSessions = summary.rejected
                self?.legacyMB = Double(summary.legacyBytes) / 1_000_000
            }
        }
    }

    private func checkPermissionsAndConfigure() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            sessionQueue.async { [weak self] in self?.configureSession() }
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
                if granted {
                    self?.sessionQueue.async { self?.configureSession() }
                } else {
                    self?.fail(phase: "error", "Permiso de cámara denegado. Actívalo en Ajustes > Bichito 3D Pro > Cámara.")
                }
            }
        default:
            fail(phase: "error", "Permiso de cámara no disponible o restringido. Actívalo en Ajustes > Bichito 3D Pro > Cámara.")
        }
    }

    private func fail(phase: String, _ message: String) {
        DispatchQueue.main.async {
            self.phase = phase
            self.captureError = message
            self.statusMessage = message
        }
    }

    private static func fourCC(_ code: OSType) -> String {
        let bytes = [UInt8((code >> 24) & 0xff), UInt8((code >> 16) & 0xff), UInt8((code >> 8) & 0xff), UInt8(code & 0xff)]
        return String(bytes: bytes, encoding: .ascii) ?? "\(code)"
    }

    /// Native Float32 depth avoids any conversion; other formats are recorded as such.
    private func selectDepthFormat(_ device: AVCaptureDevice) -> String {
        let formats = device.activeFormat.supportedDepthDataFormats
        let native = formats.filter { CMFormatDescriptionGetMediaSubType($0.formatDescription) == kCVPixelFormatType_DepthFloat32 }
        if let best = native.max(by: {
            CMVideoFormatDescriptionGetDimensions($0.formatDescription).width < CMVideoFormatDescriptionGetDimensions($1.formatDescription).width
        }) {
            device.activeDepthDataFormat = best
            let size = CMVideoFormatDescriptionGetDimensions(best.formatDescription)
            return "Profundidad DepthFloat32 nativa \(size.width)x\(size.height)"
        }
        let available = formats.map { Self.fourCC(CMFormatDescriptionGetMediaSubType($0.formatDescription)) }.joined(separator: ",")
        return "Sin DepthFloat32 nativo (disponibles: \(available)); se convierte y se registra el formato original"
    }

    private func configureSession() {
        captureSession.beginConfiguration()
        captureSession.sessionPreset = .vga640x480
        guard let device = AVCaptureDevice.default(.builtInTrueDepthCamera, for: .video, position: .front) else {
            captureSession.commitConfiguration()
            fail(phase: "error", "No se detectó la cámara TrueDepth (Face ID) en este iPhone.")
            return
        }
        do {
            let input = try AVCaptureDeviceInput(device: device)
            guard captureSession.canAddInput(input) else {
                captureSession.commitConfiguration()
                fail(phase: "error", "iOS no permite añadir la cámara TrueDepth a la sesión de captura.")
                return
            }
            captureSession.addInput(input)
        } catch {
            captureSession.commitConfiguration()
            fail(phase: "error", "No se pudo abrir la cámara TrueDepth: \(error.localizedDescription)")
            return
        }
        guard captureSession.canAddOutput(videoDataOutput), captureSession.canAddOutput(depthDataOutput) else {
            captureSession.commitConfiguration()
            fail(phase: "error", "Este formato de cámara no admite vídeo y profundidad sincronizados.")
            return
        }
        videoDataOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA)]
        videoDataOutput.alwaysDiscardsLateVideoFrames = true
        captureSession.addOutput(videoDataOutput)
        // Raw, unfiltered depth: holes stay holes (NaN/0); nothing is invented on the phone.
        depthDataOutput.isFilteringEnabled = false
        depthDataOutput.alwaysDiscardsLateDepthData = true
        captureSession.addOutput(depthDataOutput)
        depthDataOutput.connection(with: .depthData)?.isEnabled = true
        var format = ""
        do {
            try device.lockForConfiguration()
            format = selectDepthFormat(device)
            for range in device.activeFormat.videoSupportedFrameRateRanges where range.minFrameRate <= 15 && range.maxFrameRate >= 15 {
                device.activeVideoMinFrameDuration = CMTime(value: 1, timescale: 15)
                device.activeVideoMaxFrameDuration = CMTime(value: 1, timescale: 15)
                break
            }
            device.unlockForConfiguration()
        } catch {
            format = "Configuración del sensor no aplicada: \(error.localizedDescription)"
        }
        captureSession.commitConfiguration()
        dataOutputSynchronizer = AVCaptureDataOutputSynchronizer(dataOutputs: [videoDataOutput, depthDataOutput])
        dataOutputSynchronizer?.setDelegate(self, queue: synchronizerQueue)
        captureSession.startRunning()
        let running = captureSession.isRunning
        DispatchQueue.main.async {
            self.sensorFormat = format
            self.isSessionRunning = running
            self.phase = running ? "idle" : "error"
            self.statusMessage = running ? "TrueDepth listo. Pulsa INICIAR CAPTURA (o 2× Vol+)." : "La sesión de cámara no arrancó."
        }
    }

    // MARK: Interruptions

    @objc private func sessionInterrupted(_ note: Notification) {
        var text = "motivo desconocido"
        if let raw = note.userInfo?[AVCaptureSessionInterruptionReasonKey] as? Int,
           let reason = AVCaptureSession.InterruptionReason(rawValue: raw) {
            switch reason {
            case .videoDeviceNotAvailableInBackground: text = "la app pasó a segundo plano"
            case .videoDeviceInUseByAnotherClient: text = "otra app está usando la cámara"
            case .videoDeviceNotAvailableWithMultipleForegroundApps: text = "multitarea en pantalla"
            case .videoDeviceNotAvailableDueToSystemPressure: text = "presión del sistema/temperatura"
            case .audioDeviceInUseByAnotherClient: text = "audio ocupado por otra app"
            @unknown default: text = "código \(raw)"
            }
        }
        DispatchQueue.main.async {
            self.isSessionRunning = false
            if self.isScanning {
                self.stopScan(source: .interruption, reason: "iOS interrumpió la cámara: \(text). Los fotogramas grabados se conservan.")
            } else {
                self.statusMessage = "Cámara en pausa: \(text)."
            }
        }
    }

    @objc private func sessionInterruptionEnded(_ note: Notification) {
        DispatchQueue.main.async {
            self.isSessionRunning = true
            if !self.isScanning { self.phase = "idle"; self.statusMessage = "Cámara TrueDepth disponible de nuevo." }
        }
    }

    @objc private func sessionRuntimeError(_ note: Notification) {
        let error = note.userInfo?[AVCaptureSessionErrorKey] as? NSError
        let text = error.map { "\($0.domain) \($0.code): \($0.localizedDescription)" } ?? "sin detalle"
        DispatchQueue.main.async {
            if self.isScanning { self.stopScan(source: .interruption, reason: "Error de la cámara: \(text)") }
            else { self.statusMessage = "Error de la cámara: \(text). Reiniciando sesión…" }
        }
        sessionQueue.async { [weak self] in
            guard let self = self, !self.captureSession.isRunning else { return }
            self.captureSession.startRunning()
            let running = self.captureSession.isRunning
            DispatchQueue.main.async { self.isSessionRunning = running }
        }
    }

    // MARK: Commands (main thread)

    func toggleScan(source: CaptureSource) {
        // Volume KVO may emit bursts; never let it flip a state that just changed.
        if source == .volume && Date().timeIntervalSince(lastTransition) < 1.5 { return }
        if isScanning { stopScan(source: source) } else { startScan(source: source) }
    }

    func startScan(source: CaptureSource) {
        guard !isScanning, !isBusy else { return } // idempotent
        guard isSessionRunning else {
            statusMessage = "No se puede iniciar: la cámara TrueDepth no está activa (\(statusMessage))."
            captureError = statusMessage
            return
        }
        if source == .volume && Date().timeIntervalSince(lastTransition) < 1.5 { return }
        let summary = CaptureRecorder.summary()
        if summary.bytes + 16_000_000 >= CaptureRecorder.cacheLimit {
            phase = "blockedCache"
            captureError = "No se puede iniciar: \(summary.bytes / 1_000_000) MB de \(summary.sessions) sesión(es) anteriores siguen sin confirmar por Windows y llenan la caché (\(CaptureRecorder.cacheLimit / 1_000_000) MB). Abre Bichito3D en Windows en la misma red para recibirlas, o descártalas en 'Sesiones pendientes'."
            statusMessage = captureError
            return
        }
        captureError = ""; stopReason = ""
        isBusy = true; phase = "starting"; lastTransition = Date()
        statusMessage = "Iniciando captura (\(source.rawValue))…"
        processingQueue.async { [weak self] in
            guard let self = self else { return }
            do {
                try self.recorder.start()
                self.frameID = 0
                self.lock.lock(); self.stats = Stats(); self.captureGate = true; self.pauseStart = nil; self.lock.unlock()
                self.motionTracker.startTracking()
                self.scanActive = true
                DispatchQueue.main.async {
                    self.isBusy = false
                    self.isScanning = true
                    self.capturedFrames = 0; self.droppedFrames = 0; self.recordedMB = 0
                    self.phase = "waitingSensor"
                    self.startedAt = Date()
                    self.statusMessage = "Captura iniciada por \(source.rawValue): esperando el primer fotograma TrueDepth…"
                    self.scheduleWatchdog()
                    if let stop = self.pendingStop {
                        self.pendingStop = nil
                        self.stopScan(source: stop.0, reason: stop.1)
                    }
                }
            } catch {
                DispatchQueue.main.async {
                    self.isBusy = false
                    self.phase = "error"
                    self.captureError = "No se pudo crear la sesión en el almacenamiento del iPhone: \(error.localizedDescription)"
                    self.statusMessage = self.captureError
                }
            }
        }
    }

    func stopScan(source: CaptureSource, reason: String? = nil) {
        if isBusy && phase == "starting" { pendingStop = (source, reason); return }
        guard isScanning, !isBusy else { return } // idempotent
        if source == .volume && Date().timeIntervalSince(lastTransition) < 1.5 { return }
        lastTransition = Date()
        let failure = reason ?? ""
        if !failure.isEmpty { captureError = failure }
        let why = failure.isEmpty ? "Detenida por \(source.rawValue)" : "\(source.rawValue): \(failure)"
        stopReason = why
        isScanning = false
        isBusy = true
        phase = "finishing"
        statusMessage = "Cerrando captura (\(why))…"
        processingQueue.async { [weak self] in
            guard let self = self else { return }
            guard self.scanActive else { DispatchQueue.main.async { self.isBusy = false }; return }
            self.scanActive = false
            self.motionTracker.stopTracking()
            self.lock.lock(); self.captureGate = false; let stats = self.stats; self.pauseStart = nil; self.lock.unlock()
            var message = ""
            var nextPhase = failure.isEmpty ? "idle" : "error"
            do {
                let url = try self.recorder.finish(dropped: stats.dropped, reason: why)
                let count = self.recorder.count
                if count == 0 {
                    try? FileManager.default.removeItem(at: url)
                    message = failure.isEmpty
                        ? "Captura detenida por \(source.rawValue) antes de grabar ningún fotograma. Sensor: \(stats.text)."
                        : "\(failure) Ningún fotograma grabado. Sensor: \(stats.text)."
                } else {
                    message = failure.isEmpty
                        ? "Captura finalizada (\(source.rawValue)): \(count) fotogramas guardados, \(stats.dropped) omitidos. Enviando bloques pendientes a Windows…"
                        : "\(failure) \(count) fotogramas conservados; se enviarán a Windows."
                    self.uploader.start()
                }
                DispatchQueue.main.async { self.capturedFrames = count; self.droppedFrames = stats.dropped }
            } catch {
                nextPhase = "error"
                message = "Fallo de almacenamiento al cerrar la captura: \(error.localizedDescription). Los bloques ya escritos se conservan."
            }
            DispatchQueue.main.async {
                self.isBusy = false
                self.phase = nextPhase
                if nextPhase == "error" { self.captureError = message }
                self.statusMessage = message
                self.sensorDiagnostics = stats.text
                self.refreshPending()
            }
        }
    }

    func retryTransfer() {
        uploader.retryRejected()
        uploader.start()
        statusMessage = "Reintento de envío activo; los bloques ya confirmados no se reenvían."
    }

    /// Explicit user decision only (confirmation dialog in the UI).
    func discardPendingSessions() {
        guard !isScanning, !isBusy else { return }
        transferQueue.async { [weak self] in
            let freed = CaptureRecorder.discardPending(except: nil)
            DispatchQueue.main.async {
                self?.statusMessage = "Sesiones no confirmadas descartadas por el usuario: \(freed / 1_000_000) MB liberados."
                self?.captureError = ""
                if self?.phase == "blockedCache" { self?.phase = "idle" }
                self?.refreshPending()
            }
        }
    }

    /// While no frame has been recorded, report sensor progress instead of a
    /// generic failure; give up with the concrete counters after 15 s.
    private func scheduleWatchdog() {
        DispatchQueue.main.asyncAfter(deadline: .now() + 2) { [weak self] in
            guard let self = self, self.isScanning else { return }
            self.lock.lock(); let stats = self.stats; self.lock.unlock()
            self.sensorDiagnostics = stats.text
            if stats.recorded == 0 {
                let elapsed = Date().timeIntervalSince(self.startedAt)
                if !self.captureSession.isRunning {
                    self.stopScan(source: .sensor, reason: "La sesión de cámara se detuvo antes del primer fotograma.")
                    return
                }
                if elapsed >= 15 {
                    self.stopScan(source: .sensor, reason: "Sin fotogramas TrueDepth válidos en 15 s (\(stats.text)).")
                    return
                }
                self.phase = "waitingSensor"
                self.statusMessage = "Esperando al sensor TrueDepth (\(Int(elapsed)) s): \(stats.text)."
            }
            self.scheduleWatchdog()
        }
    }

    // MARK: Frames

    func dataOutputSynchronizer(_ synchronizer: AVCaptureDataOutputSynchronizer,
                                didOutput synchronizedDataCollection: AVCaptureSynchronizedDataCollection) {
        lock.lock()
        if captureGate { stats.callbacks += 1 }
        if processing {
            if captureGate { stats.busy += 1 }
            lock.unlock()
            return
        }
        processing = true
        lock.unlock()
        processingQueue.async { [weak self] in
            guard let self = self else { return }
            defer { self.lock.lock(); self.processing = false; self.lock.unlock() }
            autoreleasepool { self.process(synchronizedDataCollection) }
        }
    }

    private func process(_ collection: AVCaptureSynchronizedDataCollection) {
        guard let d = collection.synchronizedData(for: depthDataOutput) as? AVCaptureSynchronizedDepthData,
              let v = collection.synchronizedData(for: videoDataOutput) as? AVCaptureSynchronizedSampleBufferData,
              !d.depthDataWasDropped, !v.sampleBufferWasDropped,
              let pixel = CMSampleBufferGetImageBuffer(v.sampleBuffer) else {
            if scanActive { lock.lock(); stats.syncDropped += 1; lock.unlock() }
            return
        }
        let timestamp = CMSampleBufferGetPresentationTimeStamp(v.sampleBuffer).seconds
        if scanActive { record(depth: d.depthData, pixel: pixel, timestamp: timestamp) }
        // Five small JPEGs per second, independent of raw capture storage.
        if timestamp - lastPreview >= 0.2 {
            lastPreview = timestamp
            let image = CIImage(cvPixelBuffer: pixel).oriented(.right)
            let small = image.transformed(by: CGAffineTransform(scaleX: 0.5, y: 0.5))
            if let cg = ciContext.createCGImage(small, from: small.extent) {
                let data = UIImage(cgImage: cg).jpegData(compressionQuality: 0.4)
                lock.lock(); jpeg = data; lock.unlock()
            }
        }
    }

    private func record(depth: AVDepthData, pixel: CVPixelBuffer, timestamp: Double) {
        guard let packet = FrameSerializer.serialize(frameID: frameID, timestamp: timestamp,
            depthData: depth, colorPixelBuffer: pixel,
            deviceMotion: motionTracker.latestDeviceMotion, rawCapture: true) else {
            // Startup may briefly deliver depth without calibration: wait, never fail.
            lock.lock(); stats.uncalibrated += 1; lock.unlock()
            return
        }
        do {
            try recorder.append(packet)
            frameID &+= 1
            lock.lock()
            stats.recorded = recorder.count
            let resumed = pauseStart != nil
            pauseStart = nil
            let snapshot = stats
            lock.unlock()
            let mb = Double(recorder.bytes) / 1_000_000
            if snapshot.recorded == 1 || snapshot.recorded % 5 == 0 || resumed {
                DispatchQueue.main.async {
                    self.capturedFrames = snapshot.recorded; self.recordedMB = mb; self.droppedFrames = snapshot.dropped
                    if self.isScanning && (self.phase != "capturing" || resumed) {
                        self.phase = "capturing"
                        self.statusMessage = resumed
                            ? "Captura reanudada: Windows liberó caché."
                            : "Capturando profundidad Float32 + RGB sin pérdida; bloques hacia Windows."
                    }
                    if ProcessInfo.processInfo.thermalState == .critical {
                        self.stopScan(source: .thermal, reason: "Temperatura crítica del iPhone. Los fotogramas capturados se conservan.")
                    }
                }
            }
        } catch let full as CaptureRecorder.CacheFull {
            lock.lock()
            stats.cacheSkipped += 1
            if pauseStart == nil { pauseStart = Date() }
            let paused = Date().timeIntervalSince(pauseStart ?? Date())
            lock.unlock()
            let silent = Date().timeIntervalSince(uploader.lastAckDate)
            let transfer = uploader.current.text
            DispatchQueue.main.async {
                guard self.isScanning else { return }
                if paused > 20 && silent > 15 {
                    self.stopScan(source: .network, reason: "La red no sostiene la captura: con la caché llena Windows no confirmó bloques durante \(Int(silent)) s (\(transfer)). Los fotogramas grabados se conservan y se enviarán al reconectar.")
                } else {
                    self.phase = "pausedCache"
                    self.statusMessage = "Pausa por red: \(full.localizedDescription) Se reanuda sola al confirmar Windows; los fotogramas no grabados se cuentan como omitidos (no es compresión)."
                }
            }
        } catch {
            let failure = error as NSError
            let index = recorder.count
            DispatchQueue.main.async {
                self.stopScan(source: .storage, reason: "No se pudo guardar el fotograma \(index): \(failure.domain) \(failure.code) · \(failure.localizedDescription).")
            }
        }
    }

    /// Snapshot for /status (Windows) and the web panel. Main thread only.
    func statusDictionary() -> [String: Any] {
        lock.lock(); let stats = self.stats; lock.unlock()
        return ["isScanning": isScanning, "busy": isBusy, "phase": phase,
                "frames": capturedFrames, "dropped": droppedFrames, "recordedMB": recordedMB,
                "status": statusMessage, "captureError": captureError, "stopReason": stopReason,
                "sensor": sensorFormat, "sensorStats": stats.text,
                "transfer": transferKind, "transferMessage": transferMessage,
                "pendingMB": pendingMB, "pendingSessions": pendingSessions, "rejectedSessions": rejectedSessions,
                "legacyMB": legacyMB, "cacheLimitMB": Double(CaptureRecorder.cacheLimit) / 1_000_000,
                "windowsIP": streamClient.serverIP, "windowsConnected": streamClient.isConnected]
    }
}
