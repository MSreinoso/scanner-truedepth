import Foundation
import Combine
import AVFoundation
import CoreImage
import UIKit
import CryptoKit

/// Capture only: raw depth + BGRA on disk, lightweight preview, post-scan upload.
final class DepthCaptureManager: NSObject, ObservableObject, AVCaptureDataOutputSynchronizerDelegate {
    @Published var isSessionRunning = false
    @Published var isScanning = false
    @Published var isBusy = false
    @Published var capturedFrames = 0
    @Published var droppedFrames = 0
    @Published var recordedMB = 0.0
    @Published var statusMessage = "Iniciando TrueDepth…"
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
    private var scanActive = false
    private var frameID: UInt32 = 0
    private var lastPreview = -Double.infinity
    private let lock = NSLock()
    private var processing = false
    private var dropped = 0
    private var captureGate = false // Protected by lock, including synchronizer callbacks.
    private var jpeg: Data?
    var latestJPEGData: Data? {
        lock.lock(); defer { lock.unlock() }
        return jpeg
    }

    override init() {
        super.init()
        streamClient.startDiscovery()
        CaptureRecorder.sealInterruptedSessions()
        uploader.onStatus = { [weak self] text, complete in
            DispatchQueue.main.async {
                self?.statusMessage = text
                if complete && self?.isScanning == false { self?.isBusy = false }
            }
        }
        uploader.start()
        checkPermissionsAndConfigure()
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
                    DispatchQueue.main.async {
                        self?.statusMessage = "Acceso a la cámara TrueDepth denegado"
                    }
                }
            }
        default:
            DispatchQueue.main.async {
                self.statusMessage = "Permiso de cámara no disponible"
            }
        }
    }
    
    private func configureSession() {
        captureSession.beginConfiguration()
        captureSession.sessionPreset = .vga640x480
        
        guard let device = AVCaptureDevice.default(.builtInTrueDepthCamera, for: .video, position: .front) else {
            DispatchQueue.main.async {
                self.statusMessage = "No se detectó sensor TrueDepth (Face ID)."
            }
            captureSession.commitConfiguration()
            return
        }
        
        do {
            let input = try AVCaptureDeviceInput(device: device)
            if captureSession.canAddInput(input) {
                captureSession.addInput(input)
            }
            
            // Respect the video format's supported rate; depth is usually 30 Hz.
            try device.lockForConfiguration()
            for range in device.activeFormat.videoSupportedFrameRateRanges {
                if range.minFrameRate <= 15 && range.maxFrameRate >= 15 {
                    let rate = 15.0
                    device.activeVideoMinFrameDuration = CMTime(value: 1, timescale: CMTimeScale(rate))
                    device.activeVideoMaxFrameDuration = CMTime(value: 1, timescale: CMTimeScale(rate))
                    break
                }
            }
            device.unlockForConfiguration()
        } catch {
            DispatchQueue.main.async {
                self.statusMessage = "Configuración sensor: \(error.localizedDescription)"
            }
        }
        
        if captureSession.canAddOutput(videoDataOutput) {
            videoDataOutput.videoSettings = [
                kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA)
            ]
            videoDataOutput.alwaysDiscardsLateVideoFrames = true
            captureSession.addOutput(videoDataOutput)
        }
        
        if captureSession.canAddOutput(depthDataOutput) {
            // Output raw, non-smoothed millimetric depth data (EinStar architecture)
            depthDataOutput.isFilteringEnabled = false
            captureSession.addOutput(depthDataOutput)
            if let connection = depthDataOutput.connection(with: .depthData) {
                connection.isEnabled = true
            }
        }
        
        captureSession.commitConfiguration()
        
        dataOutputSynchronizer = AVCaptureDataOutputSynchronizer(dataOutputs: [videoDataOutput, depthDataOutput])
        dataOutputSynchronizer?.setDelegate(self, queue: synchronizerQueue)
        
        captureSession.startRunning()
        DispatchQueue.main.async {
            self.isSessionRunning = true
            self.statusMessage = "TrueDepth listo. Pulsa 'INICIAR ESCANEO' o 2x Vol+"
        }
    }
    
    func toggleScan() {
        if isScanning { stopScan() } else { startScan() }
    }

    func startScan() {
        guard isSessionRunning, !isBusy, !isScanning else { return }
        isBusy = true
        processingQueue.async { [weak self] in
            guard let self = self else { return }
            do {
                try self.recorder.start()
                self.frameID = 0
                self.lock.lock(); self.dropped = 0; self.captureGate = true; self.lock.unlock()
                self.motionTracker.startTracking()
                self.scanActive = true
                DispatchQueue.main.async {
                    self.isBusy = false
                    self.isScanning = true
                    self.capturedFrames = 0
                    self.droppedFrames = 0
                    self.statusMessage = "Capturando PNG + profundidad sin pérdida. Bloques hacia la caché de Windows."
                }
            } catch { self.report(error.localizedDescription) }
        }
    }

    func stopScan() {
        guard isScanning, !isBusy else { return }
        isScanning = false
        isBusy = true
        processingQueue.async { [weak self] in
            guard let self = self, self.scanActive else { return }
            self.scanActive = false
            self.motionTracker.stopTracking()
            self.lock.lock(); self.captureGate = false; let lost = self.dropped; self.lock.unlock()
            do {
                let url = try self.recorder.finish(dropped: lost)
                let count = self.recorder.count
                DispatchQueue.main.async { self.capturedFrames = count; self.droppedFrames = lost }
                guard count > 0 else { try? FileManager.default.removeItem(at: url); self.report("No se obtuvieron frames TrueDepth. Revisa el sensor."); return }
                self.report("Finalizando bloques pendientes y verificando sesión en Windows…", busy: true)
                self.uploader.start()
            } catch { self.report("Error cerrando captura: \(error.localizedDescription)") }
        }
    }

    func retryTransfer() {
        uploader.start()
        statusMessage = "Reintento automático activo; los bloques confirmados no se vuelven a enviar."
    }

    private func report(_ text: String, busy: Bool = false) {
        DispatchQueue.main.async { [weak self] in
            self?.statusMessage = text
            self?.isBusy = busy
        }
    }

    func dataOutputSynchronizer(_ synchronizer: AVCaptureDataOutputSynchronizer,
                                didOutput synchronizedDataCollection: AVCaptureSynchronizedDataCollection) {
        lock.lock()
        if processing {
            if captureGate { dropped += 1 }
            lock.unlock()
            return
        }
        processing = true
        lock.unlock()
        processingQueue.async { [weak self] in
            guard let self = self else { return }
            defer { self.lock.lock(); self.processing = false; self.lock.unlock() }
            autoreleasepool {
                guard let d = synchronizedDataCollection.synchronizedData(for: self.depthDataOutput) as? AVCaptureSynchronizedDepthData,
                      let v = synchronizedDataCollection.synchronizedData(for: self.videoDataOutput) as? AVCaptureSynchronizedSampleBufferData,
                      !d.depthDataWasDropped, !v.sampleBufferWasDropped,
                      let pixel = CMSampleBufferGetImageBuffer(v.sampleBuffer) else {
                    if self.scanActive {
                        self.lock.lock(); self.dropped += 1; self.lock.unlock()
                    }
                    return
                }
                let timestamp = CMSampleBufferGetPresentationTimeStamp(v.sampleBuffer).seconds
                if self.scanActive {
                    do {
                        guard let packet = FrameSerializer.serialize(frameID: self.frameID, timestamp: timestamp,
                            depthData: d.depthData, colorPixelBuffer: pixel,
                            deviceMotion: self.motionTracker.latestDeviceMotion, rawCapture: true) else {
                            // Startup may temporarily provide depth without calibration.
                            // Skip this sample; it is not a fatal storage failure.
                            self.lock.lock(); self.dropped += 1; self.lock.unlock()
                            if self.recorder.count == 0 {
                                DispatchQueue.main.async { self.statusMessage = "Esperando frame TrueDepth calibrado…" }
                            }
                            return
                        }
                        try self.recorder.append(packet)
                        self.frameID &+= 1
                        self.lock.lock(); let missed = self.dropped; self.lock.unlock()
                        let total = self.recorder.count
                        let mb = Double(self.recorder.bytes) / 1048576
                        if total % 5 == 0 {
                            DispatchQueue.main.async {
                                self.capturedFrames = total; self.recordedMB = mb; self.droppedFrames = missed
                                if ProcessInfo.processInfo.thermalState == .critical {
                                    self.stopScan()
                                }
                            }
                        }
                    } catch {
                        DispatchQueue.main.async {
                            self.stopScan()
                            self.statusMessage = error.localizedDescription
                        }
                    }
                }
                // Five small JPEGs per second, independent of raw capture storage.
                if timestamp - self.lastPreview >= 0.2 {
                    self.lastPreview = timestamp
                    let image = CIImage(cvPixelBuffer: pixel).oriented(.right)
                    let small = image.transformed(by: CGAffineTransform(scaleX: 0.5, y: 0.5))
                    if let cg = self.ciContext.createCGImage(small, from: small.extent) {
                        let data = UIImage(cgImage: cg).jpegData(compressionQuality: 0.4)
                        self.lock.lock(); self.jpeg = data; self.lock.unlock()
                    }
                }
            }
        }
    }
}
