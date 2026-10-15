import Foundation
import AVFoundation
import CoreImage
import UIKit

class DepthCaptureManager: NSObject, ObservableObject, AVCaptureDataOutputSynchronizerDelegate {
    @Published var isSessionRunning = false
    @Published var lastCapturedDepthData: AVDepthData?
    @Published var statusMessage: String = "Iniciando cámara..."
    
    var latestJPEGData: Data?
    private let ciContext = CIContext()
    
    let captureSession = AVCaptureSession()
    private let sessionQueue = DispatchQueue(label: "com.truedepth.scanner.sessionQueue")
    private var dataOutputSynchronizer: AVCaptureDataOutputSynchronizer?
    
    private let videoDataOutput = AVCaptureVideoDataOutput()
    private let depthDataOutput = AVCaptureDepthDataOutput()
    
    override init() {
        super.init()
        checkPermissionsAndConfigure()
    }
    
    private func checkPermissionsAndConfigure() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            sessionQueue.async { self.configureSession() }
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { granted in
                if granted {
                    self.sessionQueue.async { self.configureSession() }
                } else {
                    DispatchQueue.main.async {
                        self.statusMessage = "Acceso a la cámara denegado"
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
        } catch {
            DispatchQueue.main.async {
                self.statusMessage = "Error al conectar cámara: \(error.localizedDescription)"
            }
            captureSession.commitConfiguration()
            return
        }
        
        if captureSession.canAddOutput(videoDataOutput) {
            videoDataOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA)]
            captureSession.addOutput(videoDataOutput)
        }
        
        if captureSession.canAddOutput(depthDataOutput) {
            depthDataOutput.isFilteringEnabled = true
            captureSession.addOutput(depthDataOutput)
            if let connection = depthDataOutput.connection(with: .depthData) {
                connection.isEnabled = true
            }
        }
        
        captureSession.commitConfiguration()
        
        dataOutputSynchronizer = AVCaptureDataOutputSynchronizer(dataOutputs: [videoDataOutput, depthDataOutput])
        dataOutputSynchronizer?.setDelegate(self, queue: sessionQueue)
        
        captureSession.startRunning()
        DispatchQueue.main.async {
            self.isSessionRunning = true
            self.statusMessage = "TrueDepth listo (20 - 50 cm)"
        }
    }
    
    func dataOutputSynchronizer(_ synchronizer: AVCaptureDataOutputSynchronizer, didOutput synchronizedDataCollection: AVCaptureSynchronizedDataCollection) {
        if let synchDepthData = synchronizedDataCollection.synchronizedData(for: depthDataOutput) as? AVCaptureSynchronizedDepthData,
           !synchDepthData.depthDataWasDropped {
            let depthData = synchDepthData.depthData
            
            if let synchVideoData = synchronizedDataCollection.synchronizedData(for: videoDataOutput) as? AVCaptureSynchronizedSampleBufferData,
               !synchVideoData.sampleBufferWasDropped,
               let pixelBuffer = CMSampleBufferGetImageBuffer(synchVideoData.sampleBuffer) {
                
                let ciImage = CIImage(cvPixelBuffer: pixelBuffer).oriented(.right)
                if let cgImage = ciContext.createCGImage(ciImage, from: ciImage.extent) {
                    let uiImage = UIImage(cgImage: cgImage)
                    self.latestJPEGData = uiImage.jpegData(compressionQuality: 0.5)
                }
            }
            
            DispatchQueue.main.async {
                self.lastCapturedDepthData = depthData
            }
        }
    }
}
