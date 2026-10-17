import Foundation
import AVFoundation
import CoreImage
import UIKit
import simd

/// Unique 3D spatial voxel grid coordinate key for deduplication.
struct VoxelKey: Hashable {
    let x: Int32
    let y: Int32
    let z: Int32
}

/// Canonical key for triangular facet deduplication across frames.
struct TriangleVoxelKey: Hashable {
    let k1: VoxelKey
    let k2: VoxelKey
    let k3: VoxelKey
    
    init(_ a: VoxelKey, _ b: VoxelKey, _ c: VoxelKey) {
        var arr = [a, b, c]
        arr.sort {
            if $0.x != $1.x { return $0.x < $1.x }
            if $0.y != $1.y { return $0.y < $1.y }
            return $0.z < $1.z
        }
        self.k1 = arr[0]
        self.k2 = arr[1]
        self.k3 = arr[2]
    }
}

/// High-performance continuous 3D scanner manager.
/// Streams frames at 30 FPS, continuously integrates multi-shot depth buffers into a spatial voxel grid,
/// performs temporal denoising, and outputs CAD binary STL / PLY models.
class DepthCaptureManager: NSObject, ObservableObject, AVCaptureDataOutputSynchronizerDelegate {
    @Published var isSessionRunning = false
    @Published var isScanning = false
    @Published var pointCount = 0
    @Published var triangleCount = 0
    @Published var statusMessage: String = "Iniciando sensor TrueDepth..."
    @Published var latestMesh: Mesh3D?
    
    // User-configurable parameters
    var maxDepthMeters: Float = 0.50
    var minDepthMeters: Float = 0.15
    var voxelSizeMeters: Float = 0.001 // 1.0 mm (metrological resolution)
    var enableSOR: Bool = true
    var basePlaneCutoffZ: Float? = nil
    
    // Cached buffers for preview and networking
    private(set) var latestJPEGData: Data?
    private(set) var latestColorPixelBuffer: CVPixelBuffer?
    private(set) var latestSTLURL: URL?
    private(set) var latestPLYURL: URL?
    private(set) var latestSTLData: Data?
    private(set) var latestPLYData: Data?
    private(set) var latestNewPointsJSON: Data?
    
    // Accumulated 3D Geometry
    private var voxelSet = Set<VoxelKey>()
    private var triangleSet = Set<TriangleVoxelKey>()
    private var accumulatedVertices: [SIMD3<Float>] = []
    private var accumulatedColors: [SIMD3<UInt8>] = []
    private var accumulatedTriangles: [Triangle3D] = []
    
    // Thread Isolation & Queues
    let captureSession = AVCaptureSession()
    private let sessionQueue = DispatchQueue(label: "com.truedepth.sessionQueue", qos: .userInitiated)
    private let synchronizerQueue = DispatchQueue(label: "com.truedepth.syncQueue", qos: .userInteractive)
    private let processingQueue = DispatchQueue(label: "com.truedepth.processingQueue", qos: .userInitiated)
    
    // Backpressure guard
    private var isProcessingFrame = false
    private let processLock = NSLock()
    
    private var dataOutputSynchronizer: AVCaptureDataOutputSynchronizer?
    private let videoDataOutput = AVCaptureVideoDataOutput()
    private let depthDataOutput = AVCaptureDepthDataOutput()
    private let ciContext = CIContext(options: [.useSoftwareRenderer: false])
    private var frameCounter: Int = 0
    
    // 6DoF Inertial Motion Tracker (60 Hz IMU fusion for spatial anchoring)
    let motionTracker = MotionTracker()
    
    // Persistent TCP Binary Streaming Client (EinStar architecture)
    let streamClient = BinaryStreamClient()
    private var streamFrameID: UInt32 = 0
    
    override init() {
        super.init()
        streamClient.startDiscovery()
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
            
            // Frame rate configuration: lock to 60 FPS for high-rate continuous stream
            try device.lockForConfiguration()
            for range in device.activeFormat.videoSupportedFrameRateRanges {
                if range.maxFrameRate >= 59.0 {
                    device.activeVideoMinFrameDuration = CMTime(value: 1, timescale: 60)
                    device.activeVideoMaxFrameDuration = CMTime(value: 1, timescale: 60)
                    break
                } else if range.maxFrameRate >= 29.0 {
                    let rate = min(60.0, range.maxFrameRate)
                    device.activeVideoMinFrameDuration = CMTime(value: 1, timescale: CMTimeScale(rate))
                    device.activeVideoMaxFrameDuration = CMTime(value: 1, timescale: CMTimeScale(rate))
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
    
    // MARK: - Continuous Scan State Machine
    func toggleScan() {
        if isScanning {
            stopScan()
        } else {
            startScan()
        }
    }
    
    func startScan() {
        guard isSessionRunning else { return }
        let generator = UINotificationFeedbackGenerator()
        generator.notificationOccurred(.success)
        
        // Lock 6DoF spatial anchor to current camera origin and start 60 Hz inertial tracking
        motionTracker.resetAnchor()
        motionTracker.startTracking()
        streamFrameID = 0
        
        processingQueue.async { [weak self] in
            guard let self = self else { return }
            DispatchQueue.main.async {
                self.isScanning = true
                self.statusMessage = "GRABANDO ESCANEO 3D CONTINUO (60 FPS)..."
            }
        }
    }
    
    func stopScan() {
        let generator = UINotificationFeedbackGenerator()
        generator.notificationOccurred(.warning)
        
        // Stop motion sensor updates
        motionTracker.stopTracking()
        
        processingQueue.async { [weak self] in
            guard let self = self else { return }
            
            // Build final Mesh3D from accumulated geometry
            let mesh = Mesh3D(
                vertices: self.accumulatedVertices,
                vertexColors: self.accumulatedColors,
                normals: [],
                triangles: self.accumulatedTriangles
            )
            
            let stlUrl = try? STLExporter.exportToBinarySTL(mesh: mesh)
            let plyUrl = try? PLYExporter.exportToBinaryPLY(mesh: mesh)
            let stlBytes = stlUrl != nil ? try? Data(contentsOf: stlUrl!) : nil
            let plyBytes = plyUrl != nil ? try? Data(contentsOf: plyUrl!) : nil
            
            DispatchQueue.main.async {
                self.isScanning = false
                self.latestMesh = mesh
                self.latestSTLURL = stlUrl
                self.latestPLYURL = plyUrl
                self.latestSTLData = stlBytes
                self.latestPLYData = plyBytes
                self.statusMessage = "Escaneo completado: \(self.accumulatedVertices.count) pts, \(self.accumulatedTriangles.count) triángulos."
            }
        }
    }
    
    func clearScan() {
        motionTracker.resetAnchor()
        processingQueue.async { [weak self] in
            guard let self = self else { return }
            self.voxelSet.removeAll()
            self.triangleSet.removeAll()
            self.accumulatedVertices.removeAll()
            self.accumulatedColors.removeAll()
            self.accumulatedTriangles.removeAll()
            self.latestNewPointsJSON = nil
            self.latestSTLURL = nil
            self.latestPLYURL = nil
            self.latestSTLData = nil
            self.latestPLYData = nil
            
            DispatchQueue.main.async {
                self.pointCount = 0
                self.triangleCount = 0
                self.latestMesh = nil
                self.statusMessage = "Escaneo limpiado. Listo para nueva captura."
            }
        }
    }
    
    // MARK: - AVCaptureDataOutputSynchronizerDelegate
    func dataOutputSynchronizer(_ synchronizer: AVCaptureDataOutputSynchronizer,
                                didOutput synchronizedDataCollection: AVCaptureSynchronizedDataCollection) {
        processLock.lock()
        if isProcessingFrame {
            processLock.unlock()
            return
        }
        isProcessingFrame = true
        processLock.unlock()
        
        processingQueue.async { [weak self] in
            defer {
                self?.processLock.lock()
                self?.isProcessingFrame = false
                self?.processLock.unlock()
            }
            
            guard let self = self else { return }
            
            // Extract synchronized Depth Buffer
            guard let synchDepthData = synchronizedDataCollection.synchronizedData(for: self.depthDataOutput) as? AVCaptureSynchronizedDepthData,
                  !synchDepthData.depthDataWasDropped else {
                return
            }
            
            let depthData = synchDepthData.depthData
            
            // Extract synchronized Video Buffer
            if let synchVideoData = synchronizedDataCollection.synchronizedData(for: self.videoDataOutput) as? AVCaptureSynchronizedSampleBufferData,
               !synchVideoData.sampleBufferWasDropped,
               let pixelBuffer = CMSampleBufferGetImageBuffer(synchVideoData.sampleBuffer) {
                
                self.latestColorPixelBuffer = pixelBuffer
                
                // Generate streaming JPEG for live preview
                let ciImage = CIImage(cvPixelBuffer: pixelBuffer).oriented(.right)
                if let cgImage = self.ciContext.createCGImage(ciImage, from: ciImage.extent) {
                    let uiImage = UIImage(cgImage: cgImage)
                    self.latestJPEGData = uiImage.jpegData(compressionQuality: 0.5)
                }
            }
            
            // State B: Active Video Scan -> Stream synchronized raw 30 FPS frames to PC
            if self.isScanning {
                let timestamp = CMSampleBufferGetPresentationTimeStamp(synchVideoData.sampleBuffer).seconds
                let motion = self.motionTracker.latestDeviceMotion
                
                if let packet = FrameSerializer.serialize(
                    frameID: self.streamFrameID,
                    timestamp: timestamp,
                    depthData: depthData,
                    colorPixelBuffer: pixelBuffer,
                    deviceMotion: motion
                ) {
                    self.streamClient.sendFrame(packet)
                    self.streamFrameID += 1
                }
                
                // Also integrate locally for device preview
                self.integrateDepthFrame(depthData: depthData, colorBuffer: self.latestColorPixelBuffer)
            }
        }
    }
    
    // MARK: - Continuous Voxel Grid & Surface Integration
    private func integrateDepthFrame(depthData: AVDepthData, colorBuffer: CVPixelBuffer?) {
        let depthDataConverted: AVDepthData = (depthData.depthDataType != kCVPixelFormatType_DepthFloat32) ?
            depthData.converting(toDepthDataType: kCVPixelFormatType_DepthFloat32) : depthData
        
        let depthBuffer = depthDataConverted.depthDataMap
        CVPixelBufferLockBaseAddress(depthBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depthBuffer, .readOnly) }
        
        guard let depthBaseAddress = CVPixelBufferGetBaseAddress(depthBuffer) else { return }
        let depthWidth = CVPixelBufferGetWidth(depthBuffer)
        let depthHeight = CVPixelBufferGetHeight(depthBuffer)
        let depthFloatBuffer = depthBaseAddress.assumingMemoryBound(to: Float32.self)
        
        // Lock color buffer if available
        var colorBaseAddress: UnsafeMutableRawPointer? = nil
        var colorBytesPerRow: Int = 0
        var colorWidth = 0, colorHeight = 0
        if let colorBuf = colorBuffer {
            CVPixelBufferLockBaseAddress(colorBuf, .readOnly)
            colorBaseAddress = CVPixelBufferGetBaseAddress(colorBuf)
            colorBytesPerRow = CVPixelBufferGetBytesPerRow(colorBuf)
            colorWidth = CVPixelBufferGetWidth(colorBuf)
            colorHeight = CVPixelBufferGetHeight(colorBuf)
        }
        defer {
            if let colorBuf = colorBuffer { CVPixelBufferUnlockBaseAddress(colorBuf, .readOnly) }
        }
        
        // Intrinsic parameters
        var fx: Float = Float(depthWidth), fy: Float = Float(depthWidth)
        var cx: Float = Float(depthWidth) / 2.0, cy: Float = Float(depthHeight) / 2.0
        
        if let calibration = depthDataConverted.cameraCalibrationData {
            let intrinsicMatrix = calibration.intrinsicMatrix
            let referenceSize = calibration.intrinsicMatrixReferenceDimensions
            let scaleX = Float(depthWidth) / Float(referenceSize.width)
            let scaleY = Float(depthHeight) / Float(referenceSize.height)
            fx = intrinsicMatrix.columns.0.x * scaleX
            fy = intrinsicMatrix.columns.1.y * scaleY
            cx = intrinsicMatrix.columns.2.x * scaleX
            cy = intrinsicMatrix.columns.2.y * scaleY
        }
        
        let step = (depthWidth > 400) ? 2 : 1
        let gridCols = depthWidth / step
        let gridRows = depthHeight / step
        let voxelSize = self.voxelSizeMeters
        let maxD = self.maxDepthMeters
        let minD = self.minDepthMeters
        let maxEdgeDiscontinuity: Float = 0.015 // 15 mm
        
        var frameGridPoints = Array(repeating: Array<SIMD3<Float>?>(repeating: nil, count: gridCols), count: gridRows)
        var frameGridKeys = Array(repeating: Array<VoxelKey?>(repeating: nil, count: gridCols), count: gridRows)
        var frameGridColors = Array(repeating: Array<SIMD3<UInt8>>(repeating: SIMD3<UInt8>(180, 180, 180), count: gridCols), count: gridRows)
        var newPointsBatch: [[Float]] = []
        
        for r in 0..<gridRows {
            let yPixel = r * step
            let rowOffset = yPixel * depthWidth
            
            for c in 0..<gridCols {
                let xPixel = c * step
                let rawZ = depthFloatBuffer[rowOffset + xPixel]
                
                if rawZ.isNaN || rawZ.isInfinite || rawZ < minD || rawZ > maxD { continue }
                if let planeLimit = self.basePlaneCutoffZ, rawZ >= planeLimit { continue }
                
                // Sample Color at (xPixel, yPixel) from video frame
                var vertexColor = SIMD3<UInt8>(180, 180, 180)
                if let base = colorBaseAddress, colorWidth > 0, colorHeight > 0 {
                    let cX = Int((Float(xPixel) / Float(depthWidth)) * Float(colorWidth))
                    let cY = Int((Float(yPixel) / Float(depthHeight)) * Float(colorHeight))
                    if cX >= 0 && cX < colorWidth && cY >= 0 && cY < colorHeight {
                        let offset = cY * colorBytesPerRow + cX * 4
                        let bPtr = base.advanced(by: offset).assumingMemoryBound(to: UInt8.self)
                        let byteB = bPtr[0], byteG = bPtr[1], byteR = bPtr[2]
                        vertexColor = SIMD3<UInt8>(byteR, byteG, byteB)
                    }
                }
                
                // Bilateral Edge-Preserving Depth Smoothing & Anti-Spike Filter
                var smoothedZ = rawZ
                if c > 0 && c < gridCols - 1 && r > 0 && r < gridRows - 1 {
                    let zL = depthFloatBuffer[rowOffset + (c - 1) * step]
                    let zR = depthFloatBuffer[rowOffset + (c + 1) * step]
                    let zU = depthFloatBuffer[(r - 1) * step * depthWidth + xPixel]
                    let zD = depthFloatBuffer[(r + 1) * step * depthWidth + xPixel]
                    
                    let validL = !zL.isNaN && zL >= minD && zL <= maxD
                    let validR = !zR.isNaN && zR >= minD && zR <= maxD
                    let validU = !zU.isNaN && zU >= minD && zU <= maxD
                    let validD = !zD.isNaN && zD >= minD && zD <= maxD
                    
                    // Spike rejection: isolate and suppress spurious IR reflection peaks
                    var isSpike = false
                    if validL && validR && abs(zL - zR) < 0.008 && abs(rawZ - (zL + zR) * 0.5) > 0.012 {
                        smoothedZ = (zL + zR) * 0.5
                        isSpike = true
                    } else if validU && validD && abs(zU - zD) < 0.008 && abs(rawZ - (zU + zD) * 0.5) > 0.012 {
                        smoothedZ = (zU + zD) * 0.5
                        isSpike = true
                    }
                    
                    // Guided bilateral smoothing across contiguous surface patches
                    if !isSpike {
                        var sumZ: Float = rawZ * 2.0
                        var sumW: Float = 2.0
                        let neighbors = [(validL, zL), (validR, zR), (validU, zU), (validD, zD)]
                        for (valid, nZ) in neighbors {
                            if valid && abs(rawZ - nZ) < 0.015 {
                                sumZ += nZ
                                sumW += 1.0
                            }
                        }
                        smoothedZ = sumZ / sumW
                    }
                }
                
                let xM = (Float(xPixel) - cx) * smoothedZ / fx
                let yM = (Float(yPixel) - cy) * smoothedZ / fy
                
                // Rigid 6DoF World Anchor Transformation:
                // Anchors the 3D model statically in world coordinates as the camera moves around it
                let pt = self.motionTracker.transformPointToWorld(xM: xM, yM: yM, z: smoothedZ)
                
                // Spatial Voxel Hash Key (in world meters)
                let vKey = VoxelKey(
                    x: Int32(floor(pt.x / voxelSize)),
                    y: Int32(floor(pt.y / voxelSize)),
                    z: Int32(floor(pt.z / voxelSize))
                )
                
                frameGridPoints[r][c] = pt
                frameGridKeys[r][c] = vKey
                frameGridColors[r][c] = vertexColor
                
                // Deduplicate Vertices
                if self.voxelSet.insert(vKey).inserted {
                    self.accumulatedVertices.append(pt)
                    self.accumulatedColors.append(vertexColor)
                    newPointsBatch.append([pt.x * 1000.0, pt.y * 1000.0, pt.z * 1000.0])
                }
            }
        }
        
        // Continuous Triangle Mesh Integration across the voxel grid
        for r in 0..<(gridRows - 1) {
            for c in 0..<(gridCols - 1) {
                guard let p00 = frameGridPoints[r][c], let k00 = frameGridKeys[r][c],
                      let p10 = frameGridPoints[r + 1][c], let k10 = frameGridKeys[r + 1][c],
                      let p01 = frameGridPoints[r][c + 1], let k01 = frameGridKeys[r][c + 1],
                      let p11 = frameGridPoints[r + 1][c + 1], let k11 = frameGridKeys[r + 1][c + 1] else { continue }
                
                let c00 = frameGridColors[r][c], c10 = frameGridColors[r + 1][c], c01 = frameGridColors[r][c + 1], c11 = frameGridColors[r + 1][c + 1]
                
                // Triangle 1: p00 -> p10 -> p01
                if k00 != k10 && k10 != k01 && k01 != k00 {
                    let triKey1 = TriangleVoxelKey(k00, k10, k01)
                    if self.triangleSet.insert(triKey1).inserted {
                        if simd_distance(p00, p10) <= maxEdgeDiscontinuity && simd_distance(p10, p01) <= maxEdgeDiscontinuity {
                            let normal = simd_cross(p10 - p00, p01 - p00)
                            let len = simd_length(normal)
                            if len > 1e-6 {
                                self.accumulatedTriangles.append(Triangle3D(v1: p00, v2: p10, v3: p01, normal: normal / len, c1: c00, c2: c10, c3: c01))
                            }
                        }
                    }
                }
                
                // Triangle 2: p10 -> p11 -> p01
                if k10 != k11 && k11 != k01 && k01 != k10 {
                    let triKey2 = TriangleVoxelKey(k10, k11, k01)
                    if self.triangleSet.insert(triKey2).inserted {
                        if simd_distance(p10, p11) <= maxEdgeDiscontinuity && simd_distance(p11, p01) <= maxEdgeDiscontinuity {
                            let normal = simd_cross(p11 - p10, p01 - p10)
                            let len = simd_length(normal)
                            if len > 1e-6 {
                                self.accumulatedTriangles.append(Triangle3D(v1: p10, v2: p11, v3: p01, normal: normal / len, c1: c10, c2: c11, c3: c01))
                            }
                        }
                    }
                }
            }
        }
        
        // Telemetry Update for PC Streaming & UI
        frameCounter += 1
        if !newPointsBatch.isEmpty && frameCounter % 3 == 0 {
            // Sample for network stream
            let sampled = newPointsBatch.enumerated().compactMap { i, p in i % 2 == 0 ? p : nil }
            if let json = try? JSONSerialization.data(withJSONObject: sampled) {
                self.latestNewPointsJSON = json
            }
        }
        
        let totalPts = self.accumulatedVertices.count
        let totalTris = self.accumulatedTriangles.count
        DispatchQueue.main.async {
            self.pointCount = totalPts
            self.triangleCount = totalTris
        }
    }
}
