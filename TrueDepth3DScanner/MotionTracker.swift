import Foundation
import CoreMotion
import simd

/// High-rate 6DoF Visual-Inertial Motion Tracker.
/// Uses Apple's CoreMotion sensor fusion (IMU Gyroscope + Accelerometer) at 60 Hz
/// to anchor scanned 3D points rigidly in world space as the user moves around the object.
class MotionTracker: ObservableObject {
    private let motionManager = CMMotionManager()
    private let queue = OperationQueue()
    
    // Current 6DoF Camera Pose in World Coordinates: [R | t]
    private(set) var currentWorldTransform: simd_float4x4 = matrix_identity_float4x4
    private(set) var currentRotation: simd_float3x3 = matrix_identity_float3x3
    private(set) var currentTranslation: SIMD3<Float> = SIMD3<Float>(0, 0, 0)
    
    // Velocity and drift compensation
    private var velocity: SIMD3<Float> = SIMD3<Float>(0, 0, 0)
    private var lastTimestamp: TimeInterval = 0
    private var initialAttitudeMatrix: simd_float3x3?
    // Most recent raw CMDeviceMotion snapshot for frame synchronization
    private(set) var latestDeviceMotion: CMDeviceMotion?
    
    init() {
        queue.name = "com.truedepth.motionTracker"
        queue.qualityOfService = .userInteractive
        queue.maxConcurrentOperationCount = 1
    }
    
    func startTracking() {
        guard motionManager.isDeviceMotionAvailable else {
            print("Aviso: Sensor de movimiento CoreMotion no disponible.")
            return
        }
        
        motionManager.deviceMotionUpdateInterval = 1.0 / 60.0 // 60 FPS high-rate tracking
        motionManager.showsDeviceMovementDisplay = true
        
        // Use reference frame with gravity alignment (Z vertical)
        motionManager.startDeviceMotionUpdates(using: .xArbitraryZVertical, to: queue) { [weak self] motion, error in
            guard let self = self, let motion = motion, error == nil else { return }
            self.processMotion(motion)
        }
    }
    
    func stopTracking() {
        motionManager.stopDeviceMotionUpdates()
    }
    
    func resetAnchor() {
        queue.addOperation { [weak self] in
            self?.initialAttitudeMatrix = nil
            self?.currentTranslation = SIMD3<Float>(0, 0, 0)
            self?.velocity = SIMD3<Float>(0, 0, 0)
            self?.currentWorldTransform = matrix_identity_float4x4
            self?.currentRotation = matrix_identity_float3x3
            self?.lastTimestamp = 0
        }
    }
    
    private func processMotion(_ motion: CMDeviceMotion) {
        self.latestDeviceMotion = motion
        let attitude = motion.attitude
        let rotMat = attitude.rotationMatrix
        
        // Convert CoreMotion rotation matrix to SIMD 3x3
        // CMRotationMatrix format: m11, m12, m13 ...
        let R = simd_float3x3(
            SIMD3<Float>(Float(rotMat.m11), Float(rotMat.m21), Float(rotMat.m31)),
            SIMD3<Float>(Float(rotMat.m12), Float(rotMat.m22), Float(rotMat.m32)),
            SIMD3<Float>(Float(rotMat.m13), Float(rotMat.m23), Float(rotMat.m33))
        )
        
        if initialAttitudeMatrix == nil {
            initialAttitudeMatrix = R
            lastTimestamp = motion.timestamp
            return
        }
        
        guard let initR = initialAttitudeMatrix else { return }
        
        // Transform current device coordinates to initial anchor coordinates:
        // R_anchor = M0 * Mt^T
        let R_anchor = simd_mul(initR, simd_transpose(R))
        self.currentRotation = R_anchor
        
        // Integrate user acceleration (gravity compensated by Apple sensor fusion)
        let dt = Float(max(1e-4, min(0.05, motion.timestamp - lastTimestamp)))
        lastTimestamp = motion.timestamp
        
        let userAcc = motion.userAcceleration
        let accLocal = SIMD3<Float>(Float(userAcc.x), Float(userAcc.y), Float(userAcc.z)) * 9.80665 // m/s^2
        let accWorld = simd_mul(R_anchor, accLocal)
        
        // High-pass filtering / damping to prevent unconstrained double-integration drift in handheld scans
        let damping: Float = 0.94
        velocity = (velocity + accWorld * dt) * damping
        currentTranslation += velocity * dt
        
        // Build 4x4 Rigid Homogeneous Transformation Matrix [R | t]
        var T = simd_float4x4(R_anchor)
        T.columns.3 = SIMD4<Float>(currentTranslation.x, currentTranslation.y, currentTranslation.z, 1.0)
        self.currentWorldTransform = T
    }
    
    /// Rigidly transforms an optical pixel point (xM, yM, z in meters) into the static world anchor frame.
    /// Hardware convention: +X Right, +Y Up, +Z Forward.
    /// Output convention for CAD / OpenGL: +X Right, +Y Up, -Z Forward.
    func transformPointToWorld(xM: Float, yM: Float, z: Float) -> SIMD3<Float> {
        let pDev = SIMD3<Float>(xM, -yM, z)
        let pWorld = simd_mul(currentRotation, pDev) + currentTranslation
        return SIMD3<Float>(pWorld.x, pWorld.y, -pWorld.z)
    }
    
    /// Rigidly transforms a local point from camera coordinate frame into the static world reference frame.
    func transformPointToWorld(_ localPoint: SIMD3<Float>) -> SIMD3<Float> {
        return transformPointToWorld(xM: localPoint.x, yM: -localPoint.y, z: -localPoint.z)
    }
}
