import Foundation
import Combine
import CoreMotion

/// Orientation samples only. No inertial translation or point-cloud processing on iOS.
final class MotionTracker: ObservableObject {
    private let manager = CMMotionManager()
    private let queue = OperationQueue()
    private let lock = NSLock()
    private var sample: CMDeviceMotion?
    var latestDeviceMotion: CMDeviceMotion? {
        lock.lock(); defer { lock.unlock() }; return sample
    }
    init() {
        queue.maxConcurrentOperationCount = 1
        queue.qualityOfService = .utility
    }
    func startTracking() {
        guard manager.isDeviceMotionAvailable else { return }
        manager.deviceMotionUpdateInterval = 1.0 / 30
        manager.startDeviceMotionUpdates(using: .xArbitraryZVertical, to: queue) { [weak self] motion, _ in
            guard let self = self else { return }
            self.lock.lock(); self.sample = motion; self.lock.unlock()
        }
    }
    func stopTracking() { manager.stopDeviceMotionUpdates() }
    deinit { manager.stopDeviceMotionUpdates() }
}
