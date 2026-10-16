import Foundation
import AVFoundation
import MediaPlayer
import UIKit

/// Manages hardware volume button triggers without requiring physical touchscreen contact.
/// Listens for double-press on Volume Up and fires a haptic feedback signal.
class VolumeButtonTriggerManager: NSObject, ObservableObject {
    private var audioSession = AVAudioSession.sharedInstance()
    private var volumeObserver: NSKeyValueObservation?
    private var lastPressTime: TimeInterval = 0
    private var previousVolume: Float = 0.5
    private let doubleClickThreshold: TimeInterval = 0.45
    
    var onTriggerAction: (() -> Void)?
    
    override init() {
        super.init()
        setupAudioSession()
        setupVolumeListener()
    }
    
    deinit {
        volumeObserver?.invalidate()
    }
    
    private func setupAudioSession() {
        do {
            try audioSession.setCategory(.ambient, options: .mixWithOthers)
            try audioSession.setActive(true)
            previousVolume = audioSession.outputVolume
        } catch {
            print("Error configurando AVAudioSession para disparador de volumen: \(error)")
        }
    }
    
    private func setupVolumeListener() {
        volumeObserver = audioSession.observe(\.outputVolume, options: [.new]) { [weak self] session, change in
            guard let self = self else { return }
            let currentVolume = session.outputVolume
            let currentTime = ProcessInfo.processInfo.systemUptime
            
            // Check if volume increased (Volume Up pressed)
            if currentVolume > self.previousVolume || (self.previousVolume >= 0.99 && currentVolume >= 0.99) {
                let interval = currentTime - self.lastPressTime
                if interval <= self.doubleClickThreshold && interval > 0.05 {
                    // Double Click Detected!
                    DispatchQueue.main.async {
                        let generator = UINotificationFeedbackGenerator()
                        generator.notificationOccurred(.success)
                        self.onTriggerAction?()
                    }
                    self.lastPressTime = 0
                } else {
                    self.lastPressTime = currentTime
                }
            }
            
            self.previousVolume = currentVolume
        }
    }
}
