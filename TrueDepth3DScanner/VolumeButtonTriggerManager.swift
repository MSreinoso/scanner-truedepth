import Foundation
import AVFoundation
import MediaPlayer
import UIKit

/// Manages physical 'Volume Up' button events to act as the global scanning toggle.
/// Uses AVAudioSession with a background silent audio loop and an invisible MPVolumeView
/// to continuously capture hardware clicks without freezing the UI or displaying the native iOS volume overlay.
/// Enforces a strict 350ms double-tap debounce threshold.
class VolumeButtonTriggerManager: NSObject, ObservableObject {
    private var audioSession = AVAudioSession.sharedInstance()
    private var volumeObserver: NSKeyValueObservation?
    private var silentPlayer: AVAudioPlayer?
    private var volumeView: MPVolumeView?
    
    private var lastPressTime: TimeInterval = 0
    private var previousVolume: Float = 0.5
    private let doubleClickThreshold: TimeInterval = 0.35 // Strict 350ms threshold
    
    var onTriggerAction: (() -> Void)?
    
    override init() {
        super.init()
        setupAudioSession()
        startSilentAudioLoop()
        setupVolumeHUDSuppression()
        setupVolumeListener()
    }
    
    deinit {
        volumeObserver?.invalidate()
        silentPlayer?.stop()
        volumeView?.removeFromSuperview()
    }
    
    private func setupAudioSession() {
        do {
            try audioSession.setCategory(.playback, options: [.mixWithOthers])
            try audioSession.setActive(true)
            previousVolume = audioSession.outputVolume
        } catch {
            print("AVAudioSession error: \(error)")
        }
    }
    
    private func startSilentAudioLoop() {
        let wavData = Self.createSilentWAVData()
        do {
            let player = try AVAudioPlayer(data: wavData)
            player.numberOfLoops = -1 // Infinite background loop
            player.volume = 0.001
            player.prepareToPlay()
            player.play()
            self.silentPlayer = player
        } catch {
            print("Error starting silent audio loop: \(error)")
        }
    }
    
    private func setupVolumeHUDSuppression() {
        DispatchQueue.main.async {
            let view = MPVolumeView(frame: CGRect(x: -200, y: -200, width: 1, height: 1))
            view.clipsToBounds = true
            view.alpha = 0.01
            if let window = UIApplication.shared.connectedScenes
                .compactMap({ $0 as? UIWindowScene })
                .flatMap({ $0.windows })
                .first(where: { $0.isKeyWindow }) {
                window.addSubview(view)
                self.volumeView = view
            }
        }
    }
    
    private func setupVolumeListener() {
        volumeObserver = audioSession.observe(\.outputVolume, options: [.new]) { [weak self] session, change in
            guard let self = self else { return }
            let currentVolume = session.outputVolume
            let currentTime = ProcessInfo.processInfo.systemUptime
            
            // Check if Volume Up was pressed (volume increased or held at max)
            if currentVolume > self.previousVolume || (self.previousVolume >= 0.95 && currentVolume >= 0.95) {
                let interval = currentTime - self.lastPressTime
                
                // Double-click check: strictly within 350ms window
                if interval <= self.doubleClickThreshold && interval > 0.04 {
                    // Double-Tap Detected!
                    DispatchQueue.main.async {
                        let generator = UINotificationFeedbackGenerator()
                        generator.notificationOccurred(.success)
                        self.onTriggerAction?()
                    }
                    self.lastPressTime = 0
                } else {
                    self.lastPressTime = currentTime
                }
                
                // Reset volume slider towards 0.5 to keep range open for subsequent clicks
                self.resetVolumeToCenter()
            }
            
            self.previousVolume = currentVolume
        }
    }
    
    private func resetVolumeToCenter() {
        guard let view = volumeView else { return }
        for subview in view.subviews {
            if let slider = subview as? UISlider {
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.05) {
                    slider.value = 0.5
                }
                break
            }
        }
    }
    
    /// Generates 1-second uncompressed silent 16-bit PCM WAV in memory
    private static func createSilentWAVData() -> Data {
        let sampleRate: Int32 = 44100
        let numChannels: Int16 = 1
        let bitsPerSample: Int16 = 16
        let numSamples: Int32 = sampleRate
        let blockAlign = numChannels * (bitsPerSample / 8)
        let dataSize = numSamples * Int32(blockAlign)
        
        var data = Data()
        data.append(contentsOf: "RIFF".utf8)
        var chunkSize = 36 + dataSize
        withUnsafeBytes(of: &chunkSize) { data.append(contentsOf: $0) }
        data.append(contentsOf: "WAVEfmt ".utf8)
        var subchunk1Size: Int32 = 16
        withUnsafeBytes(of: &subchunk1Size) { data.append(contentsOf: $0) }
        var audioFormat: Int16 = 1 // PCM
        withUnsafeBytes(of: &audioFormat) { data.append(contentsOf: $0) }
        var channels = numChannels
        withUnsafeBytes(of: &channels) { data.append(contentsOf: $0) }
        var sRate = sampleRate
        withUnsafeBytes(of: &sRate) { data.append(contentsOf: $0) }
        var bRate = sampleRate * Int32(blockAlign)
        withUnsafeBytes(of: &bRate) { data.append(contentsOf: $0) }
        var bAlign = blockAlign
        withUnsafeBytes(of: &bAlign) { data.append(contentsOf: $0) }
        var bps = bitsPerSample
        withUnsafeBytes(of: &bps) { data.append(contentsOf: $0) }
        data.append(contentsOf: "data".utf8)
        var dSize = dataSize
        withUnsafeBytes(of: &dSize) { data.append(contentsOf: $0) }
        data.append(contentsOf: [UInt8](repeating: 0, count: Int(dataSize)))
        return data
    }
}
