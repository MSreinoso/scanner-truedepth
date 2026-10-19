import Foundation
import CryptoKit

/// A bounded, durable producer queue. Each binary WebSocket message is one frame.
/// ACK deletes only the corresponding local block; commit deletes its manifest.
final class ChunkStreamer {
    private var task: Task<Void, Never>?
    private let address: () -> String
    var onStatus: ((String, Bool) -> Void)?
    init(address: @escaping () -> String) { self.address = address }
    func start() {
        guard task == nil else { return }
        task = Task.detached(priority: .utility) { [weak self] in
            var delay: UInt64 = 1
            while !Task.isCancelled {
                guard let self = self else { return }
                do {
                    let folders = (try? FileManager.default.contentsOfDirectory(at: CaptureRecorder.directory, includingPropertiesForKeys: nil)) ?? []
                    if let folder = folders.sorted(by: { $0.path < $1.path }).first(where: {
                        FileManager.default.fileExists(atPath: $0.appendingPathComponent("session.json").path)
                    }) {
                        try await self.sendSession(folder)
                    } else { try await Task.sleep(nanoseconds: 500_000_000) }
                    delay = 1
                } catch {
                    self.onStatus?("Conexión interrumpida: reintento automático. Bloques pendientes conservados.", false)
                    try? await Task.sleep(nanoseconds: delay * 1_000_000_000)
                    delay = min(delay * 2, 15)
                }
            }
        }
    }
    private func reply(_ socket: URLSessionWebSocketTask) async throws -> [String: Any] {
        let message = try await socket.receive()
        let data: Data
        switch message { case .string(let text): data = Data(text.utf8); case .data(let value): data = value; @unknown default: throw CocoaError(.fileReadUnknown) }
        guard let value = try JSONSerialization.jsonObject(with: data) as? [String: Any], value["error"] == nil else { throw CocoaError(.fileReadCorruptFile) }
        return value
    }
    private func sendSession(_ folder: URL) async throws {
        let ip = address()
        guard !ip.isEmpty, let url = URL(string: "ws://\(ip):50508") else { throw URLError(.notConnectedToInternet) }
        let socket = URLSession.shared.webSocketTask(with: url)
        socket.maximumMessageSize = 9 * 1024 * 1024
        socket.resume()
        defer { socket.cancel(with: .normalClosure, reason: nil) }
        let id = folder.lastPathComponent
        let open = try JSONSerialization.data(withJSONObject: ["op": "open", "session": id, "version": 3])
        try await socket.send(.string(String(decoding: open, as: UTF8.self)))
        let hello = try await reply(socket)
        guard hello["op"] as? String == "opened", hello["session"] as? String == id else { throw CocoaError(.fileReadCorruptFile) }
        while !Task.isCancelled {
            let chunks = (try FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil))
                .filter { $0.pathExtension == "b3dc" }.sorted { $0.path < $1.path }
            if let chunk = chunks.first, let number = UInt32(chunk.deletingPathExtension().lastPathComponent) {
                let data = try Data(contentsOf: chunk)
                let digest = SHA256.hash(data: data)
                let hex = digest.map { String(format: "%02x", $0) }.joined()
                var seq = number.littleEndian
                var message = withUnsafeBytes(of: &seq) { Data($0) }
                message.append(contentsOf: digest); message.append(data)
                try await socket.send(.data(message))
                let ack = try await reply(socket)
                guard ack["op"] as? String == "ack", ack["session"] as? String == id,
                      (ack["seq"] as? NSNumber)?.uint32Value == number, ack["sha256"] as? String == hex else { throw CocoaError(.fileReadCorruptFile) }
                try FileManager.default.removeItem(at: chunk)
                onStatus?("Bloque \(number + 1) confirmado. Caché: \(CaptureRecorder.pendingBytes() / 1_000_000) MB.", false)
                continue
            }
            let data = try Data(contentsOf: folder.appendingPathComponent("session.json"))
            var manifest = try JSONSerialization.jsonObject(with: data) as! [String: Any]
            if manifest["finished"] as? Bool == true {
                manifest["op"] = "finish"
                let message = try JSONSerialization.data(withJSONObject: manifest)
                try await socket.send(.string(String(decoding: message, as: UTF8.self)))
                let receipt = try await reply(socket)
                guard receipt["op"] as? String == "committed", receipt["session"] as? String == id,
                      (receipt["frames"] as? NSNumber)?.intValue == (manifest["frames"] as? NSNumber)?.intValue else { throw CocoaError(.fileReadCorruptFile) }
                // Session directory contains only acknowledged chunks and metadata.
                guard (try FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil)).allSatisfy({ $0.lastPathComponent == "session.json" }) else { throw CocoaError(.fileWriteUnknown) }
                try FileManager.default.removeItem(at: folder)
                onStatus?("Sesión recibida y verificada en Windows. Caché local liberada.", true)
                return
            }
            try await Task.sleep(nanoseconds: 300_000_000)
        }
    }
    deinit { task?.cancel() }
}
