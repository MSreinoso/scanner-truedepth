import Foundation
import CryptoKit

/// Durable, bounded producer queue. Each binary WebSocket message carries one
/// block; a block is deleted on the iPhone only after Windows answers with an
/// ACK whose session, sequence and SHA-256 match (Windows ACKs after fsync).
final class ChunkStreamer {
    /// Windows answered with an explicit error (data or storage), not a network fault.
    struct RemoteError: LocalizedError {
        let message: String
        var errorDescription: String? { "Windows rechazó la sesión: \(message)" }
    }
    struct Timeout: LocalizedError {
        var errorDescription: String? { "Windows no respondió a tiempo" }
    }
    struct NoAddress: LocalizedError {
        var errorDescription: String? { "Sin IP de Windows: conecta por descubrimiento automático o escribe la IP manualmente" }
    }

    enum Kind: String { case idle, waitingWindows, connecting, sending, committed, rejected, networkError }

    private var task: Task<Void, Never>?
    private let address: () -> String
    private let lock = NSLock()
    private var lastAck = Date.distantPast
    private var state: (Kind, String) = (.idle, "Sin envíos pendientes")
    private var failures: [String: (String, Int)] = [:]
    var port = 50508
    var window = 6
    var replyTimeout: Double = 20
    /// kind, user text. Called from a background task.
    var onStatus: ((Kind, String) -> Void)?

    init(address: @escaping () -> String) { self.address = address }

    var lastAckDate: Date { lock.lock(); defer { lock.unlock() }; return lastAck }
    var current: (kind: Kind, text: String) { lock.lock(); defer { lock.unlock() }; return (kind: state.0, text: state.1) }

    private func publish(_ kind: Kind, _ text: String) {
        lock.lock(); state = (kind, text); lock.unlock()
        onStatus?(kind, text)
    }

    func start() {
        lock.lock(); defer { lock.unlock() }
        guard task == nil else { return }
        task = Task.detached(priority: .utility) { [weak self] in
            var delay: UInt64 = 1
            while !Task.isCancelled {
                guard let self = self else { return }
                guard let folder = self.nextSession() else {
                    self.publishIdle()
                    try? await Task.sleep(nanoseconds: 500_000_000)
                    continue
                }
                do {
                    try await self.sendSession(folder)
                    delay = 1
                } catch let error as RemoteError {
                    self.recordRejection(folder, error.message)
                    try? await Task.sleep(nanoseconds: delay * 1_000_000_000)
                    delay = min(delay * 2, 15)
                } catch is NoAddress {
                    self.publish(.waitingWindows, "Esperando a Windows: bloques conservados en el iPhone. \(NoAddress().errorDescription ?? "")")
                    try? await Task.sleep(nanoseconds: 1_000_000_000)
                } catch {
                    self.publish(.networkError, "Conexión con Windows interrumpida (\(error.localizedDescription)). Reintento automático; ningún bloque pendiente se borra.")
                    try? await Task.sleep(nanoseconds: delay * 1_000_000_000)
                    delay = min(delay * 2, 15)
                }
            }
        }
    }

    /// Clears parked sessions so they are offered to Windows again.
    func retryRejected() {
        lock.lock(); failures.removeAll(); lock.unlock()
        let folders = (try? FileManager.default.contentsOfDirectory(at: CaptureRecorder.directory, includingPropertiesForKeys: nil)) ?? []
        for folder in folders {
            let path = folder.appendingPathComponent("session.json")
            guard let data = try? Data(contentsOf: path),
                  var value = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
                  value["rejected"] != nil else { continue }
            value.removeValue(forKey: "rejected")
            if let contents = try? JSONSerialization.data(withJSONObject: value) { try? contents.write(to: path, options: .atomic) }
        }
    }

    private func publishIdle() {
        let pending = CaptureRecorder.summary()
        if pending.rejected > 0 {
            publish(.rejected, "\(pending.rejected) sesión(es) apartada(s) tras rechazos repetidos de Windows. Se conservan; usa Reintentar envío.")
        } else if current.kind != .committed {
            publish(.idle, "Sin envíos pendientes")
        }
    }

    /// Oldest unrejected session first: draining old sessions frees cache for capture.
    private func nextSession() -> URL? {
        let folders = (try? FileManager.default.contentsOfDirectory(at: CaptureRecorder.directory,
            includingPropertiesForKeys: [.creationDateKey])) ?? []
        let candidates = folders.filter { folder in
            let path = folder.appendingPathComponent("session.json")
            guard let data = try? Data(contentsOf: path),
                  let value = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else { return false }
            return value["rejected"] == nil
        }
        return candidates.min { a, b in
            let da = (try? a.resourceValues(forKeys: [.creationDateKey]).creationDate) ?? .distantPast
            let db = (try? b.resourceValues(forKeys: [.creationDateKey]).creationDate) ?? .distantPast
            return da == db ? a.path < b.path : da < db
        }
    }

    private func recordRejection(_ folder: URL, _ message: String) {
        let id = folder.lastPathComponent
        lock.lock()
        let previous = failures[id]
        let count = previous?.0 == message ? previous!.1 + 1 : 1
        failures[id] = (message, count)
        lock.unlock()
        if count >= 3 {
            // Deterministic rejection: park it (never delete) so other sessions can move.
            let path = folder.appendingPathComponent("session.json")
            if let data = try? Data(contentsOf: path),
               var value = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] {
                value["rejected"] = message
                if let contents = try? JSONSerialization.data(withJSONObject: value) { try? contents.write(to: path, options: .atomic) }
            }
            publish(.rejected, "Windows rechazó 3 veces la sesión \(id.prefix(8)): \(message). Conservada en el iPhone.")
        } else {
            publish(.rejected, "Windows rechazó un bloque de \(id.prefix(8)): \(message). Reintentando (\(count)/3).")
        }
    }

    private func receive(_ socket: URLSessionWebSocketTask) async throws -> [String: Any] {
        let seconds = replyTimeout
        let message = try await withThrowingTaskGroup(of: URLSessionWebSocketTask.Message.self) { group -> URLSessionWebSocketTask.Message in
            group.addTask { try await socket.receive() }
            group.addTask {
                try await Task.sleep(nanoseconds: UInt64(seconds * 1_000_000_000))
                socket.cancel(with: .goingAway, reason: nil)
                throw Timeout()
            }
            guard let first = try await group.next() else { throw Timeout() }
            group.cancelAll()
            return first
        }
        let data: Data
        switch message {
        case .string(let text): data = Data(text.utf8)
        case .data(let value): data = value
        @unknown default: throw CocoaError(.fileReadUnknown)
        }
        guard let value = try JSONSerialization.jsonObject(with: data) as? [String: Any] else { throw CocoaError(.fileReadCorruptFile) }
        if let error = value["error"] { throw RemoteError(message: "\(error)") }
        return value
    }

    private static func same(_ a: Any?, _ b: String) -> Bool {
        guard let a = a as? String else { return false }
        return a.lowercased() == b.lowercased()
    }

    private func sendSession(_ folder: URL) async throws {
        let ip = address().trimmingCharacters(in: .whitespacesAndNewlines)
        guard !ip.isEmpty else { throw NoAddress() }
        guard let url = URL(string: "ws://\(ip):\(port)") else { throw NoAddress() }
        let id = folder.lastPathComponent
        publish(.connecting, "Conectando con Windows \(ip):\(port) para la sesión \(id.prefix(8))…")
        let socket = URLSession.shared.webSocketTask(with: url)
        socket.maximumMessageSize = 9 * 1024 * 1024
        socket.resume()
        defer { socket.cancel(with: .normalClosure, reason: nil) }
        let open = try JSONSerialization.data(withJSONObject: ["op": "open", "session": id, "version": 3])
        try await socket.send(.string(String(decoding: open, as: UTF8.self)))
        let hello = try await receive(socket)
        // Windows canonicalizes UUIDs; compare case-insensitively.
        guard hello["op"] as? String == "opened", Self.same(hello["session"], id) else { throw CocoaError(.fileReadCorruptFile) }
        var inflight: [(url: URL, seq: UInt32, hex: String)] = []
        while !Task.isCancelled {
            let chunks = (try FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil))
                .filter { $0.pathExtension == "b3dc" }.sorted { $0.path < $1.path }
            let queued = Set(inflight.map { $0.url.lastPathComponent })
            for chunk in chunks where inflight.count < window && !queued.contains(chunk.lastPathComponent) {
                guard let number = UInt32(chunk.deletingPathExtension().lastPathComponent) else { continue }
                if let last = inflight.last, number <= last.seq { continue }
                let data = try Data(contentsOf: chunk)
                let digest = SHA256.hash(data: data)
                let hex = digest.map { String(format: "%02x", $0) }.joined()
                var seq = number.littleEndian
                var message = withUnsafeBytes(of: &seq) { Data($0) }
                message.append(contentsOf: digest); message.append(data)
                try await socket.send(.data(message))
                inflight.append((url: chunk, seq: number, hex: hex))
            }
            if let head = inflight.first {
                let ack = try await receive(socket)
                guard ack["op"] as? String == "ack", Self.same(ack["session"], id),
                      (ack["seq"] as? NSNumber)?.uint32Value == head.seq, ack["sha256"] as? String == head.hex else {
                    throw CocoaError(.fileReadCorruptFile)
                }
                // Only now is the block durable on Windows.
                try FileManager.default.removeItem(at: head.url)
                inflight.removeFirst()
                lock.lock(); lastAck = Date(); failures.removeValue(forKey: id); lock.unlock()
                publish(.sending, "Bloque \(head.seq + 1) confirmado por Windows. Caché del iPhone: \(CaptureRecorder.pendingBytes() / 1_000_000) MB.")
                continue
            }
            let data = try Data(contentsOf: folder.appendingPathComponent("session.json"))
            guard var manifest = try JSONSerialization.jsonObject(with: data) as? [String: Any] else { throw CocoaError(.fileReadCorruptFile) }
            if manifest["finished"] as? Bool == true {
                manifest["op"] = "finish"
                manifest.removeValue(forKey: "stopReason")
                let message = try JSONSerialization.data(withJSONObject: manifest)
                try await socket.send(.string(String(decoding: message, as: UTF8.self)))
                let receipt = try await receive(socket)
                guard receipt["op"] as? String == "committed", Self.same(receipt["session"], id),
                      (receipt["frames"] as? NSNumber)?.intValue == (manifest["frames"] as? NSNumber)?.intValue else {
                    throw CocoaError(.fileReadCorruptFile)
                }
                // Session directory contains only acknowledged chunks and metadata.
                guard (try FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil))
                    .allSatisfy({ $0.lastPathComponent == "session.json" }) else { throw CocoaError(.fileWriteUnknown) }
                try FileManager.default.removeItem(at: folder)
                let frames = (manifest["frames"] as? NSNumber)?.intValue ?? 0
                publish(.committed, "Sesión \(id.prefix(8)) verificada en Windows (\(frames) fotogramas). Caché local liberada.")
                return
            }
            if manifest["frames"] as? Int == 0 && chunks.isEmpty {
                publish(.sending, "Conectado con Windows; esperando el primer fotograma de la captura.")
            }
            try await Task.sleep(nanoseconds: 200_000_000)
        }
    }
    deinit { task?.cancel() }
}
