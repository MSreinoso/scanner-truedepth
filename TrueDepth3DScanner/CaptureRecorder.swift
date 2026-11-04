import Foundation
import Compression
import CryptoKit

/// Append-only native depth/BGRA records. No JPEG, fusion, or compression here.
final class CaptureRecorder {
    static let header = Data("B3DREC02".utf8)
    private(set) var url: URL?
    private(set) var count = 0
    private(set) var bytes: UInt64 = 0

    /// Tests may redirect storage; production always uses Documents/Scans.
    static var overrideDirectory: URL?
    static var directory: URL {
        let folder = overrideDirectory ?? FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Scans", isDirectory: true)
        try? FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        return folder
    }

    private var codec = LosslessFrameCodec()
    /// Phone-side cache for blocks not yet confirmed by Windows. This bounds the
    /// local backlog only; a complete session may be much larger on Windows.
    static var cacheLimit: UInt64 = 180_000_000

    /// Raised instead of writing when unconfirmed blocks would exceed the cache.
    /// Pending blocks are never overwritten or discarded to make room.
    struct CacheFull: LocalizedError {
        let pending: UInt64
        var errorDescription: String? {
            "Caché del iPhone llena: \(pending / 1_000_000) MB esperan confirmación de Windows (límite \(CaptureRecorder.cacheLimit / 1_000_000) MB)."
        }
    }

    struct PendingSummary {
        var bytes: UInt64 = 0
        var sessions = 0
        var rejected = 0
        var legacyBytes: UInt64 = 0
        var legacyFiles = 0
    }

    static func sealInterruptedSessions() {
        let folders = (try? FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)) ?? []
        for folder in folders {
            let path = folder.appendingPathComponent("session.json")
            guard let data = try? Data(contentsOf: path),
                  var value = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
                  value["finished"] as? Bool == false else { continue }
            let files = (try? FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil)) ?? []
            var frames = (value["frames"] as? NSNumber)?.intValue ?? 0
            for file in files where file.lastPathComponent.hasSuffix(".b3dc.pending") {
                let destination = file.deletingPathExtension()
                if !FileManager.default.fileExists(atPath: destination.path) { try? FileManager.default.moveItem(at: file, to: destination) }
            }
            for file in files where file.pathExtension == "b3dc" || file.lastPathComponent.hasSuffix(".b3dc.pending") {
                if let seq = Int(file.lastPathComponent.components(separatedBy: ".")[0]) { frames = max(frames, seq + 1) }
            }
            value["frames"] = frames
            if frames == 0 && files.allSatisfy({ $0.lastPathComponent == "session.json" }) {
                try? FileManager.default.removeItem(at: folder)
                continue
            }
            value["finished"] = true
            value["interrupted"] = true
            if let contents = try? JSONSerialization.data(withJSONObject: value) { try? contents.write(to: path, options: .atomic) }
        }
    }

    static func pendingBytes(excluding excluded: URL? = nil) -> UInt64 {
        let files = FileManager.default.enumerator(at: directory, includingPropertiesForKeys: [.fileSizeKey])
        var total: UInt64 = 0
        let skip = excluded?.standardizedFileURL.path
        while let url = files?.nextObject() as? URL {
            guard url.pathExtension == "b3dc" || url.lastPathComponent.hasSuffix(".b3dc.pending") else { continue }
            if let skip = skip, url.deletingLastPathComponent().standardizedFileURL.path == skip { continue }
            total += UInt64((try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)
        }
        return total
    }

    /// Unconfirmed v3 sessions plus legacy v2 archives that no longer upload.
    static func summary() -> PendingSummary {
        var result = PendingSummary()
        let entries = (try? FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: [.fileSizeKey, .isDirectoryKey])) ?? []
        for entry in entries {
            let isFolder = (try? entry.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) ?? false
            if isFolder {
                let manifest = entry.appendingPathComponent("session.json")
                guard FileManager.default.fileExists(atPath: manifest.path) else { continue }
                result.sessions += 1
                if let data = try? Data(contentsOf: manifest),
                   let value = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any], value["rejected"] != nil {
                    result.rejected += 1
                }
                let files = (try? FileManager.default.contentsOfDirectory(at: entry, includingPropertiesForKeys: [.fileSizeKey])) ?? []
                for file in files where file.pathExtension == "b3dc" || file.lastPathComponent.hasSuffix(".b3dc.pending") {
                    result.bytes += UInt64((try? file.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)
                }
            } else if ["b3draw", "b3dz", "partial"].contains(entry.pathExtension) {
                result.legacyFiles += 1
                result.legacyBytes += UInt64((try? entry.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)
            }
        }
        return result
    }

    /// User-confirmed only: removes sessions Windows never confirmed and v2 archives.
    static func discardPending(except active: URL?) -> UInt64 {
        var freed: UInt64 = 0
        let keep = active?.standardizedFileURL.path
        let entries = (try? FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: [.fileSizeKey])) ?? []
        for entry in entries where entry.standardizedFileURL.path != keep {
            if let walker = FileManager.default.enumerator(at: entry, includingPropertiesForKeys: [.fileSizeKey]) {
                while let file = walker.nextObject() as? URL { freed += UInt64((try? file.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0) }
            }
            freed += UInt64((try? entry.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)
            try? FileManager.default.removeItem(at: entry)
        }
        return freed
    }

    func start() throws {
        let folder = Self.directory.appendingPathComponent(UUID().uuidString.lowercased(), isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        url = folder; count = 0; bytes = 0; codec = LosslessFrameCodec()
        try writeManifest(finished: false, dropped: 0)
    }

    private func writeManifest(finished: Bool, dropped: Int, reason: String? = nil) throws {
        guard let folder = url else { throw CocoaError(.fileWriteUnknown) }
        var value: [String: Any] = ["version": 3, "session": folder.lastPathComponent,
            "frames": count, "dropped": dropped, "finished": finished]
        if let reason = reason, !reason.isEmpty { value["stopReason"] = reason }
        let data = try JSONSerialization.data(withJSONObject: value)
        try data.write(to: folder.appendingPathComponent("session.json"), options: .atomic)
    }

    func append(_ packet: Data) throws {
        guard let folder = url else { throw CocoaError(.fileWriteUnknown) }
        let encoded = try codec.encode(packet)
        let pending = Self.pendingBytes()
        guard pending + UInt64(encoded.count) < Self.cacheLimit else {
            // The skipped frame must not become the reference of a later repeat block.
            codec = LosslessFrameCodec()
            throw CacheFull(pending: pending)
        }
        let path = folder.appendingPathComponent(String(format: "%09d.b3dc", count))
        let partial = path.appendingPathExtension("pending")
        do {
            try encoded.write(to: partial, options: .atomic)
            let handle = try FileHandle(forWritingTo: partial)
            try handle.synchronize(); try handle.close()
            try FileManager.default.moveItem(at: partial, to: path)
        } catch {
            try? FileManager.default.removeItem(at: partial)
            codec = LosslessFrameCodec()
            throw error
        }
        count += 1; bytes += UInt64(encoded.count)
        try writeManifest(finished: false, dropped: 0)
    }

    func finish(dropped: Int, reason: String? = nil) throws -> URL {
        if let folder = url {
            for file in try FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil) where file.lastPathComponent.hasSuffix(".b3dc.pending") {
                let destination = file.deletingPathExtension()
                if !FileManager.default.fileExists(atPath: destination.path) { try FileManager.default.moveItem(at: file, to: destination) }
                if let seq = Int(file.lastPathComponent.components(separatedBy: ".")[0]) { count = max(count, seq + 1) }
            }
        }
        try writeManifest(finished: true, dropped: dropped, reason: reason)
        guard let url = url else { throw CocoaError(.fileWriteUnknown) }
        return url
    }

    /// Streaming raw DEFLATE, bounded to two 256 KiB buffers after capture ends.
    static func compress(_ inputURL: URL) throws -> (URL, String) {
        let outputURL = inputURL.deletingPathExtension().appendingPathExtension("b3dz")
        let partial = outputURL.appendingPathExtension("partial")
        FileManager.default.createFile(atPath: partial.path, contents: nil)
        let input = try FileHandle(forReadingFrom: inputURL)
        let output = try FileHandle(forWritingTo: partial)
        defer { try? input.close(); try? output.close() }
        let capacity = 256 * 1024
        let buffer = UnsafeMutablePointer<UInt8>.allocate(capacity: capacity)
        defer { buffer.deallocate() }
        var stream = compression_stream(dst_ptr: buffer, dst_size: capacity, src_ptr: buffer, src_size: 0, state: nil)
        guard compression_stream_init(&stream, COMPRESSION_STREAM_ENCODE, COMPRESSION_ZLIB) != COMPRESSION_STATUS_ERROR else {
            throw CocoaError(.fileWriteUnknown)
        }
        defer { compression_stream_destroy(&stream) }
        var hash = SHA256()
        var finished = false
        while !finished {
            let data = try input.read(upToCount: capacity) ?? Data()
            try data.withUnsafeBytes { raw in
                stream.src_ptr = raw.bindMemory(to: UInt8.self).baseAddress ?? UnsafePointer(buffer)
                stream.src_size = data.count
                repeat {
                    stream.dst_ptr = buffer
                    stream.dst_size = capacity
                    let flags = data.isEmpty ? Int32(COMPRESSION_STREAM_FINALIZE.rawValue) : 0
                    let state = compression_stream_process(&stream, flags)
                    guard state != COMPRESSION_STATUS_ERROR else { throw CocoaError(.fileWriteUnknown) }
                    let chunk = Data(bytes: buffer, count: capacity - stream.dst_size)
                    try output.write(contentsOf: chunk)
                    hash.update(data: chunk)
                    if state == COMPRESSION_STATUS_END { finished = true }
                } while stream.src_size > 0 || stream.dst_size == 0 || (data.isEmpty && !finished)
            }
        }
        try output.synchronize()
        try output.close()
        if FileManager.default.fileExists(atPath: outputURL.path) { try FileManager.default.removeItem(at: outputURL) }
        try FileManager.default.moveItem(at: partial, to: outputURL)
        return (outputURL, hash.finalize().map { String(format: "%02x", $0) }.joined())
    }

}
