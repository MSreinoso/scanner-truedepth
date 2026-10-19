import Foundation
import Compression
import CryptoKit

/// Append-only native depth/BGRA records. No JPEG, fusion, or compression here.
final class CaptureRecorder {
    static let header = Data("B3DREC02".utf8)
    private var file: FileHandle?
    private(set) var url: URL?
    private(set) var count = 0
    private(set) var bytes: UInt64 = 0

    static var directory: URL {
        let base = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let folder = base.appendingPathComponent("Scans", isDirectory: true)
        try? FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        return folder
    }

    func start() throws {
        let path = Self.directory.appendingPathComponent(UUID().uuidString + ".b3draw")
        FileManager.default.createFile(atPath: path.path, contents: Self.header)
        file = try FileHandle(forWritingTo: path)
        try file?.seekToEnd()
        url = path
        count = 0
        bytes = 8
    }

    func append(_ packet: Data) throws {
        guard let file = file else { throw CocoaError(.fileWriteUnknown) }
        if count % 15 == 0 {
            let attrs = try FileManager.default.attributesOfFileSystem(forPath: Self.directory.path)
            if let free = attrs[.systemFreeSize] as? NSNumber, free.uint64Value < 512 * 1024 * 1024 {
                throw NSError(domain: "Bichito3D", code: 1, userInfo: [NSLocalizedDescriptionKey: "Espacio bajo: captura detenida y conservada."])
            }
        }
        var size = UInt32(packet.count).littleEndian
        try withUnsafeBytes(of: &size) { try file.write(contentsOf: Data($0)) }
        try file.write(contentsOf: packet)
        count += 1
        bytes += UInt64(packet.count + 4)
    }

    func finish(dropped: Int) throws -> URL {
        guard let file = file, let url = url else { throw CocoaError(.fileWriteUnknown) }
        var zero: UInt32 = 0
        try withUnsafeBytes(of: &zero) { try file.write(contentsOf: Data($0)) }
        let metadata = try JSONSerialization.data(withJSONObject: ["frames": count, "dropped": dropped, "version": 2])
        var size = UInt32(metadata.count).littleEndian
        try withUnsafeBytes(of: &size) { try file.write(contentsOf: Data($0)) }
        try file.write(contentsOf: metadata)
        try file.synchronize()
        try file.close()
        self.file = nil
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

    deinit { try? file?.close() }
}
