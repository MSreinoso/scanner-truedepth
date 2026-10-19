import Foundation

// Exercise the exact iOS streaming compressor on macOS with multiple input and
// output buffers, including an incompressible final partial block.
let folder = URL(fileURLWithPath: CommandLine.arguments[1], isDirectory: true)
try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
var state: UInt64 = 0x123456789ABCDEF
var input = Data(capacity: 2 * 1024 * 1024 + 13)
for _ in 0..<(2 * 1024 * 1024 + 13) {
    state ^= state << 13; state ^= state >> 7; state ^= state << 17
    input.append(UInt8(truncatingIfNeeded: state))
}
let source = folder.appendingPathComponent("compressor.b3draw")
try input.write(to: source)
let (archive, digest) = try CaptureRecorder.compress(source)
try digest.write(to: archive.appendingPathExtension("sha256"), atomically: true, encoding: .utf8)
print("Streaming compression fixture complete: \(archive.lastPathComponent)")
