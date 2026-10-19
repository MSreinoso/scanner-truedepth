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

func append<T>(_ value: T, to packet: inout Data) { var v = value; withUnsafeBytes(of: &v) { packet.append(contentsOf: $0) } }
func frame(_ id: UInt32) -> Data {
    var packet = Data()
    append(UInt32(0xB1C01702), to: &packet); append(id, to: &packet); append(Double(id) + 1, to: &packet)
    let values: [Float] = [1,0,0,0,1,0,0,0,1,0,0,0,100,100,1,1]
    for value in values { append(value, to: &packet) }
    append(UInt16(2), to: &packet); append(UInt16(2), to: &packet); append(UInt32(16), to: &packet)
    let depthBits: [UInt32] = [0x80000000,0x7fc01234,0x3e99999a,0x00000001]
    for bits in depthBits { append(bits, to: &packet) }
    append(UInt16(2), to: &packet); append(UInt16(2), to: &packet); append(UInt32(16), to: &packet)
    packet.append(contentsOf: [7,22,35,255,98,43,10,255,0,255,66,255,200,30,90,255])
    append(UInt32(2), to: &packet); packet.append(Data("{}".utf8)); return packet
}
let codec = LosslessFrameCodec()
for id in 0..<2 {
    let raw = frame(UInt32(id)); let encoded = try codec.encode(raw)
    try raw.write(to: folder.appendingPathComponent("frame\(id).raw"))
    try encoded.write(to: folder.appendingPathComponent("frame\(id).b3dc"))
}
