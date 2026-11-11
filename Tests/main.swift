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

// Live distance guidance (centre of the depth map).
func guide(_ fill: (Int, Int) -> Float) -> RangeGuide {
    var map = [Float](repeating: 0, count: 64 * 48)
    for y in 0..<48 { for x in 0..<64 { map[y * 64 + x] = fill(x, y) } }
    return map.withUnsafeBufferPointer { RangeGuide.evaluate(depth: $0, width: 64, height: 48) }
}
precondition(guide { _, _ in 0.30 }.level == .ok)
precondition(guide { _, _ in 0.10 }.level == .near)
precondition(guide { _, _ in 0.80 }.level == .far)
// Object too close: the sensor returns no depth in the centre, only background around it.
precondition(guide { x, y in (x > 15 && x < 50 && y > 10 && y < 38) ? .nan : 0.45 }.level == .noDepth)
print("RangeGuide OK")

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
// Camera BGRA may contain zero/unused alpha. It must remain byte-exact and
// must not cancel capture on its first frame.
var nonOpaque = frame(2)
let rgbStart = 88 + 16 + 8
for index in 0..<4 { nonOpaque[rgbStart + index * 4 + 3] = UInt8(index * 32) }
let encodedNonOpaque = try codec.encode(nonOpaque)
try nonOpaque.write(to: folder.appendingPathComponent("frame2.raw"))
try encodedNonOpaque.write(to: folder.appendingPathComponent("frame2.b3dc"))
let recorder = CaptureRecorder()
try recorder.start()
try recorder.append(nonOpaque)
precondition(recorder.count == 1)
let recorded = try recorder.finish(dropped: 0)
try FileManager.default.removeItem(at: recorded)

// Optional end-to-end run against the Windows block server:
//   compression-test <fixture> <port> <scans-dir> <raw-dir>
if CommandLine.arguments.count > 4, let port = Int(CommandLine.arguments[2]) {
    let scans = URL(fileURLWithPath: CommandLine.arguments[3], isDirectory: true)
    let rawFolder = URL(fileURLWithPath: CommandLine.arguments[4], isDirectory: true)
    try FileManager.default.createDirectory(at: rawFolder, withIntermediateDirectories: true)
    CaptureRecorder.overrideDirectory = scans
    let session = CaptureRecorder()
    try session.start()
    for id in 0..<12 {
        let raw = frame(UInt32(id)) // identical pixels after frame 0 exercise exact-repeat blocks
        try session.append(raw)
        try raw.write(to: rawFolder.appendingPathComponent(String(format: "ws%03d.raw", id)))
    }
    let finished = try session.finish(dropped: 0)
    // Pre-3.3 phones created upper-case session folders; keep that case covered.
    let legacy = finished.deletingLastPathComponent().appendingPathComponent(finished.lastPathComponent.uppercased())
    try FileManager.default.moveItem(at: finished, to: legacy)
    let manifestURL = legacy.appendingPathComponent("session.json")
    var manifest = try JSONSerialization.jsonObject(with: Data(contentsOf: manifestURL)) as! [String: Any]
    manifest["session"] = legacy.lastPathComponent
    try JSONSerialization.data(withJSONObject: manifest).write(to: manifestURL)
    let streamer = ChunkStreamer(address: { "127.0.0.1" })
    streamer.port = port
    streamer.replyTimeout = 5
    streamer.window = 3
    streamer.onStatus = { kind, text in print("[stream] \(kind.rawValue): \(text)") }
    streamer.start()
    let deadline = Date().addingTimeInterval(100)
    while FileManager.default.fileExists(atPath: legacy.path) && Date() < deadline {
        RunLoop.current.run(until: Date().addingTimeInterval(0.2))
    }
    precondition(!FileManager.default.fileExists(atPath: legacy.path), "Session was not committed and released")
    print("IOS_UPLOAD_OK \(legacy.lastPathComponent)")
}
