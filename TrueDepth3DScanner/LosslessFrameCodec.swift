import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

/// Pixel-exact PNG for opaque sensor BGRA; Zstd preserves all Float32 depth bits.
final class LosslessFrameCodec {
    private var previousPixels: Data?
    func encode(_ packet: Data) throws -> Data {
        func u32(_ offset: Int) -> Int { packet.withUnsafeBytes { Int($0.loadUnaligned(fromByteOffset: offset, as: UInt32.self).littleEndian) } }
        func u16(_ offset: Int) -> Int { packet.withUnsafeBytes { Int($0.loadUnaligned(fromByteOffset: offset, as: UInt16.self).littleEndian) } }
        let depthSize = u32(84), rgbHeader = 88 + depthSize
        let w = u16(rgbHeader), h = u16(rgbHeader + 2), rgbSize = u32(rgbHeader + 4)
        let rgbStart = rgbHeader + 8, metadataStart = rgbStart + rgbSize
        guard depthSize > 0, rgbSize == w * h * 4, metadataStart + 4 <= packet.count else { throw CocoaError(.fileReadCorruptFile) }
        var identity = Data(packet[88..<rgbHeader]); identity.append(packet[rgbStart..<metadataStart])
        let repeated = previousPixels == identity
        var depth = Data(), png = Data()
        var rawBGRA = false
        if !repeated {
            // Byte shuffle is reversible, including NaN payloads and signed zero.
            var shuffled = Data(count: depthSize)
            let values = depthSize / 4
            shuffled.withUnsafeMutableBytes { dst in
                packet.withUnsafeBytes { src in
                    let d = dst.bindMemory(to: UInt8.self), s = src.bindMemory(to: UInt8.self)
                    for lane in 0..<4 { for i in 0..<values { d[lane * values + i] = s[88 + i * 4 + lane] } }
                }
            }
            depth = Data(count: ZSTD_compressBound(depthSize))
            let encoded = depth.withUnsafeMutableBytes { dst in
                shuffled.withUnsafeBytes { src in ZSTD_compress(dst.baseAddress, dst.count, src.baseAddress, src.count, 1) }
            }
            guard ZSTD_isError(encoded) == 0 else { throw CocoaError(.fileWriteUnknown) }
            depth.count = encoded
            let pixels = Data(packet[rgbStart..<metadataStart])
            let opaque = pixels.withUnsafeBytes({ bytes in
                let values = bytes.bindMemory(to: UInt8.self)
                return stride(from: 3, to: values.count, by: 4).allSatisfy { values[$0] == 255 }
            })
            if !opaque {
                // Some camera buffers carry unused/non-opaque alpha bytes. Never
                // reject a valid sensor frame or premultiply its RGB channels.
                rawBGRA = true
                png = Data(count: ZSTD_compressBound(pixels.count))
                let size = png.withUnsafeMutableBytes { dst in
                    pixels.withUnsafeBytes { src in ZSTD_compress(dst.baseAddress, dst.count, src.baseAddress, src.count, 1) }
                }
                guard ZSTD_isError(size) == 0 else { throw CocoaError(.fileWriteUnknown) }
                png.count = size
            } else {
            guard let provider = CGDataProvider(data: pixels as CFData),
                  let image = CGImage(width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: w * 4,
                    space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedFirst.rawValue).union(.byteOrder32Little),
                    provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent) else { throw CocoaError(.fileWriteUnknown) }
            let output = NSMutableData()
            guard let destination = CGImageDestinationCreateWithData(output, UTType.png.identifier as CFString, 1, nil) else { throw CocoaError(.fileWriteUnknown) }
            CGImageDestinationAddImage(destination, image, nil)
            guard CGImageDestinationFinalize(destination) else { throw CocoaError(.fileWriteUnknown) }
            png = output as Data
            }
        }
        previousPixels = identity
        var header = Data(packet.prefix(88))
        func set(_ value: UInt32, at offset: Int) {
            var le = value.littleEndian
            withUnsafeBytes(of: &le) { header.replaceSubrange(offset..<offset+4, with: $0) }
        }
        set(repeated ? 0xB1C01704 : (rawBGRA ? 0xB1C01705 : 0xB1C01703), at: 0)
        set(UInt32(depth.count), at: 84)
        var result = header; result.append(depth)
        result.append(packet[rgbHeader..<rgbHeader+4])
        var length = UInt32(png.count).littleEndian
        withUnsafeBytes(of: &length) { result.append(contentsOf: $0) }
        result.append(png); result.append(packet[metadataStart...])
        return result
    }
}
