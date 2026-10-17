//
//  FrameSerializer.swift
//  TrueDepth3DScanner
//
//  Binary frame serializer for TCP streaming.
//  Packs synchronized TrueDepth depth data, RGB video frames, and CoreMotion
//  IMU data into a compact binary packet following a custom wire protocol.
//
//  Wire Protocol Layout (little-endian):
//  ┌────────────────────────┬──────────┬───────┐
//  │ Field                  │ Type     │ Bytes │
//  ├────────────────────────┼──────────┼───────┤
//  │ Magic                  │ uint32   │ 4     │  0xB1C01700
//  │ Frame ID               │ uint32   │ 4     │
//  │ Timestamp              │ float64  │ 8     │  PTS seconds
//  │ IMU Rotation           │ f32[9]   │ 36    │  3×3 row-major
//  │ IMU UserAccel          │ f32[3]   │ 12    │  gravity-free m/s²
//  │ Intrinsics             │ f32[4]   │ 16    │  fx, fy, cx, cy
//  │ Depth Width            │ uint16   │ 2     │
//  │ Depth Height           │ uint16   │ 2     │
//  │ Depth Payload Size     │ uint32   │ 4     │
//  │ Depth Payload          │ bytes    │ var   │  zlib(Float16 map)
//  │ RGB Width              │ uint16   │ 2     │
//  │ RGB Height             │ uint16   │ 2     │
//  │ RGB Payload Size       │ uint32   │ 4     │
//  │ RGB Payload            │ bytes    │ var   │  JPEG @ 0.80
//  └────────────────────────┴──────────┴───────┘
//

import Foundation
import AVFoundation
import CoreMotion
import simd
import Compression
import CoreImage
import UIKit

/// Serializes synchronized depth, color, and IMU frames into a compact binary
/// packet suitable for streaming over TCP to a PC-side reconstruction engine.
final class FrameSerializer {

    // MARK: - Constants

    /// Magic number identifying the start of a valid frame packet.
    private static let magic: UInt32 = 0xB1C0_1700

    /// GPU-accelerated CIContext for pixel buffer → UIImage conversion.
    private static let ciContext = CIContext(options: [.useSoftwareRenderer: false])

    // MARK: - Public API

    /// Serializes a single captured frame into the binary wire format.
    ///
    /// - Parameters:
    ///   - frameID:          Monotonically increasing frame counter.
    ///   - timestamp:        Presentation timestamp of the CMSampleBuffer (seconds).
    ///   - depthData:        The `AVDepthData` from the TrueDepth camera.
    ///   - colorPixelBuffer: The RGB pixel buffer from the video output, or `nil`
    ///                       if color is unavailable for this frame.
    ///   - deviceMotion:     The most recent `CMDeviceMotion` snapshot, or `nil`
    ///                       if motion data is unavailable.
    ///   - jpegQuality:      JPEG compression quality for the RGB image (0.0–1.0).
    /// - Returns: A `Data` blob containing the packed binary frame, or `nil` on failure.
    static func serialize(
        frameID: UInt32,
        timestamp: Double,
        depthData: AVDepthData,
        colorPixelBuffer: CVPixelBuffer?,
        deviceMotion: CMDeviceMotion?,
        jpegQuality: CGFloat = 0.80
    ) -> Data? {

        // ──────────────────────────────────────────────
        // 1. Depth buffer — ensure Float32 pixel format
        // ──────────────────────────────────────────────
        let convertedDepth: AVDepthData
        if depthData.depthDataType != kCVPixelFormatType_DepthFloat32 {
            convertedDepth = depthData.converting(toDepthDataType: kCVPixelFormatType_DepthFloat32)
        } else {
            convertedDepth = depthData
        }

        let depthBuffer = convertedDepth.depthDataMap
        CVPixelBufferLockBaseAddress(depthBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depthBuffer, .readOnly) }

        let depthWidth  = CVPixelBufferGetWidth(depthBuffer)
        let depthHeight = CVPixelBufferGetHeight(depthBuffer)
        let depthBytesPerRow = CVPixelBufferGetBytesPerRow(depthBuffer)

        guard let depthBaseAddress = CVPixelBufferGetBaseAddress(depthBuffer) else {
            return nil
        }

        // ──────────────────────────────────────────────
        // 2. Camera intrinsics scaled to depth resolution
        // ──────────────────────────────────────────────
        var fx: Float = 0, fy: Float = 0, cx: Float = 0, cy: Float = 0

        if let calibration = convertedDepth.cameraCalibrationData {
            // intrinsicMatrix is a 3×3 matrix referenced to the calibration's
            // intrinsicMatrixReferenceDimensions. Scale to actual depth size.
            let refDims = calibration.intrinsicMatrixReferenceDimensions
            let scaleX  = Float(depthWidth)  / Float(refDims.width)
            let scaleY  = Float(depthHeight) / Float(refDims.height)

            let K = calibration.intrinsicMatrix // matrix_float3x3, column-major
            fx = K[0][0] * scaleX   // column 0, row 0
            fy = K[1][1] * scaleY   // column 1, row 1
            cx = K[2][0] * scaleX   // column 2, row 0
            cy = K[2][1] * scaleY   // column 2, row 1
        }

        // ──────────────────────────────────────────────
        // 3. Convert depth Float32 → Float16
        // ──────────────────────────────────────────────
        let pixelCount  = depthWidth * depthHeight
        let halfSize    = pixelCount * MemoryLayout<Float16>.size

        var halfBuffer  = Data(count: halfSize)
        halfBuffer.withUnsafeMutableBytes { halfPtr in
            let dst = halfPtr.bindMemory(to: Float16.self)
            let src = depthBaseAddress.assumingMemoryBound(to: Float32.self)
            let srcRowFloats = depthBytesPerRow / MemoryLayout<Float32>.size

            for row in 0..<depthHeight {
                let rowOffset = row * srcRowFloats
                let dstOffset = row * depthWidth
                for col in 0..<depthWidth {
                    dst[dstOffset + col] = Float16(src[rowOffset + col])
                }
            }
        }

        // ──────────────────────────────────────────────
        // 4. Zlib-compress the Float16 depth payload
        // ──────────────────────────────────────────────
        let compressedBound = halfSize + 512  // zlib can expand slightly on tiny data
        var compressedDepth = Data(count: compressedBound)

        let compressedSize: Int = halfBuffer.withUnsafeBytes { srcRaw in
            let srcPtr = srcRaw.bindMemory(to: UInt8.self).baseAddress!
            return compressedDepth.withUnsafeMutableBytes { dstRaw in
                let dstPtr = dstRaw.bindMemory(to: UInt8.self).baseAddress!
                return compression_encode_buffer(
                    dstPtr, compressedBound,
                    srcPtr, halfSize,
                    nil,
                    COMPRESSION_ZLIB
                )
            }
        }

        guard compressedSize > 0 else { return nil }
        compressedDepth.count = compressedSize

        // ──────────────────────────────────────────────
        // 5. RGB → JPEG
        // ──────────────────────────────────────────────
        var jpegData = Data()
        var rgbWidth:  UInt16 = 0
        var rgbHeight: UInt16 = 0

        if let colorPB = colorPixelBuffer {
            rgbWidth  = UInt16(CVPixelBufferGetWidth(colorPB))
            rgbHeight = UInt16(CVPixelBufferGetHeight(colorPB))

            let ciImage = CIImage(cvPixelBuffer: colorPB)
            if let cgImage = ciContext.createCGImage(ciImage, from: ciImage.extent) {
                let uiImage = UIImage(cgImage: cgImage)
                if let encoded = uiImage.jpegData(compressionQuality: jpegQuality) {
                    jpegData = encoded
                }
            }
        }

        // ──────────────────────────────────────────────
        // 6. IMU — rotation matrix & user acceleration
        // ──────────────────────────────────────────────
        var rotation: [Float32] = Array(repeating: 0, count: 9)
        var userAccel: [Float32] = [0, 0, 0]

        if let motion = deviceMotion {
            // CMRotationMatrix is row-major (m11…m33). Pack row-major.
            let r = motion.attitude.rotationMatrix
            rotation = [
                Float32(r.m11), Float32(r.m12), Float32(r.m13),
                Float32(r.m21), Float32(r.m22), Float32(r.m23),
                Float32(r.m31), Float32(r.m32), Float32(r.m33)
            ]
            let a = motion.userAcceleration
            userAccel = [Float32(a.x), Float32(a.y), Float32(a.z)]
        } else {
            // Identity rotation, zero acceleration
            rotation = [
                1, 0, 0,
                0, 1, 0,
                0, 0, 1
            ]
        }

        // ──────────────────────────────────────────────
        // 7. Pack everything into the binary payload
        // ──────────────────────────────────────────────
        // Pre-calculate total size for a single allocation:
        //   header:  4 + 4 + 8 + 36 + 12 + 16 = 80
        //   depth:   2 + 2 + 4 + compressedSize
        //   rgb:     2 + 2 + 4 + jpegData.count
        let totalSize = 80
            + 2 + 2 + 4 + compressedSize
            + 2 + 2 + 4 + jpegData.count

        var packet = Data(capacity: totalSize)

        // Magic
        appendValue(magic, to: &packet)
        // Frame ID
        appendValue(frameID, to: &packet)
        // Timestamp (float64)
        appendValue(timestamp, to: &packet)

        // IMU rotation (9 × float32)
        for r in rotation {
            appendValue(r, to: &packet)
        }
        // IMU user acceleration (3 × float32)
        for a in userAccel {
            appendValue(a, to: &packet)
        }

        // Camera intrinsics (4 × float32)
        appendValue(fx, to: &packet)
        appendValue(fy, to: &packet)
        appendValue(cx, to: &packet)
        appendValue(cy, to: &packet)

        // Depth dimensions
        appendValue(UInt16(depthWidth), to: &packet)
        appendValue(UInt16(depthHeight), to: &packet)

        // Depth payload
        appendValue(UInt32(compressedSize), to: &packet)
        packet.append(compressedDepth)

        // RGB dimensions
        appendValue(rgbWidth, to: &packet)
        appendValue(rgbHeight, to: &packet)

        // RGB payload
        appendValue(UInt32(jpegData.count), to: &packet)
        packet.append(jpegData)

        return packet
    }

    // MARK: - Helpers

    /// Appends the raw bytes of a value to the given `Data` buffer (little-endian
    /// on all Apple platforms since they use ARM / little-endian x86).
    private static func appendValue<T>(_ value: T, to data: inout Data) {
        var v = value
        withUnsafeBytes(of: &v) { data.append(contentsOf: $0) }
    }
}
