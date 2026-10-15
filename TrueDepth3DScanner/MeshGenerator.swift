import Foundation
import AVFoundation
import simd

struct Triangle3D {
    var v1: SIMD3<Float>
    var v2: SIMD3<Float>
    var v3: SIMD3<Float>
    var normal: SIMD3<Float>
}

struct Mesh3D {
    var vertices: [SIMD3<Float>]
    var triangles: [Triangle3D]
}

class MeshGenerator {
    static func generateMesh(from depthData: AVDepthData,
                             minDepth: Float = 0.15,
                             maxDepth: Float = 0.70,
                             stepSize: Int = 2,
                             maxEdgeDiscontinuity: Float = 0.02) -> Mesh3D? {
        
        let depthDataConverted: AVDepthData
        if depthData.depthDataType != kCVPixelFormatType_DepthFloat32 {
            depthDataConverted = depthData.converting(toDepthDataType: kCVPixelFormatType_DepthFloat32)
        } else {
            depthDataConverted = depthData
        }
        
        let pixelBuffer = depthDataConverted.depthDataMap
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else { return nil }
        
        let floatBuffer = baseAddress.assumingMemoryBound(to: Float32.self)
        
        var fx: Float = Float(width)
        var fy: Float = Float(width)
        var cx: Float = Float(width) / 2.0
        var cy: Float = Float(height) / 2.0
        
        if let calibration = depthDataConverted.cameraCalibrationData {
            let intrinsicMatrix = calibration.intrinsicMatrix
            let referenceSize = calibration.intrinsicMatrixReferenceDimensions
            let scaleX = Float(width) / Float(referenceSize.width)
            let scaleY = Float(height) / Float(referenceSize.height)
            
            fx = intrinsicMatrix.columns.0.x * scaleX
            fy = intrinsicMatrix.columns.1.y * scaleY
            cx = intrinsicMatrix.columns.2.x * scaleX
            cy = intrinsicMatrix.columns.2.y * scaleY
        }
        
        let gridCols = width / stepSize
        let gridRows = height / stepSize
        var grid = Array(repeating: Array<SIMD3<Float>?>(repeating: nil, count: gridCols), count: gridRows)
        var allVertices: [SIMD3<Float>] = []
        
        for r in 0..<gridRows {
            let yPixel = r * stepSize
            let rowOffset = yPixel * width
            
            for c in 0..<gridCols {
                let xPixel = c * stepSize
                let z = floatBuffer[rowOffset + xPixel]
                
                if !z.isNaN && !z.isInfinite && z >= minDepth && z <= maxDepth {
                    let x = (Float(xPixel) - cx) * z / fx
                    let y = (Float(yPixel) - cy) * z / fy
                    let point = SIMD3<Float>(x, -y, -z)
                    grid[r][c] = point
                    allVertices.append(point)
                }
            }
        }
        
        var triangles: [Triangle3D] = []
        
        for r in 0..<(gridRows - 1) {
            for c in 0..<(gridCols - 1) {
                guard let p00 = grid[r][c],
                      let p10 = grid[r + 1][c],
                      let p01 = grid[r][c + 1],
                      let p11 = grid[r + 1][c + 1] else {
                    continue
                }
                
                func isValidTriangle(_ a: SIMD3<Float>, _ b: SIMD3<Float>, _ c: SIMD3<Float>) -> Bool {
                    let d1 = simd_distance(a, b)
                    let d2 = simd_distance(b, c)
                    let d3 = simd_distance(c, a)
                    return d1 <= maxEdgeDiscontinuity && d2 <= maxEdgeDiscontinuity && d3 <= maxEdgeDiscontinuity
                }
                
                if isValidTriangle(p00, p10, p01) {
                    let normal = simd_normalize(simd_cross(p10 - p00, p01 - p00))
                    triangles.append(Triangle3D(v1: p00, v2: p10, v3: p01, normal: normal))
                }
                
                if isValidTriangle(p10, p11, p01) {
                    let normal = simd_normalize(simd_cross(p11 - p10, p01 - p10))
                    triangles.append(Triangle3D(v1: p10, v2: p11, v3: p01, normal: normal))
                }
            }
        }
        
        return Mesh3D(vertices: allVertices, triangles: triangles)
    }
}
