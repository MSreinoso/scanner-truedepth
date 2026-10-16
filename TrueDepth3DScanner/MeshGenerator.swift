import Foundation
import AVFoundation
import simd

/// Single 3D vertex with geometric coordinates, surface normal, and 24-bit sRGB color.
struct ColoredVertex {
    var position: SIMD3<Float>
    var normal: SIMD3<Float>
    var color: SIMD3<UInt8> // R, G, B
}

/// Triangular facet adhering to CAD right-hand rule winding order (CCW).
struct Triangle3D {
    var v1: SIMD3<Float>
    var v2: SIMD3<Float>
    var v3: SIMD3<Float>
    var normal: SIMD3<Float>
    var c1: SIMD3<UInt8> = SIMD3<UInt8>(200, 200, 200)
    var c2: SIMD3<UInt8> = SIMD3<UInt8>(200, 200, 200)
    var c3: SIMD3<UInt8> = SIMD3<UInt8>(200, 200, 200)
}

/// Solid 3D manifold mesh structure.
struct Mesh3D {
    var vertices: [SIMD3<Float>]
    var vertexColors: [SIMD3<UInt8>]
    var normals: [SIMD3<Float>]
    var triangles: [Triangle3D]
}

/// High-performance geometric reconstruction engine.
/// Integrates pinhole inverse projection, Statistical Outlier Removal (SOR),
/// RGB texture fusion, and base plane clipping.
class MeshGenerator {
    
    // MARK: - Primary Reconstruction Pipeline
    static func generateMesh(
        from depthData: AVDepthData,
        colorPixelBuffer: CVPixelBuffer? = nil,
        minDepth: Float = 0.15,
        maxDepth: Float = 0.70,
        stepSize: Int = 2,
        maxEdgeDiscontinuity: Float = 0.015,
        enableSOR: Bool = true,
        sorNeighborCount: Int = 8,
        sorStdDevMultiplier: Float = 1.5,
        basePlaneCutoffZ: Float? = nil
    ) -> Mesh3D? {
        
        // 1. Convert depth data to 32-bit floating point depth map (meters)
        let depthDataConverted: AVDepthData
        if depthData.depthDataType != kCVPixelFormatType_DepthFloat32 {
            depthDataConverted = depthData.converting(toDepthDataType: kCVPixelFormatType_DepthFloat32)
        } else {
            depthDataConverted = depthData
        }
        
        let depthBuffer = depthDataConverted.depthDataMap
        CVPixelBufferLockBaseAddress(depthBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depthBuffer, .readOnly) }
        
        guard let depthBaseAddress = CVPixelBufferGetBaseAddress(depthBuffer) else { return nil }
        let depthWidth = CVPixelBufferGetWidth(depthBuffer)
        let depthHeight = CVPixelBufferGetHeight(depthBuffer)
        let depthFloatBuffer = depthBaseAddress.assumingMemoryBound(to: Float32.self)
        
        // 2. Lock color buffer if provided for texture fusion
        var colorBaseAddress: UnsafeMutableRawPointer? = nil
        var colorBytesPerRow: Int = 0
        var colorWidth: Int = 0
        var colorHeight: Int = 0
        
        if let colorBuf = colorPixelBuffer {
            CVPixelBufferLockBaseAddress(colorBuf, .readOnly)
            colorBaseAddress = CVPixelBufferGetBaseAddress(colorBuf)
            colorBytesPerRow = CVPixelBufferGetBytesPerRow(colorBuf)
            colorWidth = CVPixelBufferGetWidth(colorBuf)
            colorHeight = CVPixelBufferGetHeight(colorBuf)
        }
        defer {
            if let colorBuf = colorPixelBuffer {
                CVPixelBufferUnlockBaseAddress(colorBuf, .readOnly)
            }
        }
        
        // 3. Extract optical calibration matrix (Pinhole Intrinsic Parameters)
        var fx: Float = Float(depthWidth)
        var fy: Float = Float(depthWidth)
        var cx: Float = Float(depthWidth) / 2.0
        var cy: Float = Float(depthHeight) / 2.0
        
        if let calibration = depthDataConverted.cameraCalibrationData {
            let intrinsicMatrix = calibration.intrinsicMatrix
            let referenceSize = calibration.intrinsicMatrixReferenceDimensions
            let scaleX = Float(depthWidth) / Float(referenceSize.width)
            let scaleY = Float(depthHeight) / Float(referenceSize.height)
            
            fx = intrinsicMatrix.columns.0.x * scaleX
            fy = intrinsicMatrix.columns.1.y * scaleY
            cx = intrinsicMatrix.columns.2.x * scaleX
            cy = intrinsicMatrix.columns.2.y * scaleY
        }
        
        let gridCols = depthWidth / stepSize
        let gridRows = depthHeight / stepSize
        
        var gridPoints = Array(repeating: Array<SIMD3<Float>?>(repeating: nil, count: gridCols), count: gridRows)
        var gridColors = Array(repeating: Array<SIMD3<UInt8>>(repeating: SIMD3<UInt8>(200, 200, 200), count: gridCols), count: gridRows)
        
        // 4. Inverse Pinhole Projection & Direct Color Sampling
        for r in 0..<gridRows {
            let yPixel = r * stepSize
            let rowOffset = yPixel * depthWidth
            
            for c in 0..<gridCols {
                let xPixel = c * stepSize
                let z = depthFloatBuffer[rowOffset + xPixel]
                
                // Depth Range & Base Plane Clipping Filter
                if z.isNaN || z.isInfinite || z < minDepth || z > maxDepth {
                    continue
                }
                
                if let planeLimit = basePlaneCutoffZ, z >= planeLimit {
                    continue
                }
                
                // Pinhole Ray Casting into Metric Space (X, Y, Z in meters)
                let xMetric = (Float(xPixel) - cx) * z / fx
                let yMetric = (Float(yPixel) - cy) * z / fy
                
                // Coordinate standard: X right, Y up, Z towards viewer (+Z facing camera)
                let point3D = SIMD3<Float>(xMetric, -yMetric, -z)
                gridPoints[r][c] = point3D
                
                // Extract RGB Color from synchronized video buffer
                if let base = colorBaseAddress, colorWidth > 0, colorHeight > 0 {
                    let colorX = Int((Float(xPixel) / Float(depthWidth)) * Float(colorWidth))
                    let colorY = Int((Float(yPixel) / Float(depthHeight)) * Float(colorHeight))
                    
                    if colorX >= 0 && colorX < colorWidth && colorY >= 0 && colorY < colorHeight {
                        let pixelOffset = colorY * colorBytesPerRow + colorX * 4
                        let bytePtr = base.advanced(by: pixelOffset).assumingMemoryBound(to: UInt8.self)
                        // kCVPixelFormatType_32BGRA
                        let blue = bytePtr[0]
                        let green = bytePtr[1]
                        let red = bytePtr[2]
                        gridColors[r][c] = SIMD3<UInt8>(red, green, blue)
                    }
                }
            }
        }
        
        // 5. Statistical Outlier Removal (SOR): Fast organized depth-space pass
        if enableSOR {
            applyOrganizedSORFilter(
                grid: &gridPoints,
                rows: gridRows,
                cols: gridCols,
                maxDistanceDiscontinuity: maxEdgeDiscontinuity * 1.5,
                minNeighborCount: 3
            )
        }
        
        // 6. Surface Triangulation & Normal Computation (CAD Manifold Rule)
        var triangles: [Triangle3D] = []
        var allVertices: [SIMD3<Float>] = []
        var allColors: [SIMD3<UInt8>] = []
        var allNormals: [SIMD3<Float>] = []
        
        for r in 0..<(gridRows - 1) {
            for c in 0..<(gridCols - 1) {
                guard let p00 = gridPoints[r][c],
                      let p10 = gridPoints[r + 1][c],
                      let p01 = gridPoints[r][c + 1],
                      let p11 = gridPoints[r + 1][c + 1] else {
                    continue
                }
                
                let c00 = gridColors[r][c]
                let c10 = gridColors[r + 1][c]
                let c01 = gridColors[r][c + 1]
                let c11 = gridColors[r + 1][c + 1]
                
                // Triangle 1: (p00 -> p10 -> p01)
                if isValidEdge(p00, p10, maxEdgeDiscontinuity) &&
                   isValidEdge(p10, p01, maxEdgeDiscontinuity) &&
                   isValidEdge(p01, p00, maxEdgeDiscontinuity) {
                    
                    let crossProd = simd_cross(p10 - p00, p01 - p00)
                    let length = simd_length(crossProd)
                    if length > 1e-6 {
                        let normal = crossProd / length
                        triangles.append(Triangle3D(v1: p00, v2: p10, v3: p01, normal: normal, c1: c00, c2: c10, c3: c01))
                        allVertices.append(contentsOf: [p00, p10, p01])
                        allColors.append(contentsOf: [c00, c10, c01])
                        allNormals.append(contentsOf: [normal, normal, normal])
                    }
                }
                
                // Triangle 2: (p10 -> p11 -> p01)
                if isValidEdge(p10, p11, maxEdgeDiscontinuity) &&
                   isValidEdge(p11, p01, maxEdgeDiscontinuity) &&
                   isValidEdge(p01, p10, maxEdgeDiscontinuity) {
                    
                    let crossProd = simd_cross(p11 - p10, p01 - p10)
                    let length = simd_length(crossProd)
                    if length > 1e-6 {
                        let normal = crossProd / length
                        triangles.append(Triangle3D(v1: p10, v2: p11, v3: p01, normal: normal, c1: c10, c2: c11, c3: c01))
                        allVertices.append(contentsOf: [p10, p11, p01])
                        allColors.append(contentsOf: [c10, c11, c01])
                        allNormals.append(contentsOf: [normal, normal, normal])
                    }
                }
            }
        }
        
        return Mesh3D(vertices: allVertices, vertexColors: allColors, normals: allNormals, triangles: triangles)
    }
    
    // MARK: - Validation & Statistical Filter Helpers
    private static func isValidEdge(_ a: SIMD3<Float>, _ b: SIMD3<Float>, _ maxDist: Float) -> Bool {
        return simd_distance(a, b) <= maxDist
    }
    
    /// High-performance organized point cloud statistical filter.
    /// Exploits 2D spatial coherence to eliminate IR flying pixels at silhouette edges in O(N).
    private static func applyOrganizedSORFilter(
        grid: inout [[SIMD3<Float>?]],
        rows: Int,
        cols: Int,
        maxDistanceDiscontinuity: Float,
        minNeighborCount: Int
    ) {
        var validMask = Array(repeating: Array<Bool>(repeating: true, count: cols), count: rows)
        
        for r in 1..<(rows - 1) {
            for c in 1..<(cols - 1) {
                guard let p = grid[r][c] else {
                    validMask[r][c] = false
                    continue
                }
                
                var neighborCount = 0
                // 3x3 kernel neighbor inspection
                for dr in -1...1 {
                    for dc in -1...1 {
                        if dr == 0 && dc == 0 { continue }
                        if let neighbor = grid[r + dr][c + dc] {
                            if simd_distance(p, neighbor) <= maxDistanceDiscontinuity {
                                neighborCount += 1
                            }
                        }
                    }
                }
                
                // Reject floating noise with insufficient local cohesion
                if neighborCount < minNeighborCount {
                    validMask[r][c] = false
                }
            }
        }
        
        for r in 0..<rows {
            for c in 0..<cols {
                if !validMask[r][c] {
                    grid[r][c] = nil
                }
            }
        }
    }
}
