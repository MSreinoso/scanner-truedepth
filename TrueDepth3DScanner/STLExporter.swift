import Foundation
import simd

/// Industrial CAD-compliant Binary STL Exporter.
/// Guaranteed compatibility with Autodesk Fusion 360, PrusaSlicer, Ultimaker Cura, and Bambu Studio.
class STLExporter {
    enum STLError: Error, LocalizedError {
        case emptyMesh
        case fileWriteFailed(String)
        case invalidFacetCount
        
        var errorDescription: String? {
            switch self {
            case .emptyMesh:
                return "La malla generada está vacía. Ajusta la distancia al objeto."
            case .fileWriteFailed(let msg):
                return "No se pudo escribir el archivo STL binario: \(msg)"
            case .invalidFacetCount:
                return "Recuento de facetas inválido para el estándar binario CAD."
            }
        }
    }
    
    /// Exports Mesh3D to a standard-compliant Binary STL file.
    /// Scaled strictly to millimeters (1.0 CAD unit = 1.0 mm).
    static func exportToBinarySTL(mesh: Mesh3D, filename: String = "TrueDepth_Scan_CAD.stl") throws -> URL {
        guard !mesh.triangles.isEmpty else {
            throw STLError.emptyMesh
        }
        
        let tempDir = FileManager.default.temporaryDirectory
        let fileURL = tempDir.appendingPathComponent(filename)
        
        var data = Data()
        
        // MARK: - 1. 80-Byte Header (CAD Compliant)
        // CRITICAL: Must NEVER begin with ASCII "solid", which causes slicers to switch to an ASCII parser and crash.
        var header = [UInt8](repeating: 0, count: 80)
        let headerPrefix = "TrueDepth-3D-Scanner [Binary CAD Metric mm] Manifold-v2.0"
        let prefixBytes = Array(headerPrefix.utf8.prefix(80))
        for (i, b) in prefixBytes.enumerated() {
            header[i] = b
        }
        data.append(contentsOf: header)
        
        // MARK: - 2. Facet Count (4-byte unsigned int, Little-Endian)
        let triangleCount = UInt32(mesh.triangles.count)
        var triangleCountLE = triangleCount.littleEndian
        withUnsafeBytes(of: &triangleCountLE) { data.append(contentsOf: $0) }
        
        // MARK: - 3. Facet Records (50 bytes per triangle)
        var attributeByteCount: UInt16 = 0 // 0x0000 attribute byte
        
        for tri in mesh.triangles {
            // Normal Vector (12 bytes: 3x IEEE 754 32-bit float)
            var nx = tri.normal.x
            var ny = tri.normal.y
            var nz = tri.normal.z
            withUnsafeBytes(of: &nx) { data.append(contentsOf: $0) }
            withUnsafeBytes(of: &ny) { data.append(contentsOf: $0) }
            withUnsafeBytes(of: &nz) { data.append(contentsOf: $0) }
            
            // Vertex 1 (12 bytes: scaled to mm)
            var v1x = tri.v1.x * 1000.0
            var v1y = tri.v1.y * 1000.0
            var v1z = tri.v1.z * 1000.0
            withUnsafeBytes(of: &v1x) { data.append(contentsOf: $0) }
            withUnsafeBytes(of: &v1y) { data.append(contentsOf: $0) }
            withUnsafeBytes(of: &v1z) { data.append(contentsOf: $0) }
            
            // Vertex 2 (12 bytes: scaled to mm)
            var v2x = tri.v2.x * 1000.0
            var v2y = tri.v2.y * 1000.0
            var v2z = tri.v2.z * 1000.0
            withUnsafeBytes(of: &v2x) { data.append(contentsOf: $0) }
            withUnsafeBytes(of: &v2y) { data.append(contentsOf: $0) }
            withUnsafeBytes(of: &v2z) { data.append(contentsOf: $0) }
            
            // Vertex 3 (12 bytes: scaled to mm)
            var v3x = tri.v3.x * 1000.0
            var v3y = tri.v3.y * 1000.0
            var v3z = tri.v3.z * 1000.0
            withUnsafeBytes(of: &v3x) { data.append(contentsOf: $0) }
            withUnsafeBytes(of: &v3y) { data.append(contentsOf: $0) }
            withUnsafeBytes(of: &v3z) { data.append(contentsOf: $0) }
            
            // Attribute Byte Count (2 bytes: 0x0000)
            withUnsafeBytes(of: &attributeByteCount) { data.append(contentsOf: $0) }
        }
        
        // Strict Validation: Expected total bytes = 84 + (N * 50)
        let expectedBytes = 84 + Int(triangleCount) * 50
        guard data.count == expectedBytes else {
            throw STLError.invalidFacetCount
        }
        
        do {
            try data.write(to: fileURL, options: .atomic)
            return fileURL
        } catch {
            throw STLError.fileWriteFailed(error.localizedDescription)
        }
    }
}
