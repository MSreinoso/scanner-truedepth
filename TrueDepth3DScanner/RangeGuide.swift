import Foundation

/// Live distance guidance from the TrueDepth map, like a professional scanner's
/// range indicator. It looks only at the central 40 % of the image, where the
/// object being scanned is expected. Pure Foundation so it is unit-tested on CI.
struct RangeGuide: Equatable {
    enum Level: String { case ok, near, far, noDepth }
    let level: Level
    let medianCm: Double      // 0 when there is no depth in the centre
    let validPercent: Double  // centre pixels with a usable depth

    static let nearLimit: Float = 0.18
    static let farLimit: Float = 0.55

    /// `depth` is row-major metres (0, NaN or inf = no measurement).
    static func evaluate(depth: UnsafeBufferPointer<Float>, width: Int, height: Int, rowStride: Int? = nil) -> RangeGuide {
        let stride = rowStride ?? width
        guard width > 0, height > 0 else { return RangeGuide(level: .noDepth, medianCm: 0, validPercent: 0) }
        let x0 = width * 3 / 10, x1 = max(x0 + 1, width * 7 / 10)
        let y0 = height * 3 / 10, y1 = max(y0 + 1, height * 7 / 10)
        let step = max(1, min(width, height) / 60)
        var values: [Float] = []; var total = 0
        var y = y0
        while y < y1 {
            var x = x0
            while x < x1 {
                let v = depth[y * stride + x]
                total += 1
                if v.isFinite && v > 0.05 && v < 3 { values.append(v) }
                x += step
            }
            y += step
        }
        let valid = total > 0 ? Double(values.count) / Double(total) * 100 : 0
        guard values.count >= max(8, total * 3 / 10) else {
            let median = values.isEmpty ? 0 : Double(values.sorted()[values.count / 2]) * 100
            return RangeGuide(level: .noDepth, medianCm: median, validPercent: valid)
        }
        values.sort()
        let median = values[values.count / 2]
        let level: Level = median < nearLimit ? .near : (median > farLimit ? .far : .ok)
        return RangeGuide(level: level, medianCm: Double(median) * 100, validPercent: valid)
    }

    var message: String {
        switch level {
        case .ok: return String(format: "Distancia correcta: %.0f cm (%.0f %% del centro con profundidad)", medianCm, validPercent)
        case .near: return String(format: "Demasiado cerca: %.0f cm. Aleja el iPhone a 20–40 cm del objeto.", medianCm)
        case .far: return String(format: "Demasiado lejos: %.0f cm. Acércate a 20–40 cm del objeto.", medianCm)
        case .noDepth: return String(format: "Sin profundidad en el centro (%.0f %% válido): el objeto está demasiado cerca (<15 cm), es brillante/transparente o está fuera de alcance. Colócalo a 20–40 cm.", validPercent)
        }
    }
}
