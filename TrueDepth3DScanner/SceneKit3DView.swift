import SwiftUI
import SceneKit

struct SceneKit3DView: UIViewRepresentable {
    let mesh: Mesh3D?
    var showWireframe: Bool = false
    
    func makeUIView(context: Context) -> SCNView {
        let scnView = SCNView()
        scnView.backgroundColor = UIColor.black
        scnView.allowsCameraControl = true
        scnView.autoenablesDefaultLighting = true
        
        let scene = SCNScene()
        scnView.scene = scene
        
        let cameraNode = SCNNode()
        cameraNode.camera = SCNCamera()
        cameraNode.camera?.zNear = 0.01
        cameraNode.camera?.zFar = 10.0
        cameraNode.position = SCNVector3(0, 0, 0.35)
        scene.rootNode.addChildNode(cameraNode)
        
        return scnView
    }
    
    func updateUIView(_ scnView: SCNView, context: Context) {
        guard let scene = scnView.scene else { return }
        
        scene.rootNode.childNodes.forEach { node in
            if node.camera == nil {
                node.removeFromParentNode()
            }
        }
        
        guard let mesh = mesh, !mesh.triangles.isEmpty else { return }
        
        var scnVertices: [SCNVector3] = []
        var scnNormals: [SCNVector3] = []
        var indices: [Int32] = []
        
        var currentIndex: Int32 = 0
        for triangle in mesh.triangles {
            scnVertices.append(SCNVector3(triangle.v1.x, triangle.v1.y, triangle.v1.z))
            scnVertices.append(SCNVector3(triangle.v2.x, triangle.v2.y, triangle.v2.z))
            scnVertices.append(SCNVector3(triangle.v3.x, triangle.v3.y, triangle.v3.z))
            
            let n = SCNVector3(triangle.normal.x, triangle.normal.y, triangle.normal.z)
            scnNormals.append(n)
            scnNormals.append(n)
            scnNormals.append(n)
            
            indices.append(currentIndex)
            indices.append(currentIndex + 1)
            indices.append(currentIndex + 2)
            currentIndex += 3
        }
        
        let vertexSource = SCNGeometrySource(vertices: scnVertices)
        let normalSource = SCNGeometrySource(normals: scnNormals)
        let element = SCNGeometryElement(indices: indices, primitiveType: .triangles)
        
        let geometry = SCNGeometry(sources: [vertexSource, normalSource], elements: [element])
        
        let material = SCNMaterial()
        material.diffuse.contents = UIColor.systemCyan
        material.specular.contents = UIColor.white
        material.isDoubleSided = true
        material.lightingModel = .phong
        material.fillMode = showWireframe ? .lines : .fill
        geometry.materials = [material]
        
        let meshNode = SCNNode(geometry: geometry)
        scene.rootNode.addChildNode(meshNode)
    }
}
