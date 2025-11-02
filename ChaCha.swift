import SwiftUI
import Metal
import MetalKit

@main
struct ChaChaApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate
    @StateObject private var state = AppState()

    var body: some Scene {
        WindowGroup {
            if state.folderSelected {
                MetalView(state: state).ignoresSafeArea()
            } else {
                Color.black.ignoresSafeArea()
                    .onAppear {
                        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                            state.pickFolder()
                        }
                    }
            }
        }
        .commands { CommandGroup(replacing: .newItem) {} }
    }
}

class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }
}

class AppState: ObservableObject {
    @Published var folderSelected = false
    @Published var currentIndex = 0
    var imagePaths: [URL] = []

    func pickFolder() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false

        if panel.runModal() == .OK, let url = panel.url {
            let fm = FileManager.default
            let types = ["jpg", "jpeg", "png"]
            if let enumerator = fm.enumerator(at: url, includingPropertiesForKeys: nil) {
                for case let fileURL as URL in enumerator {
                    if types.contains(fileURL.pathExtension.lowercased()) {
                        imagePaths.append(fileURL)
                    }
                }
            }
            imagePaths.shuffle()
            folderSelected = true
        } else {
            NSApp.terminate(nil)
        }
    }

    func next() {
        if !imagePaths.isEmpty {
            currentIndex = (currentIndex + 1) % imagePaths.count
        }
    }

    func previous() {
        if !imagePaths.isEmpty {
            currentIndex = (currentIndex - 1 + imagePaths.count) % imagePaths.count
        }
    }

    func deleteCurrent() {
        try? FileManager.default.trashItem(at: imagePaths[currentIndex], resultingItemURL: nil)
        imagePaths.remove(at: currentIndex)
        if imagePaths.isEmpty {
            NSApp.terminate(nil)
        } else if currentIndex >= imagePaths.count {
            currentIndex = 0
        }
    }
}

struct MetalView: NSViewRepresentable {
    @ObservedObject var state: AppState

    func makeNSView(context: Context) -> MTKView {
        let view = MTKView()
        view.device = MTLCreateSystemDefaultDevice()
        view.delegate = context.coordinator
        view.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        view.framebufferOnly = false
        view.isPaused = true
        view.enableSetNeedsDisplay = true
        view.layer?.contentsScale = NSScreen.main?.backingScaleFactor ?? 2.0
        NSEvent.addLocalMonitorForEvents(matching: .keyDown) { event in
            context.coordinator.handleKey(event)
            return nil
        }
        context.coordinator.view = view
        return view
    }

    func updateNSView(_ nsView: MTKView, context: Context) {
        guard context.coordinator.device != nil else { return }
        context.coordinator.updateTexture()
    }

    func makeCoordinator() -> Renderer {
        Renderer(state: state)
    }
}

class Renderer: NSObject, MTKViewDelegate {
    let state: AppState
    var device: MTLDevice!
    var queue: MTL4CommandQueue!
    var pipeline: MTLRenderPipelineState!
    var allocator: MTL4CommandAllocator!
    var commandBuffer: MTL4CommandBuffer!
    var argumentTable: MTL4ArgumentTable!
    var residencySet: MTLResidencySet!
    var texture: MTLTexture?
    var lastIndex = -1
    weak var view: MTKView?
    var scaleBuffer: MTLBuffer!
    var imageSize = CGSize.zero

    init(state: AppState) {
        self.state = state
    }

    func updateTexture() {
        guard state.currentIndex < state.imagePaths.count, lastIndex != state.currentIndex else { return }
        lastIndex = state.currentIndex

        let url = state.imagePaths[state.currentIndex]
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
              let cgImage = CGImageSourceCreateImageAtIndex(src, 0, nil) else { return }

        let width = cgImage.width
        let height = cgImage.height
        imageSize = CGSize(width: width, height: height)

        var data = [UInt8](repeating: 0, count: width * height * 4)
        let bytesPerRow = width * 4
        let ctx = CGContext(
            data: &data,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: cgImage.colorSpace ?? CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        )!
        ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))

        if texture == nil || texture!.width != width || texture!.height != height {
            let textureDesc = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: .rgba8Unorm,
                width: width,
                height: height,
                mipmapped: false
            )
            textureDesc.usage = [.shaderRead]
            texture = device.makeTexture(descriptor: textureDesc)!
        }

        texture!.replace(
            region: MTLRegion(origin: MTLOrigin(x: 0, y: 0, z: 0),
                             size: MTLSize(width: width, height: height, depth: 1)),
            mipmapLevel: 0,
            withBytes: data,
            bytesPerRow: bytesPerRow
        )

        residencySet.removeAllAllocations()
        argumentTable.setTexture(texture!.gpuResourceID, index: 0)
        argumentTable.setAddress(scaleBuffer.gpuAddress, index: 0)
        residencySet.addAllocation(texture!)
        residencySet.addAllocation(scaleBuffer)
        residencySet.commit()
        view?.needsDisplay = true
    }

    func draw(in view: MTKView) {
        if pipeline == nil {
            device = view.device
            queue = device.makeMTL4CommandQueue()
            let compiler = try! device.makeCompiler(descriptor: MTL4CompilerDescriptor())

            let library = try! device.makeLibrary(source: """
                #include <metal_stdlib>
                using namespace metal;
                struct VertexOut {
                    float4 position [[position]];
                    float2 texCoord;
                };
                vertex VertexOut vertexShader(uint vid [[vertex_id]], constant float2 &scale [[buffer(0)]]) {
                    float2 positions[6] = {
                        float2(-1, -1), float2(1, -1), float2(-1, 1),
                        float2(-1, 1), float2(1, -1), float2(1, 1)
                    };
                    float2 texCoords[6] = {
                        float2(0, 1), float2(1, 1), float2(0, 0),
                        float2(0, 0), float2(1, 1), float2(1, 0)
                    };
                    VertexOut out;
                    out.position = float4(positions[vid] * scale, 0, 1);
                    out.texCoord = texCoords[vid];
                    return out;
                }
                fragment half4 fragmentShader(VertexOut in [[stage_in]], texture2d<half> tex [[texture(0)]]) {
                    constexpr sampler s(mag_filter::linear, min_filter::linear);
                    return tex.sample(s, in.texCoord);
                }
                """, options: nil)

            let vertDesc = MTL4LibraryFunctionDescriptor()
            vertDesc.name = "vertexShader"
            vertDesc.library = library

            let fragDesc = MTL4LibraryFunctionDescriptor()
            fragDesc.name = "fragmentShader"
            fragDesc.library = library

            let pipelineDesc = MTL4RenderPipelineDescriptor()
            pipelineDesc.vertexFunctionDescriptor = vertDesc
            pipelineDesc.fragmentFunctionDescriptor = fragDesc
            pipelineDesc.colorAttachments[0].pixelFormat = view.colorPixelFormat
            pipeline = try! compiler.makeRenderPipelineState(descriptor: pipelineDesc, dynamicLinkingDescriptor: nil, compilerTaskOptions: nil)

            let tableDesc = MTL4ArgumentTableDescriptor()
            tableDesc.maxTextureBindCount = 1
            tableDesc.maxBufferBindCount = 1
            argumentTable = try! device.makeArgumentTable(descriptor: tableDesc)

            scaleBuffer = device.makeBuffer(length: 8, options: .storageModeShared)!
            residencySet = try! device.makeResidencySet(descriptor: MTLResidencySetDescriptor())
            allocator = device.makeCommandAllocator()
            commandBuffer = device.makeCommandBuffer()!
            updateTexture()
        }

        let drawable = view.currentDrawable!
        guard texture != nil else { return }

        let viewportWidth = CGFloat(drawable.texture.width)
        let viewportHeight = CGFloat(drawable.texture.height)

        // Render at actual pixel size (no upscaling/downscaling)
        let scaleX = Float(imageSize.width / viewportWidth)
        let scaleY = Float(imageSize.height / viewportHeight)

        let scalePtr = scaleBuffer.contents().assumingMemoryBound(to: Float.self)
        scalePtr[0] = scaleX
        scalePtr[1] = scaleY

        let passDesc = MTL4RenderPassDescriptor()
        passDesc.colorAttachments[0]!.texture = drawable.texture
        passDesc.colorAttachments[0]!.loadAction = .clear
        passDesc.colorAttachments[0]!.storeAction = .store
        passDesc.colorAttachments[0]!.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)

        allocator.reset()
        commandBuffer.beginCommandBuffer(allocator: allocator)
        let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: passDesc, options: MTL4RenderEncoderOptions())!
        encoder.setRenderPipelineState(pipeline)
        commandBuffer.useResidencySet(residencySet)
        encoder.setArgumentTable(argumentTable, stages: [.vertex, .fragment])
        encoder.drawPrimitives(primitiveType: .triangle, vertexStart: 0, vertexCount: 6)
        encoder.endEncoding()
        commandBuffer.endCommandBuffer()
        queue.waitForDrawable(drawable)
        queue.commit([commandBuffer], options: nil)
        queue.signalDrawable(drawable)
        drawable.present()
    }
    
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

    func handleKey(_ event: NSEvent) {
        switch event.keyCode {
        case 123: state.previous()
        case 124, 49: state.next()
        case 51: state.deleteCurrent()
        default: break
        }
    }
}
