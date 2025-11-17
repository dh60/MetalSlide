import SwiftUI
import Metal
import MetalKit
import MetalFX

@main
struct MetalSlide: App {
    var body: some Scene {
        Window("MetalSlide", id: "main") {
            MetalView()
        }
    }
}

struct MetalView: View {
    @StateObject private var renderer = Renderer()

    var body: some View {
        ZStack(alignment: .topLeading) {
            MetalViewRepresentable(renderer: renderer)
            if renderer.showInfo {
                Text(renderer.info)
                    .padding()
                    .glassEffect(in: .rect(cornerRadius: 30))
                    .padding()
            }
        }
    }
}

struct MetalViewRepresentable: NSViewRepresentable {
    let renderer: Renderer

    func makeCoordinator() -> Renderer { renderer }

    func makeNSView(context: Context) -> MTKView {
        let view = MTKView()
        view.device = MTLCreateSystemDefaultDevice()
        view.delegate = context.coordinator
        view.isPaused = true
        view.enableSetNeedsDisplay = true
        context.coordinator.view = view
        DispatchQueue.main.async {
            let panel = NSOpenPanel()
            panel.canChooseDirectories = true
            panel.canChooseFiles = false
            guard panel.runModal() == .OK, let url = panel.url else {
                NSApp.terminate(nil)
                return
            }
            context.coordinator.imagePaths = FileManager.default.enumerator(at: url, includingPropertiesForKeys: nil)!.allObjects
                .compactMap { $0 as? URL }
                .filter { ["jpg", "jpeg", "png"].contains($0.pathExtension.lowercased()) }
                .shuffled()
            view.needsDisplay = true
        }
        NSEvent.addLocalMonitorForEvents(matching: .keyDown) {
            context.coordinator.handleKey($0)
            return nil
        }
        Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { _ in
            if context.coordinator.autoadvanceInterval > 0 &&
               Date().timeIntervalSince(context.coordinator.slideChangedTime) >= Double(context.coordinator.autoadvanceInterval) {
                context.coordinator.currentIndex = (context.coordinator.currentIndex + 1) % context.coordinator.imagePaths.count
                context.coordinator.slideChangedTime = Date()
                context.coordinator.view?.needsDisplay = true
            }
        }
        return view
    }

    func updateNSView(_ nsView: MTKView, context: Context) {}
}

class Renderer: NSObject, MTKViewDelegate, ObservableObject {
    var imagePaths: [URL] = []
    var currentIndex = 0
    var lastIndex = -1
    var device: MTLDevice!
    var queue: MTL4CommandQueue!
    var upscalePipeline: MTLRenderPipelineState?
    var downscalePipeline: MTLRenderPipelineState?
    var allocator: MTL4CommandAllocator!
    var argumentTable: MTL4ArgumentTable!
    var residencySet: MTLResidencySet!
    var texture: MTLTexture?
    weak var view: MTKView?
    var scaleBuffer: MTLBuffer!
    var compiler: MTL4Compiler!
    var spatialScaler: MTL4FXSpatialScaler?
    var scalerOutput: MTLTexture?
    var scalerFence: MTLFence!
    var commandBuffer: MTL4CommandBuffer!
    var pipelineReady = false
    var textureLoader: MTKTextureLoader!
    @Published var info = ""
    @Published var showInfo = false
    var autoadvanceInterval = 0
    var slideChangedTime = Date()

    func initializeMetal(_ view: MTKView) {
        device = view.device
        queue = device.makeMTL4CommandQueue()
        compiler = try! device.makeCompiler(descriptor: MTL4CompilerDescriptor())
        textureLoader = MTKTextureLoader(device: device)

        let tableDesc = MTL4ArgumentTableDescriptor()
        tableDesc.maxTextureBindCount = 1
        tableDesc.maxBufferBindCount = 1
        argumentTable = try! device.makeArgumentTable(descriptor: tableDesc)

        scaleBuffer = device.makeBuffer(length: 8, options: .storageModeShared)
        residencySet = try! device.makeResidencySet(descriptor: MTLResidencySetDescriptor())
        allocator = device.makeCommandAllocator()
        scalerFence = device.makeFence()
        commandBuffer = device.makeCommandBuffer()

        let shaderSource = """
            #include <metal_stdlib>
            using namespace metal;
            struct VertexOut { float4 position [[position]]; float2 texCoord; };
            vertex VertexOut vertexShader(uint vid [[vertex_id]], constant float2 &scale [[buffer(0)]]) {
                float2 positions[6] = { float2(-1,-1), float2(1,-1), float2(-1,1), float2(-1,1), float2(1,-1), float2(1,1) };
                float2 texCoords[6] = { float2(0,1), float2(1,1), float2(0,0), float2(0,0), float2(1,1), float2(1,0) };
                return { float4(positions[vid] * scale, 0, 1), texCoords[vid] };
            }

            fragment half4 passthroughFragment(VertexOut in [[stage_in]], texture2d<half> tex [[texture(0)]]) {
                constexpr sampler s(coord::normalized, address::clamp_to_edge, filter::linear);
                return tex.sample(s, in.texCoord);
            }

            float lanczos(float x, float a) {
                if (x == 0.0) return 1.0;
                if (abs(x) >= a) return 0.0;
                float pi_x = M_PI_F * x;
                return (a * sin(pi_x) * sin(pi_x / a)) / (pi_x * pi_x);
            }

            fragment half4 lanczosFragment(VertexOut in [[stage_in]], texture2d<half> tex [[texture(0)]]) {
                float2 texSize = float2(tex.get_width(), tex.get_height());
                float2 texelPos = in.texCoord * texSize;
                half4 color = half4(0.0);
                float totalWeight = 0.0;
                for (int y = -3; y <= 3; y++) {
                    for (int x = -3; x <= 3; x++) {
                        float2 centerPos = floor(texelPos) + float2(x, y) + 0.5;
                        float2 delta = texelPos - centerPos;
                        float weight = lanczos(delta.x, 3.0) * lanczos(delta.y, 3.0);
                        constexpr sampler s(coord::normalized, address::clamp_to_edge, filter::nearest);
                        color += tex.sample(s, centerPos / texSize) * weight;
                        totalWeight += weight;
                    }
                }
                return color / totalWeight;
            }
            """
        let options = MTLCompileOptions()
        options.mathMode = .fast

        device.makeLibrary(source: shaderSource, options: options) { [weak self] library, _ in
            guard let self = self, let library = library else { return }

            let vertDesc = MTL4LibraryFunctionDescriptor()
            vertDesc.name = "vertexShader"
            vertDesc.library = library

            let upscaleDesc = MTL4RenderPipelineDescriptor()
            upscaleDesc.vertexFunctionDescriptor = vertDesc
            upscaleDesc.fragmentFunctionDescriptor = { let d = MTL4LibraryFunctionDescriptor(); d.name = "passthroughFragment"; d.library = library; return d }()
            upscaleDesc.colorAttachments[0].pixelFormat = view.colorPixelFormat

            let downscaleDesc = MTL4RenderPipelineDescriptor()
            downscaleDesc.vertexFunctionDescriptor = vertDesc
            downscaleDesc.fragmentFunctionDescriptor = { let d = MTL4LibraryFunctionDescriptor(); d.name = "lanczosFragment"; d.library = library; return d }()
            downscaleDesc.colorAttachments[0].pixelFormat = view.colorPixelFormat

            Task {
                self.upscalePipeline = try? await self.compiler.makeRenderPipelineState(descriptor: upscaleDesc)
                self.downscalePipeline = try? await self.compiler.makeRenderPipelineState(descriptor: downscaleDesc)
                self.pipelineReady = true
                await MainActor.run { self.view?.needsDisplay = true }
            }
        }
    }

    func draw(in view: MTKView) {
        guard !imagePaths.isEmpty else { return }

        if !pipelineReady {
            if upscalePipeline == nil { initializeMetal(view) }
            return
        }

        if lastIndex != currentIndex {
            lastIndex = currentIndex
            texture = try? textureLoader.newTexture(URL: imagePaths[currentIndex], options: [
                .textureUsage: MTLTextureUsage.shaderRead.rawValue,
                .textureStorageMode: MTLStorageMode.private.rawValue,
                .SRGB: false
            ])
            spatialScaler = nil
        }

        guard let drawable = view.currentDrawable, let inputTexture = texture else { return }

        let viewportSize = CGSize(width: drawable.texture.width, height: drawable.texture.height)
        let imageAspect = CGFloat(inputTexture.width) / CGFloat(inputTexture.height)
        let viewportAspect = viewportSize.width / viewportSize.height
        let fitSize = imageAspect > viewportAspect
            ? CGSize(width: viewportSize.width, height: viewportSize.width / imageAspect)
            : CGSize(width: viewportSize.height * imageAspect, height: viewportSize.height)
        let (outputWidth, outputHeight) = (Int(fitSize.width), Int(fitSize.height))

        if outputWidth > inputTexture.width || outputHeight > inputTexture.height {
            if spatialScaler == nil && MTLFXSpatialScalerDescriptor.supportsDevice(device) {
                let desc = MTLFXSpatialScalerDescriptor()
                desc.inputWidth = inputTexture.width
                desc.inputHeight = inputTexture.height
                desc.outputWidth = outputWidth
                desc.outputHeight = outputHeight
                desc.colorTextureFormat = .rgba8Unorm
                desc.outputTextureFormat = .rgba8Unorm
                desc.colorProcessingMode = .perceptual

                if let scaler = desc.makeSpatialScaler(device: device, compiler: compiler) {
                    scaler.fence = scalerFence
                    spatialScaler = scaler
                    let outDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: outputWidth, height: outputHeight, mipmapped: false)
                    outDesc.usage = scaler.outputTextureUsage
                    scalerOutput = device.makeTexture(descriptor: outDesc)
                }
            }
        } else {
            spatialScaler = nil
        }

        if let scaler = spatialScaler, let output = scalerOutput {
            scaler.colorTexture = inputTexture
            scaler.outputTexture = output
            scaler.inputContentWidth = inputTexture.width
            scaler.inputContentHeight = inputTexture.height
        }

        let renderTexture = spatialScaler != nil ? scalerOutput! : inputTexture
        let pipeline = spatialScaler != nil ? upscalePipeline! : downscalePipeline!

        info = "Slide: \(currentIndex + 1) of \(imagePaths.count)\nFile: \(imagePaths[currentIndex].lastPathComponent)\nInput: \(inputTexture.width)x\(inputTexture.height)\nOutput: \(outputWidth)x\(outputHeight)\n\(spatialScaler != nil ? "Upscale: MetalFX" : "Downscale: Lanczos")"

        let scale = scaleBuffer.contents().assumingMemoryBound(to: Float.self)
        (scale[0], scale[1]) = (Float(fitSize.width / viewportSize.width), Float(fitSize.height / viewportSize.height))

        residencySet.removeAllAllocations()
        residencySet.addAllocation(inputTexture)
        residencySet.addAllocation(scaleBuffer)
        if renderTexture !== inputTexture { residencySet.addAllocation(renderTexture) }
        residencySet.commit()

        argumentTable.setTexture(renderTexture.gpuResourceID, index: 0)
        argumentTable.setAddress(scaleBuffer.gpuAddress, index: 0)

        commandBuffer.beginCommandBuffer(allocator: allocator)
        commandBuffer.useResidencySet(residencySet)
        spatialScaler?.encode(commandBuffer: commandBuffer)

        let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: view.currentMTL4RenderPassDescriptor!, options: MTL4RenderEncoderOptions())!
        if spatialScaler != nil { encoder.waitForFence(scalerFence, beforeEncoderStages: .fragment) }
        encoder.setRenderPipelineState(pipeline)
        encoder.setArgumentTable(argumentTable, stages: [.vertex, .fragment])
        encoder.drawPrimitives(primitiveType: .triangle, vertexStart: 0, vertexCount: 6)
        encoder.endEncoding()
        commandBuffer.endCommandBuffer()
        queue.waitForDrawable(drawable)
        queue.commit([commandBuffer], options: nil)
        queue.signalDrawable(drawable)
        drawable.present()
        allocator.reset()
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        view.needsDisplay = true
    }

    func handleKey(_ event: NSEvent) {
        switch event.keyCode {
        case 53: NSApp.terminate(nil)
        case 34: showInfo.toggle()
        case 123, 124, 49: // left, right, space
            currentIndex = (currentIndex + (event.keyCode == 123 ? -1 + imagePaths.count : 1)) % imagePaths.count
            slideChangedTime = Date()
            view?.needsDisplay = true
        case 51: // delete
            try? FileManager.default.trashItem(at: imagePaths[currentIndex], resultingItemURL: nil)
            imagePaths.remove(at: currentIndex)
            if currentIndex >= imagePaths.count { currentIndex = 0 }
            lastIndex = -1
            slideChangedTime = Date()
            view?.needsDisplay = true
        case 29, 18, 19, 20, 21, 23, 22, 26, 28, 25: // 0-9
            autoadvanceInterval = [29: 0, 18: 1, 19: 2, 20: 3, 21: 4, 23: 5, 22: 6, 26: 7, 28: 8, 25: 9][event.keyCode]!
        default: break
        }
    }
}
