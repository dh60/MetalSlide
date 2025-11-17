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
                VStack(alignment: .leading) {
                    Text(renderer.info)
                    Toggle("Scaling", isOn: $renderer.scalingEnabled)
                        .onChange(of: renderer.scalingEnabled) { renderer.view?.needsDisplay = true }
                }
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
        let r = context.coordinator
        NSEvent.addLocalMonitorForEvents(matching: .keyDown) { r.handleKey($0); return nil }
        Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { _ in
            if r.autoadvanceInterval > 0 && Date().timeIntervalSince(r.slideChangedTime) >= Double(r.autoadvanceInterval) {
                r.goToSlide(r.currentIndex + 1)
            }
        }
        return view
    }

    func updateNSView(_ nsView: MTKView, context: Context) {}
}

class Renderer: NSObject, MTKViewDelegate, ObservableObject {
    var imagePaths: [URL] = []
    var currentIndex = 0
    var textureDirty = true
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
    var scaler: (MTL4FXSpatialScaler, MTLTexture)?
    var scalerFence: MTLFence!
    var commandBuffer: MTL4CommandBuffer!
    @Published var info = ""
    @Published var showInfo = false
    @Published var scalingEnabled = true
    var autoadvanceInterval = 0
    var slideChangedTime = Date()

    func initializeMetal(_ view: MTKView) {
        device = view.device
        queue = device.makeMTL4CommandQueue()
        compiler = try! device.makeCompiler(descriptor: MTL4CompilerDescriptor())

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

            float bessel_j1(float x) {
                float ax = abs(x);
                if (ax < 8.0) {
                    float y = x * x;
                    float ans1 = x * (72362614232.0 + y * (-7895059235.0 + y * (242396853.1
                        + y * (-2972611.439 + y * (15704.48260 + y * (-30.16036606))))));
                    float ans2 = 144725228442.0 + y * (2300535178.0 + y * (18583304.74
                        + y * (99447.43394 + y * (376.9991397 + y * 1.0))));
                    return ans1 / ans2;
                } else {
                    float z = 8.0 / ax;
                    float y = z * z;
                    float xx = ax - 2.356194491;
                    float ans1 = 1.0 + y * (0.183105e-2 + y * (-0.3516396496e-4
                        + y * (0.2457520174e-5 + y * (-0.240337019e-6))));
                    float ans2 = 0.04687499995 + y * (-0.2002690873e-3
                        + y * (0.8449199096e-5 + y * (-0.88228987e-6 + y * 0.105787412e-6)));
                    float ans = sqrt(0.636619772 / ax) * (cos(xx) * ans1 - z * sin(xx) * ans2);
                    return x < 0.0 ? -ans : ans;
                }
            }

            float jinc(float x) {
                if (x < 0.0001) return 1.0;
                float pi_x = M_PI_F * x;
                return 2.0 * bessel_j1(pi_x) / pi_x;
            }

            fragment half4 jincFragment(VertexOut in [[stage_in]], texture2d<half> tex [[texture(0)]]) {
                float2 texSize = float2(tex.get_width(), tex.get_height());
                float2 texelPos = in.texCoord * texSize;
                half4 color = half4(0.0);
                float totalWeight = 0.0;
                const int radius = 4;
                const float windowRadius = 4.0;
                for (int y = -radius; y <= radius; y++) {
                    for (int x = -radius; x <= radius; x++) {
                        float2 centerPos = floor(texelPos) + float2(x, y) + 0.5;
                        float2 delta = texelPos - centerPos;
                        float dist = length(delta);
                        if (dist < windowRadius) {
                            float weight = jinc(dist) * jinc(dist / windowRadius);
                            constexpr sampler s(coord::normalized, address::clamp_to_edge, filter::nearest);
                            color += tex.sample(s, centerPos / texSize) * weight;
                            totalWeight += weight;
                        }
                    }
                }
                return color / totalWeight;
            }
            """
        let options = MTLCompileOptions()
        options.mathMode = .fast

        device.makeLibrary(source: shaderSource, options: options) { [weak self] library, _ in
            guard let self = self else { return }

            let vertDesc = MTL4LibraryFunctionDescriptor()
            vertDesc.name = "vertexShader"
            vertDesc.library = library

            let upscaleDesc = MTL4RenderPipelineDescriptor()
            upscaleDesc.vertexFunctionDescriptor = vertDesc
            upscaleDesc.fragmentFunctionDescriptor = { let d = MTL4LibraryFunctionDescriptor(); d.name = "passthroughFragment"; d.library = library; return d }()
            upscaleDesc.colorAttachments[0].pixelFormat = view.colorPixelFormat

            let downscaleDesc = MTL4RenderPipelineDescriptor()
            downscaleDesc.vertexFunctionDescriptor = vertDesc
            downscaleDesc.fragmentFunctionDescriptor = { let d = MTL4LibraryFunctionDescriptor(); d.name = "jincFragment"; d.library = library; return d }()
            downscaleDesc.colorAttachments[0].pixelFormat = view.colorPixelFormat

            Task {
                self.upscalePipeline = try? await self.compiler.makeRenderPipelineState(descriptor: upscaleDesc)
                self.downscalePipeline = try? await self.compiler.makeRenderPipelineState(descriptor: downscaleDesc)
                await MainActor.run { self.view?.needsDisplay = true }
            }
        }
    }

    func draw(in view: MTKView) {
        guard !imagePaths.isEmpty else { return }

        if upscalePipeline == nil {
            initializeMetal(view)
            return
        }

        if textureDirty {
            textureDirty = false
            texture = try? MTKTextureLoader(device: device).newTexture(URL: imagePaths[currentIndex], options: [
                .textureUsage: MTLTextureUsage.shaderRead.rawValue,
                .textureStorageMode: MTLStorageMode.private.rawValue,
                .SRGB: false
            ])
            scaler = nil
        }

        guard let drawable = view.currentDrawable, let inputTexture = texture else { return }

        let viewportSize = CGSize(width: drawable.texture.width, height: drawable.texture.height)
        let fitSize = CGFloat(inputTexture.width) / CGFloat(inputTexture.height) > viewportSize.width / viewportSize.height
            ? CGSize(width: viewportSize.width, height: viewportSize.width / CGFloat(inputTexture.width) * CGFloat(inputTexture.height))
            : CGSize(width: viewportSize.height / CGFloat(inputTexture.height) * CGFloat(inputTexture.width), height: viewportSize.height)

        let displaySize = scalingEnabled ? fitSize : CGSize(width: inputTexture.width, height: inputTexture.height)
        var scalingMode = ""
        scaler = nil
        if scalingEnabled {
            if Int(fitSize.width) > inputTexture.width || Int(fitSize.height) > inputTexture.height {
                if MTLFXSpatialScalerDescriptor.supportsDevice(device) {
                    let desc = MTLFXSpatialScalerDescriptor()
                    desc.inputWidth = inputTexture.width
                    desc.inputHeight = inputTexture.height
                    desc.outputWidth = Int(fitSize.width)
                    desc.outputHeight = Int(fitSize.height)
                    desc.colorTextureFormat = .rgba8Unorm
                    desc.outputTextureFormat = .rgba8Unorm
                    desc.colorProcessingMode = .perceptual

                    if let s = desc.makeSpatialScaler(device: device, compiler: compiler) {
                        s.fence = scalerFence
                        let outDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: Int(fitSize.width), height: Int(fitSize.height), mipmapped: false)
                        outDesc.usage = s.outputTextureUsage
                        scaler = (s, device.makeTexture(descriptor: outDesc)!)
                    }
                }
                scalingMode = "\nUpscaling: MetalFX"
            } else {
                scalingMode = "\nDownscaling: Jinc"
            }
        }

        if let (s, output) = scaler {
            s.colorTexture = inputTexture
            s.outputTexture = output
            s.inputContentWidth = inputTexture.width
            s.inputContentHeight = inputTexture.height
        }

        info = "Slide: \(currentIndex + 1) of \(imagePaths.count)\nFile: \(imagePaths[currentIndex].lastPathComponent)\nInput: \(inputTexture.width)x\(inputTexture.height)\nOutput: \(Int(displaySize.width))x\(Int(displaySize.height))\(scalingMode)"

        scaleBuffer.contents().assumingMemoryBound(to: Float.self).pointee = Float(displaySize.width / viewportSize.width)
        scaleBuffer.contents().assumingMemoryBound(to: Float.self).advanced(by: 1).pointee = Float(displaySize.height / viewportSize.height)

        residencySet.removeAllAllocations()
        residencySet.addAllocation(inputTexture)
        residencySet.addAllocation(scaleBuffer)
        if let (_, output) = scaler { residencySet.addAllocation(output) }
        residencySet.commit()

        argumentTable.setTexture((scaler?.1 ?? inputTexture).gpuResourceID, index: 0)
        argumentTable.setAddress(scaleBuffer.gpuAddress, index: 0)

        commandBuffer.beginCommandBuffer(allocator: allocator)
        commandBuffer.useResidencySet(residencySet)
        scaler?.0.encode(commandBuffer: commandBuffer)

        let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: view.currentMTL4RenderPassDescriptor!, options: MTL4RenderEncoderOptions())!
        encoder.waitForFence(scalerFence, beforeEncoderStages: .fragment)
        encoder.setRenderPipelineState(scalingEnabled && scaler == nil ? downscalePipeline! : upscalePipeline!)
        encoder.setArgumentTable(argumentTable, stages: [.vertex, .fragment])
        encoder.drawPrimitives(primitiveType: .triangle, vertexStart: 0, vertexCount: 6)
        encoder.endEncoding()
        commandBuffer.endCommandBuffer()
        queue.waitForDrawable(drawable)
        queue.commit([commandBuffer])
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
        case 123: goToSlide(currentIndex - 1) // left
        case 124, 49: goToSlide(currentIndex + 1) // right, space
        case 51: // delete
            try? FileManager.default.trashItem(at: imagePaths[currentIndex], resultingItemURL: nil)
            imagePaths.remove(at: currentIndex)
            goToSlide(currentIndex)
        case 18...29: autoadvanceInterval = event.charactersIgnoringModifiers!.first!.wholeNumberValue ?? autoadvanceInterval
        default: break
        }
    }

    func goToSlide(_ index: Int) {
        currentIndex = (index % imagePaths.count + imagePaths.count) % imagePaths.count
        textureDirty = true
        slideChangedTime = Date()
        view?.needsDisplay = true
    }
}
