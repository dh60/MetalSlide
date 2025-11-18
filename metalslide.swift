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
        .windowStyle(.hiddenTitleBar)
    }
}

struct MetalView: View {
    @StateObject private var renderer = Renderer()
    
    var body: some View {
        ZStack(alignment: .topLeading) {
            MetalViewRepresentable(renderer: renderer)
                .ignoresSafeArea()
            
            if renderer.showInfo {
                VStack(alignment: .leading) {
                    Text(renderer.info)
                    
                    Toggle("Scaling", isOn: $renderer.scalingEnabled)
                        .onChange(of: renderer.scalingEnabled) { renderer.view?.needsDisplay = true }
                    
                    Toggle("Shuffle", isOn: $renderer.shuffleEnabled)
                        .onChange(of: renderer.shuffleEnabled) { renderer.toggleShuffle() }
                }
                .padding(15)
                .glassEffect(in: .rect(cornerRadius: 30))
                .padding(.leading, 8)
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
            
            context.coordinator.imagePaths = FileManager.default
                .enumerator(at: url, includingPropertiesForKeys: nil)!
                .allObjects
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
    var pipeline: MTLRenderPipelineState?
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
    @Published var shuffleEnabled = true
    
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
        residencySet = try! device.makeResidencySet(descriptor: .init())
        allocator = device.makeCommandAllocator()
        scalerFence = device.makeFence()
        commandBuffer = device.makeCommandBuffer()
        
        let shaderSource = """
            #include <metal_stdlib>
            using namespace metal;
            
            struct VertexOut {
                float4 position [[position]];
                float2 texCoord;
            };
            
            vertex VertexOut vertexShader(uint vid [[vertex_id]], constant float2 &scale [[buffer(0)]]) {
                float2 positions[6] = {
                    float2(-1,-1), float2(1,-1), float2(-1,1),
                    float2(-1,1), float2(1,-1), float2(1,1)
                };
                float2 texCoords[6] = {
                    float2(0,1), float2(1,1), float2(0,0),
                    float2(0,0), float2(1,1), float2(1,0)
                };
                return { float4(positions[vid] * scale, 0, 1), texCoords[vid] };
            }
            
            fragment half4 fragmentShader(VertexOut in [[stage_in]], texture2d<half> tex [[texture(0)]]) {
                constexpr sampler s(coord::normalized,
                                   address::clamp_to_edge,
                                   filter::linear);
                return tex.sample(s, in.texCoord);
            }
            """
        
        device.makeLibrary(source: shaderSource, options: nil) { [weak self] library, error in
            guard let self = self, let library = library else {
                print("Shader compilation error: \(error?.localizedDescription ?? "unknown")")
                return
            }
            
            let desc = MTL4RenderPipelineDescriptor()
            desc.vertexFunctionDescriptor = {
                let d = MTL4LibraryFunctionDescriptor()
                d.name = "vertexShader"
                d.library = library
                return d
            }()
            desc.fragmentFunctionDescriptor = {
                let d = MTL4LibraryFunctionDescriptor()
                d.name = "fragmentShader"
                d.library = library
                return d
            }()
            desc.colorAttachments[0].pixelFormat = view.colorPixelFormat
            
            Task {
                do {
                    self.pipeline = try await self.compiler.makeRenderPipelineState(descriptor: desc)
                    await MainActor.run { self.view?.needsDisplay = true }
                } catch {
                    print("Pipeline creation error: \(error)")
                }
            }
        }
    }
    
    func draw(in view: MTKView) {
        guard !imagePaths.isEmpty else { return }
        
        if pipeline == nil {
            initializeMetal(view)
            return
        }
        
        if textureDirty {
            textureDirty = false
            texture = try? MTKTextureLoader(device: device).newTexture(
                URL: imagePaths[currentIndex],
                options: [
                    .SRGB: false
                ])
        }
        
        guard let drawable = view.currentDrawable,
              let inputTexture = texture else { return }
        
        let viewportSize = CGSize(width: drawable.texture.width,
                                 height: drawable.texture.height)
        
        let aspectImage = CGFloat(inputTexture.width) / CGFloat(inputTexture.height)
        let aspectView  = viewportSize.width / viewportSize.height
        
        let fitSize: CGSize = aspectImage > aspectView
            ? CGSize(width: viewportSize.width,
                    height: viewportSize.width / aspectImage)
            : CGSize(width: viewportSize.height * aspectImage,
                    height: viewportSize.height)
        
        let displaySize = scalingEnabled ? fitSize : CGSize(width: inputTexture.width,
                                                          height: inputTexture.height)
        
        var scalingMode = ""
        scaler = nil
        
        if scalingEnabled {
            let needsUpscale = Int(fitSize.width) > inputTexture.width ||
                              Int(fitSize.height) > inputTexture.height
            
            if needsUpscale && MTLFXSpatialScalerDescriptor.supportsDevice(device) {
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
                    let outDesc = MTLTextureDescriptor.texture2DDescriptor(
                        pixelFormat: .rgba8Unorm,
                        width: Int(fitSize.width),
                        height: Int(fitSize.height),
                        mipmapped: false)
                    outDesc.usage = s.outputTextureUsage
                    scaler = (s, device.makeTexture(descriptor: outDesc)!)
                }
                scalingMode = "\nUpscaling: MetalFX"
            } else {
                scalingMode = "\nDownscaling: Linear"
            }
        }
        
        if let (s, output) = scaler {
            s.colorTexture = inputTexture
            s.outputTexture = output
            s.inputContentWidth = inputTexture.width
            s.inputContentHeight = inputTexture.height
        }
        
        info = """
            Slide: \(currentIndex + 1) of \(imagePaths.count)
            File: \(imagePaths[currentIndex].lastPathComponent)
            Input: \(inputTexture.width)x\(inputTexture.height)
            Output: \(Int(displaySize.width))x\(Int(displaySize.height))\(scalingMode)
            """
        
        scaleBuffer.contents()
            .assumingMemoryBound(to: SIMD2<Float>.self)
            .pointee = SIMD2(Float(displaySize.width / viewportSize.width),
                            Float(displaySize.height / viewportSize.height))
        
        residencySet.removeAllAllocations()
        residencySet.addAllocation(inputTexture)
        residencySet.addAllocation(scaleBuffer)
        if let (_, output) = scaler { residencySet.addAllocation(output) }
        residencySet.commit()
        
        let finalTexture = scaler?.1 ?? inputTexture
        
        argumentTable.setTexture(finalTexture.gpuResourceID, index: 0)
        argumentTable.setAddress(scaleBuffer.gpuAddress, index: 0)
        
        commandBuffer.beginCommandBuffer(allocator: allocator)
        commandBuffer.useResidencySet(residencySet)
        scaler?.0.encode(commandBuffer: commandBuffer)
        
        let encoder = commandBuffer.makeRenderCommandEncoder(
            descriptor: view.currentMTL4RenderPassDescriptor!,
            options: MTL4RenderEncoderOptions())!
        
        encoder.waitForFence(scalerFence, beforeEncoderStages: .fragment)
        encoder.setRenderPipelineState(pipeline!)
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
        case 53:  NSApp.terminate(nil)                           // esc
        case 34:  showInfo.toggle()                               // i
        case 123: goToSlide(currentIndex - 1)                     // left
        case 124, 49: goToSlide(currentIndex + 1)                 // right / space
        case 51:                                                  // delete
            try? FileManager.default.trashItem(at: imagePaths[currentIndex], resultingItemURL: nil)
            imagePaths.remove(at: currentIndex)
            goToSlide(currentIndex)
        case 18...29:                                             // 1-9 for timer
            autoadvanceInterval = event.charactersIgnoringModifiers!.first!.wholeNumberValue ?? autoadvanceInterval
        default: break
        }
    }
    
    func goToSlide(_ index: Int) {
        currentIndex = (index % imagePaths.count + imagePaths.count) % imagePaths.count
        textureDirty = true
        slideChangedTime = Date()
        view?.needsDisplay = true
    }
    
    func toggleShuffle() {
        let currentPath = imagePaths[currentIndex]
        imagePaths = shuffleEnabled ? imagePaths.shuffled() : imagePaths.sorted { $0.path < $1.path }
        currentIndex = imagePaths.firstIndex(of: currentPath)!
    }
}