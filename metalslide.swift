import SwiftUI
import Metal
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
                        .onChange(of: renderer.scalingEnabled) { renderer.draw() }

                    Toggle("Shuffle", isOn: $renderer.shuffleEnabled)
                        .onChange(of: renderer.shuffleEnabled) { renderer.toggleShuffle() }
                }
                .padding(15)
                .glassEffect(in: .rect(cornerRadius: 30))
                .padding(.leading, 8)
            }
        }
        .focusable()
        .focusEffectDisabled()
        .onKeyPress(.space) { renderer.goToSlide(renderer.currentIndex + 1); return .handled }
        .onKeyPress(.leftArrow) { renderer.goToSlide(renderer.currentIndex - 1); return .handled }
        .onKeyPress(.rightArrow) { renderer.goToSlide(renderer.currentIndex + 1); return .handled }
        .onKeyPress(.delete) { renderer.deleteCurrentSlide(); return .handled }
        .onKeyPress(.escape) { NSApp.terminate(nil); return .handled }
        .onKeyPress("i") { renderer.showInfo.toggle(); return .handled }
        .onKeyPress("0") { renderer.autoadvanceInterval = 0; return .handled }
        .onKeyPress("1") { renderer.autoadvanceInterval = 1; return .handled }
        .onKeyPress("2") { renderer.autoadvanceInterval = 2; return .handled }
        .onKeyPress("3") { renderer.autoadvanceInterval = 3; return .handled }
        .onKeyPress("4") { renderer.autoadvanceInterval = 4; return .handled }
        .onKeyPress("5") { renderer.autoadvanceInterval = 5; return .handled }
        .onKeyPress("6") { renderer.autoadvanceInterval = 6; return .handled }
        .onKeyPress("7") { renderer.autoadvanceInterval = 7; return .handled }
        .onKeyPress("8") { renderer.autoadvanceInterval = 8; return .handled }
        .onKeyPress("9") { renderer.autoadvanceInterval = 9; return .handled }
    }
}

class MetalNSView: NSView {
    var renderer: Renderer?
    let metalLayer = CAMetalLayer()

    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        wantsLayer = true
        layer = metalLayer
    }

    required init?(coder: NSCoder) {
        super.init(coder: coder)
        wantsLayer = true
        layer = metalLayer
    }

    override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        updateDrawableSize()
    }

    override func setFrameSize(_ newSize: NSSize) {
        super.setFrameSize(newSize)
        updateDrawableSize()
    }

    func updateDrawableSize() {
        guard let window = window else { return }
        let scale = window.backingScaleFactor
        let size = bounds.size
        metalLayer.drawableSize = CGSize(width: size.width * scale, height: size.height * scale)
        renderer?.draw()
    }
}

struct MetalViewRepresentable: NSViewRepresentable {
    let renderer: Renderer

    func makeCoordinator() -> Renderer { renderer }

    func makeNSView(context: Context) -> MetalNSView {
        let view = MetalNSView()
        view.metalLayer.device = MTLCreateSystemDefaultDevice()!
        view.metalLayer.pixelFormat = .bgra8Unorm
        view.metalLayer.framebufferOnly = false
        view.renderer = context.coordinator
        context.coordinator.metalLayer = view.metalLayer
        context.coordinator.initializeMetal()

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

            context.coordinator.draw()
        }

        Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { _ in
            let r = context.coordinator
            if r.autoadvanceInterval > 0 && Date().timeIntervalSince(r.slideChangedTime) >= Double(r.autoadvanceInterval) {
                r.goToSlide(r.currentIndex + 1)
            }
        }

        return view
    }

    func updateNSView(_ nsView: MetalNSView, context: Context) {}
}

class Renderer: NSObject, ObservableObject {
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
    var metalLayer: CAMetalLayer!

    var scaleBuffer: MTLBuffer!
    var compiler: MTL4Compiler!
    var scaler: MTL4FXSpatialScaler?
    var scalerOutputTexture: MTLTexture?
    var cachedScalerSize: CGSize?
    var scalerFence: MTLFence!
    var commandBuffer: MTL4CommandBuffer!

    @Published var info = ""
    @Published var showInfo = false
    @Published var scalingEnabled = true
    @Published var shuffleEnabled = true

    var autoadvanceInterval = 0
    var slideChangedTime = Date()
    var isDrawing = false
    var needsRedraw = false
    
    func loadTexture(from url: URL) -> MTLTexture? {
        guard let image = NSImage(contentsOf: url),
              let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else { return nil }

        let textureDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm,
            width: cgImage.width,
            height: cgImage.height,
            mipmapped: false)
        textureDesc.usage = [.shaderRead]

        guard let texture = device.makeTexture(descriptor: textureDesc) else { return nil }

        let context = CGContext(
            data: nil,
            width: cgImage.width,
            height: cgImage.height,
            bitsPerComponent: 8,
            bytesPerRow: cgImage.width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!

        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: cgImage.width, height: cgImage.height))

        texture.replace(
            region: MTLRegionMake2D(0, 0, cgImage.width, cgImage.height),
            mipmapLevel: 0,
            withBytes: context.data!,
            bytesPerRow: cgImage.width * 4)

        return texture
    }

    func initializeMetal() {
        device = metalLayer.device
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
            desc.colorAttachments[0].pixelFormat = self.metalLayer.pixelFormat

            Task {
                do {
                    self.pipeline = try await self.compiler.makeRenderPipelineState(descriptor: desc)
                    self.draw()
                } catch {
                    print("Pipeline creation error: \(error)")
                }
            }
        }
    }
    
    func draw() {
        guard !imagePaths.isEmpty, let pipeline = pipeline else { return }
        if isDrawing {
            needsRedraw = true
            return
        }
        isDrawing = true
        needsRedraw = false
        defer {
            isDrawing = false
            if needsRedraw { draw() }
        }

        if textureDirty {
            textureDirty = false
            texture = loadTexture(from: imagePaths[currentIndex])
        }

        guard let inputTexture = texture,
              let drawable = metalLayer.nextDrawable() else { return }

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
        let needsUpscale = scalingEnabled && (Int(fitSize.width) > inputTexture.width ||
                                               Int(fitSize.height) > inputTexture.height)

        if needsUpscale {
            if cachedScalerSize != fitSize || scaler == nil {
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
                    outDesc.storageMode = .private
                    scaler = s
                    scalerOutputTexture = device.makeTexture(descriptor: outDesc)
                    cachedScalerSize = fitSize
                }
            }
            scalingMode = "\nUpscaling: MetalFX"
        } else {
            scaler = nil
            scalerOutputTexture = nil
            cachedScalerSize = nil
            if scalingEnabled {
                scalingMode = "\nDownscaling: Linear"
            }
        }

        if let s = scaler, let output = scalerOutputTexture {
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
        if let output = scalerOutputTexture { residencySet.addAllocation(output) }
        residencySet.commit()

        let finalTexture = scalerOutputTexture ?? inputTexture
        
        argumentTable.setTexture(finalTexture.gpuResourceID, index: 0)
        argumentTable.setAddress(scaleBuffer.gpuAddress, index: 0)
        
        let passDesc = MTL4RenderPassDescriptor()
        passDesc.colorAttachments[0].texture = drawable.texture
        passDesc.colorAttachments[0].loadAction = .clear
        passDesc.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        passDesc.colorAttachments[0].storeAction = .store

        commandBuffer.beginCommandBuffer(allocator: allocator)
        commandBuffer.useResidencySet(residencySet)
        scaler?.encode(commandBuffer: commandBuffer)

        let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: passDesc)!

        if scaler != nil {
            encoder.waitForFence(scalerFence, beforeEncoderStages: .fragment)
        }
        encoder.setRenderPipelineState(pipeline)
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

    func deleteCurrentSlide() {
        guard !imagePaths.isEmpty else { return }
        try? FileManager.default.trashItem(at: imagePaths[currentIndex], resultingItemURL: nil)
        imagePaths.remove(at: currentIndex)
        goToSlide(currentIndex)
    }
    
    func goToSlide(_ index: Int) {
        currentIndex = (index % imagePaths.count + imagePaths.count) % imagePaths.count
        textureDirty = true
        slideChangedTime = Date()
        draw()
    }
    
    func toggleShuffle() {
        let currentPath = imagePaths[currentIndex]
        imagePaths = shuffleEnabled ? imagePaths.shuffled() : imagePaths.sorted { $0.path < $1.path }
        currentIndex = imagePaths.firstIndex(of: currentPath)!
    }
}