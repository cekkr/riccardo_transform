import MetalKit
import Foundation

func decomposeSinusoid(data: [Float], halving: Float, precision: UInt32, maxHalvings: UInt32, referenceSize: Float, negligible: Float) -> ([Sinusoid], [Float]) {

    // 1. Get the Metal device
    guard let device = MTLCreateSystemDefaultDevice() else {
        fatalError("Metal is not supported on this device")
    }

    // 2. Create a Metal command queue
    guard let commandQueue = device.makeCommandQueue() else {
        fatalError("Could not create a command queue")
    }

    // 3. Get the Metal library and kernel function
    guard let library = device.makeDefaultLibrary(),
          let decomposeKernel = library.makeFunction(name: "decompose_sinusoid_kernel") else {
        fatalError("Could not load the Metal function")
    }

    // 4. Create a Metal pipeline state
    guard let pipelineState = try? device.makeComputePipelineState(function: decomposeKernel) else {
        fatalError("Could not create a pipeline state")
    }

    // 5. Prepare the input and output data buffers
    let dataCount = data.count
    let dataBuffer = device.makeBuffer(bytes: data, length: dataCount * MemoryLayout<Float>.stride, options: [])!
    let sinusoidsBuffer = device.makeBuffer(length: dataCount * MemoryLayout<Sinusoid>.stride, options: [])!
    let residueBuffer = device.makeBuffer(length: dataCount * MemoryLayout<Float>.stride, options: [])!
    let resultantBuffer = device.makeBuffer(length: dataCount * MemoryLayout<Float>.stride, options: [])!

    // 6. Create a Metal command buffer and encoder
    guard let commandBuffer = commandQueue.makeCommandBuffer(),
          let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
        fatalError("Could not create a command buffer or encoder")
    }

    // 7. Set the pipeline state and buffers for the compute encoder
    computeEncoder.setComputePipelineState(pipelineState)
    computeEncoder.setBuffer(dataBuffer, offset: 0, index: 0)
    computeEncoder.setBuffer(sinusoidsBuffer, offset: 0, index: 1)
    computeEncoder.setBuffer(residueBuffer, offset: 0, index: 2)
    computeEncoder.setBuffer(resultantBuffer, offset: 0, index: 3)

    // Set the constant values
    var halvingVar = halving
    computeEncoder.setBytes(&halvingVar, length: MemoryLayout<Float>.stride, index: 4)
    var precisionVar = precision
    computeEncoder.setBytes(&precisionVar, length: MemoryLayout<UInt32>.stride, index: 5)
    var maxHalvingsVar = maxHalvings
    computeEncoder.setBytes(&maxHalvingsVar, length: MemoryLayout<UInt32>.stride, index: 6)
    var referenceSizeVar = referenceSize
    computeEncoder.setBytes(&referenceSizeVar, length: MemoryLayout<Float>.stride, index: 7)
    var negligibleVar = negligible
    computeEncoder.setBytes(&negligibleVar, length: MemoryLayout<Float>.stride, index: 8)

    // 8. Set the threadgroup and grid dimensions
    let threadsPerThreadgroup = MTLSizeMake(pipelineState.maxTotalThreadsPerThreadgroup, 1, 1) 
    let threadsPerGrid = MTLSizeMake(dataCount, 1, 1)
    computeEncoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)

    // 9. End the encoding and commit the command buffer
    computeEncoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    // 10. Get the results from the output buffers
    let sinusoids = Array(UnsafeBufferPointer(start: sinusoidsBuffer.contents().bindMemory(to: Sinusoid.self, capacity: dataCount), count: dataCount))
    let resultant = Array(UnsafeBufferPointer(start: resultantBuffer.contents().bindMemory(to: Float.self, capacity: dataCount), count: dataCount))

    return (sinusoids, resultant)
}

// Example input data

let length = 100
let refPi = Double.pi / (Double(length) / 2.0)

let data: [Double] = (0..<length).map { x in
    let xDouble = Double(x)
    return sin(refPi * xDouble) +
           (sin((refPi * xDouble * 2) + (Double.pi / 4.0)) * 0.5) +
           sin(refPi * xDouble * 3) +
           sin(refPi * xDouble * 8)
}

// Call the function with your desired parameters
let (sinusoids, resultant) = decomposeSinusoid(data: data, halving: 2.0, precision: 10, maxHalvings: 50, referenceSize: 1.0, negligible: 0.01)

// Now you have the extracted sinusoids in the 'sinusoids' array
// and the reconstructed signal in the 'resultant' array