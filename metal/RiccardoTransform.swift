//
//  RiccardoTransform.swift
//  RichTransfTest
//
//  Created by Riccardo Cecchini on 29/09/24.
//

import MetalKit
import Foundation

struct Sinusoid {
    let frequency: Float
    let amplitude: Float
    let phase: Float
}

func decomposeSinusoidAdvance(data: [Float], halving: Float, precision: UInt32, maxHalvings: UInt32, referenceSize: Float, negligible: Float) -> ([Sinusoid], [Float], [Float]) {

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
          let decomposeKernel = library.makeFunction(name: "decompose_sinusoid_adv_kernel") else {
        fatalError("Could not load the Metal function")
    }

    // 4. Create a Metal pipeline state
    guard let pipelineState = try? device.makeComputePipelineState(function: decomposeKernel) else {
        fatalError("Could not create a pipeline state")
    }

    // 5. Prepare the input and output data buffers
    let dataCount = data.count
    let dataBuffer = device.makeBuffer(bytes: data, length: dataCount * MemoryLayout<Float>.stride, options: [])!
    let sinusoidsCount = UInt32(dataCount * Int(precision / 2))
    let sinusoidsBuffer = device.makeBuffer(length: Int(sinusoidsCount) * MemoryLayout<Sinusoid>.stride, options: [])!
    let residueBuffer = device.makeBuffer(length: dataCount * MemoryLayout<Float>.stride, options: [])!
    let resultantBuffer = device.makeBuffer(length: dataCount * MemoryLayout<Float>.stride, options: [])!
    let tempResidueBuffer = device.makeBuffer(length: dataCount * MemoryLayout<Float>.stride, options: [])!
    let tempResultantBuffer = device.makeBuffer(length: dataCount * MemoryLayout<Float>.stride, options: [])!
    let peaksBuffer = device.makeBuffer(length: dataCount * MemoryLayout<Int>.stride, options: [])!
    let peaksCount = data.count < 1024 ? data.count : 1024
    let currentSignalBuffer = device.makeBuffer(length: dataCount * MemoryLayout<Float>.stride, options: [])!
    let currentResultantBuffer = device.makeBuffer(length: dataCount * MemoryLayout<Float>.stride, options: [])!

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
    computeEncoder.setBuffer(tempResidueBuffer, offset: 0, index: 4)
    computeEncoder.setBuffer(tempResultantBuffer, offset: 0, index: 5)

    // Set the constant values
    var halvingVar = halving
    computeEncoder.setBytes(&halvingVar, length: MemoryLayout<Float>.stride, index: 6)
    var precisionVar = precision
    computeEncoder.setBytes(&precisionVar, length: MemoryLayout<UInt32>.stride, index: 7)
    var maxHalvingsVar = maxHalvings
    computeEncoder.setBytes(&maxHalvingsVar, length: MemoryLayout<UInt32>.stride, index: 8)
    var referenceSizeVar = referenceSize
    computeEncoder.setBytes(&referenceSizeVar, length: MemoryLayout<Float>.stride, index: 9)
    var negligibleVar = negligible
    computeEncoder.setBytes(&negligibleVar, length: MemoryLayout<Float>.stride, index: 10)
    var sinusoidsCountVar = sinusoidsCount
    computeEncoder.setBytes(&sinusoidsCountVar, length: MemoryLayout<UInt32>.stride, index: 11)
    var peaksCountVar = UInt32(peaksCount)
    computeEncoder.setBytes(&peaksCountVar, length: MemoryLayout<UInt32>.stride, index: 12)

    computeEncoder.setBuffer(peaksBuffer, offset: 0, index: 13)
    computeEncoder.setBuffer(currentSignalBuffer, offset: 0, index: 14)
    computeEncoder.setBuffer(currentResultantBuffer, offset: 0, index: 15)

    // 8. Set the threadgroup and grid dimensions
    let threadsPerThreadgroup = MTLSizeMake(pipelineState.maxTotalThreadsPerThreadgroup, 1, 1)
    let threadsPerGrid = MTLSizeMake(dataCount, 1, 1)
    computeEncoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)

    // 9. End the encoding and commit the command buffer
    computeEncoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    // 10. Get the results from the output buffers
    let sinusoids = Array(UnsafeBufferPointer(start: sinusoidsBuffer.contents().bindMemory(to: Sinusoid.self, capacity: Int(sinusoidsCount)), count: Int(sinusoidsCount)))
    let residue = Array(UnsafeBufferPointer(start: residueBuffer.contents().bindMemory(to: Float.self, capacity: dataCount), count: dataCount))
    let resultant = Array(UnsafeBufferPointer(start: resultantBuffer.contents().bindMemory(to: Float.self, capacity: dataCount), count: dataCount))
    
    // read them for debug purposes
    //let peaks = Array(UnsafeBufferPointer(start: peaksBuffer.contents().bindMemory(to: Int32.self, capacity: dataCount), count: peaksCount))

    return (sinusoids, residue, resultant)
}

func combineSinusoids(sinusoids1: [Sinusoid], sinusoids2: [Sinusoid], minimum: Float = 0.05) -> [Sinusoid] {
    var combinedSinusoids: [Float: (amplitude: Float, phase: Float)] = [:]

    // Process both sinusoid arrays
    for sinusoids in [sinusoids1, sinusoids2] {
        for sinusoid in sinusoids {
            let freq = sinusoid.frequency
            let amp = sinusoid.amplitude
            let phase = sinusoid.phase

            if let existing = combinedSinusoids[freq] {
                // Simple check to potentially skip combining very small amplitudes
                if min(existing.amplitude, amp) / max(existing.amplitude, amp) < (minimum * 2) {
                    if amp > existing.amplitude {
                        combinedSinusoids[freq] = (amplitude: amp, phase: phase)
                    }
                    continue // Skip to the next sinusoid
                }

                let complexSinusoid = Complex(length: amp, phase: phase) + Complex(length: existing.amplitude, phase: existing.phase)
                combinedSinusoids[freq] = (amplitude: complexSinusoid.length, phase: complexSinusoid.phase)
            } else {
                combinedSinusoids[freq] = (amplitude: amp, phase: phase)
            }
        }
    }

    // Convert back to Sinusoid structs and filter
    let result = combinedSinusoids.compactMap { (freq, values) -> Sinusoid? in
        let amplitude = values.amplitude
        let phase = values.phase
        if amplitude > minimum {
            return Sinusoid(frequency: freq, amplitude: amplitude, phase: phase)
        } else {
            return nil
        }
    }.sorted { $0.frequency < $1.frequency }

    return result
}

struct Complex {
    let length: Float
    let phase: Float

    init(length: Float, phase: Float) {
        self.length = length
        self.phase = phase
    }

    static func +(lhs: Complex, rhs: Complex) -> Complex {
        let real = lhs.length * cos(lhs.phase) + rhs.length * cos(rhs.phase)
        let imag = lhs.length * sin(lhs.phase) + rhs.length * sin(rhs.phase)
        let length = sqrt(real * real + imag * imag)
        let phase = atan2(imag, real)
        return Complex(length: length, phase: phase)
    }
}

// Example input data
func exampleUsage(){
    let length = 100
    let refPi = Float.pi / (Float(length) / 2.0)
    
    let data: [Float] = (0..<length).map { x in
        let xDouble = Float(x)
        return sin(refPi * xDouble) +
        (sin((refPi * xDouble * 2) + (Float.pi / 4.0)) * 0.5) //+
        //sin(refPi * xDouble * 3) +
        //sin(refPi * xDouble * 8)
    }
    
    // Call the function with your desired parameters
    let (sinusoids, residue, resultant) = decomposeSinusoidAdvance(data: data, halving: 2.0, precision: 10, maxHalvings: 50, referenceSize: 1.0, negligible: 0.01)

    // Print the extracted sinusoids
    print("Extracted Sinusoids:")
    for sinusoid in sinusoids {
       print("Frequency: \(sinusoid.frequency), Amplitude: \(sinusoid.amplitude), Phase: \(sinusoid.phase)")
    }

    // Print the reconstructed signal (resultant)
    print("\nReconstructed Signal:")
    for value in residue {
       print(value)
    }
    
    print("End execution")
}

