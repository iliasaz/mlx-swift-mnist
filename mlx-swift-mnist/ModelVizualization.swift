//
//  ModelVizualization.swift
//  mlx-swift-mnist
//
//  Created by Ilia Sazonov on 12/2/24.
//

import Foundation
import MLX
import MLXNN
import SwiftUI
//import AppKit

/// Model Visualization
@MainActor
@Observable open class ModelVizualization {
    var activations = [String: MLXArray]()
    private var model: TrainableModel?

    func setModel(_ model: TrainableModel?) {
        self.model = model
    }

    func clearActivations() {
        activations.removeAll()
    }

    func predictionDistribution(for x: MLXArray, withActivations: Bool = false) -> (probabilities: [Double], highestIndex: Int)? {
        guard let model, x.size > 0 else { return nil }
        let y: MLXArray
        if withActivations {
            y = model(x, saveActivations: { self.activations = $0 }).flattened()
        } else {
            y = model(x).flattened()
        }
        guard y.size > 0 else { return nil }
        let highestIndex = argMax(y).item(Int.self)
        return (probabilities: y.map {Double($0.item(Float.self))}, highestIndex: highestIndex)
    }

    @ViewBuilder @MainActor var view: some View {
        if let model {
            List {
                ForEach(model.children().flattened(), id: \.0) { key, module in
                    VStack(alignment: .leading) {
                        HStack {
                            Text(key)
                            Text(module.description)
                        }
                        module.view(activations: self.activations[key] ?? MLXArray())
                    }
                }
            }
            .alternatingRowBackgrounds()
            .scrollIndicators(.visible, axes: .vertical)
        } else {
            Text("Model not loaded.")
        }
    }
}


public protocol VizualizableModule {
    associatedtype Body: View

    @ViewBuilder @MainActor
    func view(activations: MLXArray) -> Body
}

extension VizualizableModule {
    @ViewBuilder @MainActor
    public func view(activations: MLXArray) -> some View {
        if activations.size == 0 {
            Text("No activations")
        } else {
            switch self {
                case is Linear: let layer = self as! Linear; layer.view(activations: activations)
                case is Softmax: let layer = self as! Softmax; layer.view(activations: activations)
                case is Conv2d: let layer = self as! Conv2d; layer.view(activations: activations)
                case is MaxPool2d: let layer = self as! MaxPool2d; layer.view(activations: activations)
                default: Text("Module")
            }
        }
    }
}

extension Module: VizualizableModule {}

// Layer activation vizualizations

/// Linear layer activations are visualized as a set of dots each corresponding to a neuron activation value and represented on a scale from green (low activation) to purple (high activation)
extension Linear {
    @ViewBuilder @MainActor
    func view(activations: MLXArray) -> some View {
        let minValue = Double(min(activations).item(Float.self))
        let maxValue = Double(max(activations).item(Float.self))
        let columns = Array(repeating: GridItem(.fixed(12)), count: 10)

        LazyVGrid(columns: columns, alignment: .leading, spacing: 0) {
            ForEach(activations.flattened().map( {Double($0.item(Float.self))} ).enumerated().sorted(by: {$0.offset < $1.offset}), id: \.offset ) { idx, activationValue in
                ActivationPointView(value: activationValue, range: minValue ..< maxValue)
            }
        }
    }
}

/// Softmax layer activations are visualized similarly to ones in Linear layer - as a set of dots each corresponding to a neuron activation value and represented on a scale from green (low activation) to purple (high activation)
extension Softmax {
    @ViewBuilder @MainActor
    func view(activations: MLXArray) -> some View {
        let minValue = Double(min(activations).item(Float.self))
        let maxValue = Double(max(activations).item(Float.self))
        let columns = [GridItem(.adaptive(minimum: 11))]

        LazyVGrid(columns: columns, alignment: .leading, spacing: 0) {
            ForEach(activations.flattened().map( {Double($0.item(Float.self))} ).enumerated().sorted(by: {$0.offset < $1.offset}), id: \.offset ) { idx, activationValue in
                ActivationPointView(value: activationValue, range: minValue ..< maxValue)
            }
        }
    }
}

/// Conv2 layer activations are visualized as a set of images each corresponding a feature extracted in a given channel. The activation value in a feature and represented on a scale from green (low activation) to purple (high activation)
extension Conv2d {
    @ViewBuilder @MainActor
    func view(activations: MLXArray) -> some View {
//        let _ = print(activations.shape)
        let images = createImage(from: activations)
        if  activations.shape.count == 4 {
            let columns = [GridItem(.adaptive(minimum: 152))]

            LazyVGrid(columns: columns, spacing: 5) {
                ForEach(images.indices, id: \.self) { index in
                    images[index]
                        .resizable()
                        .frame(width: 150, height: 150) // Adjust size as needed
                }
            }
        } else { EmptyView() }
    }
}

/// Since Maxpool2 typically goes after Conv2d, we'll use the same visialization technique
extension MaxPool2d {
    @ViewBuilder @MainActor
    func view(activations: MLXArray) -> some View {
        let images = createImage(from: activations)
        if  activations.shape.count == 4 {
            let columns = [GridItem(.adaptive(minimum: 152))]

            LazyVGrid(columns: columns, spacing: 5) {
                ForEach(images.indices, id: \.self) { index in
                    images[index]
                        .resizable()
                        .frame(width: 150, height: 150) // Adjust size as needed
                }
            }
        } else { EmptyView() }
    }
}


/// This is used ti visualize Conv2d layer as a set of images each corresponding to a feature
func createImage(from array: MLXArray) -> [Image] {
    // Ensure the array is evaluated
    array.eval()

    // Get the data from the array
    let dataArray = array.asData(access: .noCopy)

    // Extract necessary properties
    guard let dataPointer = dataArray.data.withUnsafeBytes({ $0.baseAddress?.assumingMemoryBound(to: Float.self) }),
          !dataArray.shape.isEmpty,
          dataArray.shape.count >= 3 else {
        print("Invalid array shape or data")
        return []
    }

    // Assuming the shape is [batchSize, height, width, channels]
    guard dataArray.shape[0] == 1 else {
        print("The batch has more than one element. Only single-element batches are supported.")
        return []
    }
    let height = dataArray.shape[1]
    let width = dataArray.shape[2]
    let channels = dataArray.shape.count == 4 ? dataArray.shape[3] : 1

    var images: [Image] = []

    for index in 0 ..< channels {
        // Calculate the offset for the specific image in the batch
        let startIndex = index * height * width
        let endIndex = startIndex + (height * width)

        // Extract the specific image data
        var imageData: [Float] = []
        for i in startIndex..<endIndex {
            imageData.append(dataPointer[i])
        }

        // Convert float array to byte array (assuming grayscale for simplicity)
        var byteArray: [(red: UInt8, green: UInt8, blue: UInt8, aplha: UInt8)] = []
        // get min and max values so we can normalize the array values
        let minValue: Float = imageData.min()!
        let maxValue: Float = imageData.max()!
        for pixelValue in imageData {
            if maxValue == minValue {
                byteArray.append((0, 0, 0, 0))
            } else {
                let normalizedValue = (pixelValue - minValue) / (maxValue - minValue)
                // Map normalized value to color between green and purple
                let red: UInt8 = UInt8(normalizedValue * 128.0) // From 0 to 128
                let green: UInt8 = UInt8(255.0 - (normalizedValue * 255.0)) // From 255 to 0
                let blue: UInt8 = UInt8(normalizedValue * 128.0) // From 0 to 128

                // Calculate alpha value between 60% and 80%
                let alpha: UInt8 = UInt8(192 + (normalizedValue * 48)) // From 153 (60%) to 204 (80%)

                byteArray.append((red, green, blue, alpha))
            }
        }

        // Create an NSImage from the byte array
        let bitmapRep = NSBitmapImageRep(
            bitmapDataPlanes: nil,
            pixelsWide: width,
            pixelsHigh: height,
            bitsPerSample: 8,
            samplesPerPixel: 4, // RGBA image
            hasAlpha: true,
            isPlanar: false,
            colorSpaceName: .calibratedRGB,
            bytesPerRow: width * 4,
            bitsPerPixel: 32
        )

        guard let bitmapData = bitmapRep?.bitmapData else {
            print("Failed to create bitmap data")
            continue
        }

        let _ = byteArray.withUnsafeBytes { ptr in
            memcpy(bitmapData, ptr.baseAddress!, byteArray.count * MemoryLayout<(red: UInt8, green: UInt8, blue: UInt8, alpha: UInt8)>.size)
        }

        let nsImage = NSImage(size: NSSize(width: width, height: height))
        nsImage.addRepresentation(bitmapRep!)
        images.append(Image(nsImage: nsImage))
    }
    return images
}
