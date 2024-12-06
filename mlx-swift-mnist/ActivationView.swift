//
//  ActivationView.swift
//  mlx-swift-mnist
//
//  Created by Ilia Sazonov on 11/25/24.
//

import SwiftUI
import AppKit
import MLX

struct ActivationPointView: View {
    let value: Double
    let range: Range<Double>
    let pointSize: CGFloat

    init(value: Double, range: Range<Double>, pointSize: CGFloat = 20) {
        self.value = value
        self.range = range
        self.pointSize = pointSize
    }

    var body: some View {
        RoundedRectangle(cornerRadius: 1)
            .fill(heatMapColor(for: value))
            .frame(width: pointSize, height: pointSize)
            .border(.gray.opacity(0.5))
            .help("\(value)")
    }

    /// Computes a color for the heat map based on the activation value.
    private func heatMapColor(for value: Double) -> Color {
        let normalizedValue = (value - range.lowerBound)/(range.upperBound - range.lowerBound)
        return Color.green.opacity(0.6).mix(with: .purple.opacity(0.8), by: normalizedValue)
    }
}


/// Mock-ups

struct ActivationView: View {
//    @Bindable var viewModel: ActivationViewModel
    let mlxArray: MLXArray
    @State private var images: [Image] = []

    init(mlxArray: MLXArray) {
        self.mlxArray = createMockMLXArray()
        self._images = State(initialValue: createImage(from: mlxArray))
    }

    var body: some View {
        let size = 150.0
        let columns = [GridItem(.adaptive(minimum: size+2))]

        ScrollView {
            LazyVGrid(columns: columns, spacing: 5) {
                ForEach(images.indices, id: \.self) { index in
                    images[index]
                        .resizable()
                        .frame(width: size, height: size) // Adjust size as needed
                }
            }
        }
    }
}


//struct ActivationView: View {
//    @Bindable var viewModel: ActivationViewModel
//    let columns = [GridItem(.adaptive(minimum: 11))]
////    let columns = Array(repeating: GridItem(.fixed(12)), count: 10)
//    var body: some View {
//        List {
//            ForEach(viewModel.activations.sorted(by: {$0.key < $1.key}), id: \.key) { key, value in
//                VStack(alignment: .leading) {
//                    HStack {
//                        Text(key)
//                        Text(value.shape[1...].description)
//                    }
//
//                    let minValue = Double(min(viewModel.activations[key]!).item(Float.self))
//                    let maxValue = Double(max(viewModel.activations[key]!).item(Float.self))
//
//                    LazyVGrid(columns: columns, alignment: .leading, spacing: 0) {
//                        ForEach(value.flattened().map( {Double($0.item(Float.self))} ).enumerated().sorted(by: {$0.offset < $1.offset}), id: \.offset ) { idx, activationValue in
//                            ActivationPointView(value: activationValue, range: minValue ..< maxValue)
//                        }
//                    }
//                }
//            }
//        }
//    }
//}

//@MainActor
//@Observable class ActivationViewModel {
//    var activations = [String: MLXArray]()
//
//
//}

// Function to create mock MLXArray data
func createMockMLXArray() -> MLXArray {
    // Define the dimensions
    let batchSize = 1
    let height = 28
    let width = 28
    let channels = 6

    // Total number of elements
    let totalElements = batchSize * height * width * channels

    // Create a flat array with mock data
    var flatData = [Float](repeating: 0.0, count: totalElements)
    for c in 0..<channels {
        let value = Float(c) / Float(channels - 1) // Values between 0 and 1
        for h in 0..<height {
            for w in 0..<width {
                let index = c + channels * (w + width * h)
                flatData[index] = value
            }
        }
    }

    // Initialize MLXArray
    let mlxArray = flatData.withUnsafeBufferPointer { ptr in MLXArray(ptr, [batchSize, height, width, channels]) }
    return mlxArray
}





#Preview {
    let mlxArray = createMockMLXArray()
    ActivationView(mlxArray: mlxArray)
        .frame(width: 300, height: 300)
}
