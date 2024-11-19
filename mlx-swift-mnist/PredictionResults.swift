//
//  PredictionResults.swift
//  mlx-swift-mnist
//
//  Created by Ilia Sazonov on 11/18/24.
//

import SwiftUI

struct PredictionResults: View {
    @Binding var preditedDistribution: [Prediction]
    var body: some View {
        List {
            HStack {
                Text("Value")
                    .frame(width: 50)
                Text("Probability")
                    .frame(width: 100)            }
            .font(.headline)

            ForEach(preditedDistribution, id: \.self) { prediction in
                HStack {
                    Text("\(prediction.value)")
                        .frame(width: 50)
                    Text(prediction.probability.formatted(.number))
                        .frame(width: 100)
                }
                .font(prediction.isBest ? .title3.bold() : nil)
                .foregroundColor(prediction.isBest ? .green : nil)
            }
        }
        .alternatingRowBackgrounds()
    }
}

struct Prediction: Hashable {
    let value: Int
    let probability: Float
    let isBest: Bool

    init(value: Int, probability: Float, isBest: Bool = false) {
        self.value = value
        self.probability = probability
        self.isBest = isBest
    }
}

let mockPredictions: [Prediction] = [
    .init(value: 0, probability: 0.1),
    .init(value: 1, probability: 0.2),
    .init(value: 2, probability: 0.9, isBest: true),
    .init(value: 3, probability: 0.3),
    .init(value: 4, probability: 0.4),
    .init(value: 5, probability: 0.5),
    .init(value: 6, probability: 0.6),
    .init(value: 7, probability: 0.4),
    .init(value: 8, probability: 0.3),
    .init(value: 9, probability: 0.2),
    ]


#Preview {
    @Previewable @State var predictions = mockPredictions
    PredictionResults(preditedDistribution: $predictions)
}
