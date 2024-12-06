//
//  TrainingChartView.swift
//  mlx-swift-mnist
//
//  Created by Ilia Sazonov on 11/14/24.
//

import SwiftUI
import Charts

struct TrainingChartView: View {
    @Bindable var viewModel: ViewModel  // Ensure viewModel is bindable

    // State variable to hold the selected epoch
    @State private var selectedEpoch: Int?
    @Environment(\.colorScheme) var colorScheme

    var title: some View {
        VStack(alignment: .leading) {
            Text("Training Progress")
                .font(.headline)
                .foregroundStyle(.secondary)
            Text("Losses and Accuracy")
                .font(.title.bold())

            HStack {
                HStack(spacing: 5) {
                    legendCircle
                    Text("Training")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }

                HStack(spacing: 5) {
                    legendSquare
                    Text("Testing")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
            }
        }
    }

    var body: some View {
        VStack(alignment: .leading) {
            title
                .opacity(selectedEpoch == nil ? 1 : 0)

            lossChart
                .padding(.bottom)

            accuracyChart
                .padding(.top)
        }
        .padding()
    }

    @ViewBuilder
    var lossChart: some View {
    // Loss Chart with interactivity
        Chart {
            ForEach(lossData, id: \.epoch) { dataPoint in
                LineMark(
                    x: .value("Epoch", dataPoint.epoch),
                    y: .value("Loss", dataPoint.lossValue)
                )
                .foregroundStyle(by: .value("Type", dataPoint.lossType))
                .symbol(by: .value("Type", dataPoint.lossType))
                .interpolationMethod(.catmullRom)

            }

            // Add a RuleMark to indicate the selected epoch
            if let selectedEpoch = selectedEpoch {
                RuleMark(x: .value("Epoch", selectedEpoch))
                    .lineStyle(StrokeStyle(dash: [5, 5]))
                    .offset(yStart: -20)
                    .zIndex(-1)
                    .annotation(
                        position: .top,
                        alignment: .center,
                        spacing: 0,
                        overflowResolution: .init(
                            x: .fit(to: .chart),
                            y: .disabled
                        )
                    ) {
                        lossSelectionPopover
                    }
            }
        }
        .chartXSelection(value: $selectedEpoch)
        .chartYAxisLabel("Loss")
        .chartXAxisLabel("Epoch")
        .chartForegroundStyleScale { colorPerLossType[$0]! }
    }

    @ViewBuilder
    var lossSelectionPopover: some View {
        if let selectedEpoch, let selectedLosses = getLosses(for: selectedEpoch) {
            VStack {
                Text("Epoch \(selectedEpoch)")
                    .font(.headline)
                    .foregroundColor(.secondary)

                HStack(spacing: 20) {
                    Text("\(selectedLosses.trainingLoss,  specifier: "%.2f")")
                        .font(.title.bold())
                        .foregroundColor(colorPerLossType["Training"])
                        .blendMode(colorScheme == .light ? .plusDarker : .normal)

                    Text("\(selectedLosses.testLoss,  specifier: "%.2f")")
                        .font(.title.bold())
                        .foregroundColor(colorPerLossType["Testing"])
                        .blendMode(colorScheme == .light ? .plusDarker : .normal)
                }
            }
            .background {
                RoundedRectangle(cornerRadius: 4)
                    .foregroundStyle(Color.gray.opacity(0.12))
            }
        } else {
            EmptyView()
        }
    }

    func getLosses(for epoch: Int) -> (trainingLoss: Float, testLoss: Float)? {
        guard let losses = viewModel.trainingProgress.first(where: {$0.epoch == epoch}) else { return nil }
        return (trainingLoss: losses.trainingLoss, testLoss: losses.testLoss)

    }

    @ViewBuilder
    var accuracyChart: some View {
    // Accuracy Chart with interactivity
        Chart {
            ForEach(viewModel.trainingProgress, id: \.epoch) { dataPoint in
                LineMark(
                    x: .value("Epoch", dataPoint.epoch),
                    y: .value("Accuracy", dataPoint.accuracy)
                )
                .foregroundStyle(.blue)
            }

            // Add a RuleMark to indicate the selected epoch
            if let selectedEpoch = selectedEpoch {
                RuleMark(x: .value("Epoch", selectedEpoch))
                    .lineStyle(StrokeStyle(dash: [5, 5]))
                    .offset(yStart: -20)
                    .zIndex(-1)
                    .annotation(
                        position: .top,
                        alignment: .center,
                        spacing: 0,
                        overflowResolution: .init(
                            x: .fit(to: .chart),
                            y: .disabled
                        )
                    ) {
                        accuracySelectionPopover
                    }
            }
        }
        .chartXSelection(value: $selectedEpoch)
        .chartYAxisLabel("Accuracy")
        .chartXAxisLabel("Epoch")
    }

    @ViewBuilder
    var accuracySelectionPopover: some View {
        if let selectedEpoch, let selectedAccuracy = viewModel.trainingProgress.first(where: { $0.epoch == selectedEpoch })?.accuracy  {
            VStack {
                Text("Epoch \(selectedEpoch)")
                    .font(.headline)
                    .foregroundColor(.secondary)

                HStack(spacing: 20) {
                    Text("\(selectedAccuracy * 100,  specifier: "%.2f")%")
                        .font(.title.bold())
                        .foregroundColor(.blue)
                        .blendMode(colorScheme == .light ? .plusDarker : .normal)
                }
            }
            .background {
                RoundedRectangle(cornerRadius: 4)
                    .foregroundStyle(Color.gray.opacity(0.12))
            }
        } else {
            EmptyView()
        }
    }

    fileprivate var lossData: [LossData] {
        var dataPoints = [LossData]()
        viewModel.trainingProgress.forEach {
            let trainingDataPoint = LossData(epoch: $0.epoch, lossType: .training, lossValue: $0.trainingLoss)
            dataPoints.append(trainingDataPoint)
            let testDataPoint = LossData(epoch: $0.epoch, lossType: .testing, lossValue: $0.testLoss)
            dataPoints.append(testDataPoint)
        }
        return dataPoints
    }
}

fileprivate struct LossData {
    let epoch: Int
    let lossType: LossTypePlottable
    let lossValue: Float
}

fileprivate enum LossTypePlottable: String, Plottable {
    case training = "Training"
    case testing = "Testing"
}

fileprivate let colorPerLossType: [String: Color] = [
    "Training": .purple,
    "Testing": .green
]

@ViewBuilder
var legendSquare: some View {
    RoundedRectangle(cornerRadius: 1)
        .stroke(lineWidth: 2)
        .frame(width: 5.3, height: 5.3)
        .foregroundColor(.green)
        .padding(EdgeInsets(top: 0, leading: 2, bottom: 0, trailing: 0))
}

@ViewBuilder
var legendCircle: some View {
    Circle()
        .stroke(lineWidth: 2)
        .frame(width: 5.7, height: 5.7)
        .foregroundColor(.purple)
        .padding(EdgeInsets(top: 0, leading: 2, bottom: 0, trailing: 0))
}




#Preview {
    TrainingChartView(viewModel: .preview)
//    TrainingProgressView(viewModel: .preview)
}
