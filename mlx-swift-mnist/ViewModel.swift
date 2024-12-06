//
//  ViewModel.swift
//  mlx-swift-mnist
//
//  Created by Ilia Sazonov on 11/24/24.
//

import Foundation
import SwiftUI
import TabularData
import MLX
import MLXOptimizers
import UniformTypeIdentifiers

@MainActor
@Observable class ViewModel {
    enum ModelType: String, CaseIterable, Identifiable {
        case leNet = "LeNet"
        case mlp = "MLP"

        var id: String { rawValue }
    }

    enum DataLimitRows: Equatable {
        case some(Int), all

        // Converts between enum cases and stored integer values
        init(from limit: Int?) {
            if let limit = limit, limit > 0 {
                self = .some(limit)
            } else {
                self = .all
            }
        }

        var asLimit: Int? {
            switch self {
                case .some(let limit): return limit
                case .all: return -1 // signifies All
            }
        }
    }

    // we want to save the most recent dataset size
    @ObservationIgnored @AppStorage("defaultRowLimit") private var _storedRowLimit: Int = 1000
    @ObservationIgnored var storedRowLimit: Int {
        get {
            access(keyPath: \.storedRowLimit)
            return _storedRowLimit
        }
        set {
            withMutation(keyPath: \.storedRowLimit) {
                _storedRowLimit = newValue
            }
        }
    }

    var isAllRowsChecked: Bool = false

    @ObservationIgnored private var dataLimitRows: DataLimitRows {
        isAllRowsChecked ? .all : .some(storedRowLimit)
    }

    var isLoadingData = false
    var isTraining = false
    var isSaving = false
    var isLoading = false
    var batchSize: Int = 256
    var epochs: Int = 10
    var learningRate: Float = 0.1
    var selectedModel: ModelType = .leNet  // Default model selection
    var vizModel = ModelVizualization()

    private(set) var trainingProgress: [TrainingProgressItem] = []
    var trainer = Trainer()

    func loadTrainData(fromURL url: URL) async {
        defer { isLoadingData = false }

        isLoadingData = true
        guard url.startAccessingSecurityScopedResource() else {
            print("Reading from URL not allowed")
            return
        }

        // Load training data from CSV file
        do {
            let dataFrame: DataFrame
            switch dataLimitRows {
                case .all:
                    dataFrame = try DataFrame(contentsOfCSVFile: url, options: CSVReadingOptions(hasHeaderRow: true))
                case .some(let limit):
                    dataFrame = try DataFrame(contentsOfCSVFile: url, rows: 0 ..< limit, options: CSVReadingOptions(hasHeaderRow: true))
            }
            print("dataframe shape: \(dataFrame.shape)")

            // pass the data to Trainer for converting it to MLXArray
            await trainer.loadTrainingData(dataFrame)

        } catch {
            print("Error loading data: \(error.localizedDescription)")
            return
        }
        url.stopAccessingSecurityScopedResource()
    }

    func train() async {
        isTraining = true

        // continue training a model if one is already loaded
        if await trainer.model == nil {
            let model: TrainableModel = selectedModel == .leNet ? LeNet() : MLP()
            await trainer.setModel(model)
            trainingProgress = []
        }
        vizModel.setModel(await trainer.model)
        await trainer.setOptimizer(SGD(learningRate: learningRate))
        await trainer.train(viewModel: self, epochs: epochs, batchSize: batchSize)
        isTraining = false
    }

    func resetModel() async {
        let model: TrainableModel = selectedModel == .leNet ? LeNet() : MLP()
        await trainer.setModel(model)
        vizModel.setModel(await trainer.model)
        trainingProgress = []
    }

    func save(to url: URL) async {
        do {
            print("begin saving the model")
            try await trainer.saveModel(to: url)
            print("model saved successfully")
        } catch {
            print("Error saving the model: \(error)")
        }
    }

    func load(from url: URL) async {
        do {
            let model: TrainableModel = selectedModel == .leNet ? LeNet() : MLP()
            await trainer.setModel(model)
            try await trainer.loadModel(from: url)
            trainingProgress = []
            vizModel.setModel(await trainer.model)
        } catch {
            print("Error loading the model: \(error)")
        }
    }

    func addTrainingProgress(_ progress: TrainingProgressItem) {
        let remappedProgress = TrainingProgressItem(epoch: trainingProgress.count + 1,
                                                    trainingLoss: progress.trainingLoss,
                                                    testLoss: progress.testLoss,
                                                    accuracy: progress.accuracy)
        trainingProgress.append(remappedProgress)
    }

    static var preview: ViewModel {
        let vm = ViewModel()
        vm.trainingProgress = [
            TrainingProgressItem(epoch: 1, trainingLoss: 0.9, testLoss: 0.85, accuracy: 0.1),
            TrainingProgressItem(epoch: 2, trainingLoss: 0.8, testLoss: 0.7, accuracy: 0.2),
            TrainingProgressItem(epoch: 3, trainingLoss: 0.7, testLoss: 0.6, accuracy: 0.3),
            TrainingProgressItem(epoch: 4, trainingLoss: 0.5, testLoss: 0.55, accuracy: 0.4),
            TrainingProgressItem(epoch: 5, trainingLoss: 0.3, testLoss: 0.4, accuracy: 0.45),
            TrainingProgressItem(epoch: 6, trainingLoss: 0.2, testLoss: 0.3, accuracy: 0.6),
            TrainingProgressItem(epoch: 7, trainingLoss: 0.1, testLoss: 0.25, accuracy: 0.85),
        ]
        return vm
    }
}



struct TrainingProgressItem {
    let epoch: Int
    let trainingLoss: Float
    let testLoss: Float
    let accuracy: Float
}

// we're not modifying any data in the dataframe outside MainActor
extension DataFrame: @retroactive @unchecked Sendable {}
extension AnyColumn: @retroactive @unchecked Sendable {}


extension UTType {
    static var safetensors: UTType { .init("com.iliasazonov.safetensors")! }
}

