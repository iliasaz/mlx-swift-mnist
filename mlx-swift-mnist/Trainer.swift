//
//  Trainer.swift
//  mlx-swift-mnist
//
//  Created by Ilia Sazonov on 11/13/24.
//

import Foundation
import TabularData
import MLX
import MLXOptimizers
import MLXRandom
import MLXNN

actor Trainer {
    let mnistImageSize: CGSize = CGSize(width: 28, height: 28)

    var model: (any TrainableModel)?
    var optimizer: Optimizer?

    var trainLabels = MLXArray()
    var trainImages = MLXArray()
    var testLabels = MLXArray()
    var testImages = MLXArray()

    private func splitData(_ data: DataFrame) -> (labels: (train: AnyColumnSlice, test: AnyColumnSlice), images: (train: DataFrame.Slice, test: DataFrame.Slice)) {
        let (trainSlice, testSlice) = data.randomSplit(by: 0.8)
        let imageColumnNames = data.columns.map(\.name).dropFirst()
        print("trainSlice shape: \(trainSlice.shape), testSlice shape: \(testSlice.shape)")


        // extract label and image columns
        let trainLabels = trainSlice["label"]
        let trainImages = trainSlice.selecting(columnNames: imageColumnNames)
        let testLabels = testSlice["label"]
        let testImages = testSlice.selecting(columnNames: imageColumnNames)
        return (labels: (trainLabels, testLabels), images: (trainImages, testImages))
    }

    func loadTrainingData(_ dataFrame: DataFrame) async {
        let data = splitData(dataFrame)

        // train data
        trainLabels.update(
            MLXArray(data.labels.train.map { UInt32($0 as! Int) }).reshaped([-1])
        )
        print("trainLabels shape: \(trainLabels.shape), type: \(trainLabels.dtype)")

        trainImages.update(MLXArray(data.images.train.rows.flatMap {
            $0[$0.indices].map { Float($0 as! Int)/255.0 }
        }).reshaped([-1, 28, 28, 1]))
        print("trainImages shape: \(trainImages.shape), type: \(trainImages.dtype)")

        // test data
        testLabels.update(
            MLXArray(data.labels.test.map { UInt32($0 as! Int) }).reshaped([-1])
        )
        print("testLabels shape: \(testLabels.shape), type: \(testLabels.dtype)")

        testImages.update(MLXArray(data.images.test.rows.flatMap {
            $0[$0.indices].map { Float($0 as! Int)/255.0 }
        }).reshaped([-1, 28, 28, 1]))
        print("testImages shape: \(testImages.shape), type: \(testImages.dtype)")

    }

    func setModel(_ model: any TrainableModel) {
        self.model = model
    }

    func setOptimizer(_ optimizer: Optimizer) {
        self.optimizer = optimizer
    }

    func train(viewModel: ViewModel, epochs: Int = 10, batchSize: Int = 256) async {
        guard let model, let optimizer else { return }
        eval(model.parameters())

        // the training loop
        let lg = valueAndGrad(model: model, model.loss)

        // using a consistent random seed so it behaves the same way each time
        MLXRandom.seed(0)
        var generator: RandomNumberGenerator = SplitMix64(seed: 0)

        for epoch in 0 ..< epochs {
            let start = Date.timeIntervalSinceReferenceDate
            var trainingLosses = [Float]()

            for (x, y) in model.iterateBatches(batchSize: batchSize, x: trainImages, y: trainLabels, using: &generator) {
                // loss and gradients
                let (loss, grads) = lg(model, x, y)

                // use SGD to update the weights
                optimizer.update(model: model, gradients: grads)

                // eval the parameters so the next iteration is independent
                eval(model, optimizer)
                trainingLosses.append(loss.item(Float.self))
            }
            let end = Date.timeIntervalSinceReferenceDate

            // mean training loss over the current epoch
            let epochTrainingLoss = trainingLosses.reduce(0, +) / Float(trainingLosses.count)

            // test accuracy and loss
            let testAccuracy: Float = model.eval(x: testImages, y: testLabels).item(Float.self)
            let testLoss: Float = model.loss(model: model, x: testImages, y: testLabels).item(Float.self)
            print("epoch: \(epoch), time: \(end - start), training loss: \(epochTrainingLoss), test accuracy: \(testAccuracy), test loss: \(testLoss)")

            await MainActor.run {
                viewModel.addTrainingProgress(TrainingProgressItem(epoch: epoch, trainingLoss: epochTrainingLoss, testLoss: testLoss, accuracy: testAccuracy))
            }
        }
    }

    func saveModel(to url: URL) async throws {
        guard let model else { return }
        try MLX.save(arrays: Dictionary(uniqueKeysWithValues: model.trainableParameters().flattened()), url: url)
    }

    func loadModel(from url: URL) async throws {
        guard let model else { return }
        let weights = try ModuleParameters.unflattened(loadArrays(url: url))
        try model.update(parameters: weights, verify: .all)
        eval(model)
    }

    func predictDistribution(pixelData: MLXArray) -> (probabilities: [Float], highestIndex: Int)? {
        guard let model else { return nil }
        let x = pixelData.reshaped([1, 28, 28, 1]).asType(.float32) / 255.0
        let y = model(x).flattened()
        let highestIndex = argMax(y).item(Int.self)
        return (probabilities: y.map {$0.item(Float.self)}, highestIndex: highestIndex)
    }
}
