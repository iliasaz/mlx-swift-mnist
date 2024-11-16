//
//  TrainableModel.swift
//  mlx-swift-mnist
//
//  Created by Ilia Sazonov on 11/13/24.
//

import Foundation
import MLX
import MLXNN

protocol TrainableModel: Module, UnaryLayer {
    func callAsFunction(_ x: MLXArray) -> MLXArray
    func loss(model: Module, x: MLXArray, y: MLXArray) -> MLXArray
    func eval(x: MLXArray, y: MLXArray) -> MLXArray
    func iterateBatches(batchSize: Int, x: MLXArray, y: MLXArray, using generator: inout any RandomNumberGenerator) -> any Sequence<(MLXArray, MLXArray)>
}

extension TrainableModel {
    func iterateBatches(batchSize: Int, x: MLXArray, y: MLXArray, using generator: inout any RandomNumberGenerator) -> any Sequence<(MLXArray, MLXArray)> {
        BatchSequence(batchSize: batchSize, x: x, y: y, using: &generator)
    }
}


// based on https://github.com/ml-explore/mlx-swift-examples/blob/main/Libraries/MNIST/MNIST.swift

struct BatchSequence: Sequence, IteratorProtocol {
    let batchSize: Int
    let x: MLXArray
    let y: MLXArray

    let indexes: MLXArray
    var index = 0

    init(batchSize: Int, x: MLXArray, y: MLXArray, using generator: inout any RandomNumberGenerator)
    {
        self.batchSize = batchSize
        self.x = x
        self.y = y
        self.indexes = MLXArray(Array(0 ..< y.size).shuffled(using: &generator))
    }

    mutating func next() -> (MLXArray, MLXArray)? {
        guard index < y.size else { return nil }

        let range = index ..< Swift.min(index + batchSize, y.size)
        index += batchSize
        let ids = indexes[range]
        return (x[ids], y[ids])
    }
}



