//
//  BatchSequence.swift
//  mlx-swift-mnist
//
//  Created by Ilia Sazonov on 11/26/24.
//  based on https://github.com/ml-explore/mlx-swift-examples/blob/main/Libraries/MNIST/MNIST.swift
//

import Foundation
import MLX


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
