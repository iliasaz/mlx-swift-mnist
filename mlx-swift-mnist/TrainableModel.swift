//
//  TrainableModel.swift
//  mlx-swift-mnist
//
//  Created by Ilia Sazonov on 11/13/24.
//

import Foundation
import MLX
import MLXNN

public protocol TrainableModel: Module {
    func callAsFunction(_ x: MLXArray) -> MLXArray

    /// This is a variant of the forward pass, in which we save the activations. This is used for layer visualization.
    func callAsFunction(_ x: MLXArray, saveActivations: (([String : MLXArray]) -> Void)?) -> MLXArray
    
    func loss(model: Module, x: MLXArray, y: MLXArray) -> MLXArray
    func eval(x: MLXArray, y: MLXArray) -> MLXArray
    func iterateBatches(batchSize: Int, x: MLXArray, y: MLXArray, using generator: inout any RandomNumberGenerator) -> any Sequence<(MLXArray, MLXArray)>
}

extension TrainableModel {
    func iterateBatches(batchSize: Int, x: MLXArray, y: MLXArray, using generator: inout any RandomNumberGenerator) -> any Sequence<(MLXArray, MLXArray)> {
        BatchSequence(batchSize: batchSize, x: x, y: y, using: &generator)
    }

    func callAsFunction(_ x: MLXArray, saveActivations: (([String : MLXArray]) -> Void)? = nil) -> MLXArray {
        return self(x)

    }
}


