//
//  MLP.swift
//  mlx-swift-mnist
//
//  Created by Ilia Sazonov on 11/14/24.
//

import Foundation
import MLX
import MLXNN

// based on https://github.com/saanhir/neural-lab/tree/main/MLP_from_scratch

class MLP: Module, TrainableModel {
    @ModuleInfo var fc1: Linear
    @ModuleInfo var fc2: Linear
    @ModuleInfo var fc3: Linear
    @ModuleInfo var softMax: Softmax

    override init() {
        fc1 = Linear(28 * 28, 100)
        fc2 = Linear(100, 50)
        fc3 = Linear(50, 10)
        softMax = Softmax()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let batchSize = x.shape[0]
        var x = x.reshaped([batchSize, 28*28])
        x = tanh(fc1(x))
        x = tanh(fc2(x))
        x = tanh(fc3(x))
        x = softMax(x)
        return x
    }

    func loss(model: Module, x: MLXArray, y: MLXArray) -> MLXArray {
        return crossEntropy(logits: self(x), targets: y, reduction: .mean)
    }

    func eval(x: MLXArray, y: MLXArray) -> MLXArray {
        mean(argMax(self(x), axis: 1) .== y)
    }
}




