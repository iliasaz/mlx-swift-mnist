//
//  LeNet.swift
//  mlx-swift-mnist
//
//  Created by Ilia Sazonov on 11/13/24.
//

import Foundation
import MLX
import MLXNN

// based on https://github.com/ml-explore/mlx-swift-examples/blob/main/Libraries/MNIST/MNIST.swift

class LeNet: Module, TrainableModel {
    @ModuleInfo var conv1: Conv2d
    @ModuleInfo var conv2: Conv2d
    @ModuleInfo var pool1: MaxPool2d
    @ModuleInfo var pool2: MaxPool2d
    @ModuleInfo var fc1: Linear
    @ModuleInfo var fc2: Linear
    @ModuleInfo var fc3: Linear
    @ModuleInfo var sm: Softmax

    override init() {
        conv1 = Conv2d(inputChannels: 1, outputChannels: 6, kernelSize: 5, padding: 2)
        conv2 = Conv2d(inputChannels: 6, outputChannels: 16, kernelSize: 5, padding: 0)
        pool1 = MaxPool2d(kernelSize: 2, stride: 2)
        pool2 = MaxPool2d(kernelSize: 2, stride: 2)
        fc1 = Linear(16 * 5 * 5, 120)
        fc2 = Linear(120, 84)
        fc3 = Linear(84, 10)
        sm = Softmax(axis: 1)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = x
        x = pool1(tanh(conv1(x)))
        x = pool2(tanh(conv2(x)))
        x = flattened(x, start: 1)
        x = tanh(fc1(x))
        x = tanh(fc2(x))
        x = fc3(x)
        x = sm(x)
        return x
    }

    func loss(model: Module, x: MLXArray, y: MLXArray) -> MLXArray {
        crossEntropy(logits: self(x), targets: y, reduction: .mean)
    }

    func eval(x: MLXArray, y: MLXArray) -> MLXArray {
        mean(argMax(self(x), axis: 1) .== y)
    }
}


