# MNIST training example using MLX, TabularData, Charts, and SwiftUI.

This is an example application demonstrating the use of [Apple MLX (Swift)](https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx), [TabularData](https://developer.apple.com/documentation/tabulardata), [Swift Charts](https://developer.apple.com/documentation/charts), and [SwiftUI](https://developer.apple.com/documentation/SwiftUI) frameworks for creating a modular ML training app. The app trains a neural network of choice ( MLP or LeNet) on [MNIST](https://en.wikipedia.org/wiki/MNIST_database) data set - hand written digits recognition.  

The app has teh following featues:
- Loading the training data from a CSV file,  
- Training a model (MLP or LeNet),  
- Running predictions,  
- Visualizing the activations of the model layers.    

The code is structured so that one can easily adjust the model architecture or add your own model. Layer visualizations are implemented as extensions to the layer classes and can be easily modified to suite your visualization preferences.  

## Similar works
- The implementation of MLP (multi-layer perceptron) neural network is models after [saanhir/neural-lab repo](https://github.com/saanhir/neural-lab/tree/main/MLP_from_scratch).
- The implementation of LeNet neural network is based on  [mlx-swift-examples repo](https://github.com/ml-explore/mlx-swift-examples/blob/main/Libraries/MNIST/MNIST.swift).

The application runs best on MacOS 15+ and Apple Silicon CPU/GPU.

The moduler design allows us to modify the models independently from data processing and training pipelines. One can create their own neural network model and easily plug it into the app.  

## New in v.1.2
- Saving and loading a trained model using safetensors format  
- Digit recognition from a hand drawn image  
- Continuous model training with reporting of the current losses and accuracy  
- Layer visualizations  

![Screenshots](https://github.com/iliasaz/mlx-swift-mnist/blob/main/training-results2.png)
![Screenshots](https://github.com/iliasaz/mlx-swift-mnist/blob/main/layer-viz1.png)
![Screenshots](https://github.com/iliasaz/mlx-swift-mnist/blob/main/layer-viz2.png)


