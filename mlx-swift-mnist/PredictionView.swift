//
//  PredictionView.swift
//  mlx-swift-mnist
//
//  Created by Ilia Sazonov on 11/18/24
//  based on https://github.com/ml-explore/mlx-swift-examples/blob/main/Applications/MNISTTrainer/PredictionView.swift

import MLX
import MLXNN
import SwiftUI

struct Canvas: View {
    @Binding var path: Path
    @State var lastPoint: CGPoint?

    var body: some View {
        path
            .stroke(.white, lineWidth: 10)
            .background(.black)
            .gesture(
                DragGesture(minimumDistance: 0.05)
                    .onChanged { touch in
                        add(point: touch.location)
                    }
                    .onEnded { touch in
                        lastPoint = nil
                    }
            )
    }

    func add(point: CGPoint) {
        var newPath = path
        if let lastPoint {
            newPath.move(to: lastPoint)
            newPath.addLine(to: point)
        } else {
            newPath.move(to: point)
        }
        self.path = newPath
        lastPoint = point
    }
}

extension Path {
    mutating func center(to newMidPoint: CGPoint) {
        let middleX = boundingRect.midX
        let middleY = boundingRect.midY
        self = offsetBy(dx: newMidPoint.x - middleX, dy: newMidPoint.y - middleY)
    }
}

struct PredictionView: View {
    @Bindable var viewModel = ViewModel()
    @State var path: Path = Path()
    @State var predictions = [Prediction]()
    let canvasSize = 400.0

    var body: some View {
        VStack(alignment: .leading) {
            Text("Recognizing a Hand-written Digit")
                .font(.title.bold())
                .padding(.vertical)

            HStack {
                VStack(alignment: .leading) {
                    Text("Draw a digit")
                    Canvas(path: $path)
                        .frame(width: canvasSize, height: canvasSize)
                }
                VStack(alignment: .leading) {
                    Text("Prediction Results")
                    PredictionResults(preditedDistribution: $predictions)
                        .frame(width: canvasSize, height: canvasSize)
                        .cornerRadius(5)
                }
            }
            HStack {
                Button("Predict") {
                    path.center(to: CGPoint(x: canvasSize / 2, y: canvasSize / 2))
                    predict()
                }
                Button("Clear") {
                    path = Path()
                    predictions = []
                }
            }
            Spacer()
        }
        .padding()
    }

    @MainActor
    func predict() {
        let mnistImageSize: CGSize = CGSize(width: 28, height: 28)
        let imageRenderer = ImageRenderer(
            content: Canvas(path: $path).frame(width: canvasSize, height: canvasSize))
        if let image = imageRenderer.cgImage?.grayscaleImage(with: mnistImageSize) {
            Task {
                guard let results = await viewModel.trainer.predictDistribution(pixelData: image.pixelData()) else { return }
                predictions = results.probabilities.enumerated().map {
                    Prediction(value: $0.offset, probability: $0.element, isBest: $0.offset == results.highestIndex)
                }
            }
        }
    }
}

extension CGImage {
    func grayscaleImage(with newSize: CGSize) -> CGImage? {
        let colorSpace = CGColorSpaceCreateDeviceGray()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue)

        guard
            let context = CGContext(
                data: nil,
                width: Int(newSize.width),
                height: Int(newSize.height),
                bitsPerComponent: 8,
                bytesPerRow: Int(newSize.width),
                space: colorSpace,
                bitmapInfo: bitmapInfo.rawValue)
        else {
            return nil
        }
        context.draw(self, in: CGRect(x: 0, y: 0, width: newSize.width, height: newSize.width))
        return context.makeImage()
    }

    func pixelData() -> MLXArray {
        guard let data = self.dataProvider?.data else {
            return []
        }
        let bytePtr = CFDataGetBytePtr(data)
        let count = CFDataGetLength(data)
        return MLXArray(UnsafeBufferPointer(start: bytePtr, count: count))
    }
}


#Preview {
    PredictionView()
        .frame(width: 1000, height: 800)
}
