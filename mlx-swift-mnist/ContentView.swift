//
//  ContentView.swift
//  mlx-swift-mnist
//
//  Created by Ilia Sazonov on 11/11/24.
//

import SwiftUI
import TabularData
import UniformTypeIdentifiers
import MLX
import MLXOptimizers


struct ContentView: View {
    @Bindable var viewModel = ViewModel()
    @State private var fileLoadIsPresented: Bool = false

    private let datasetURL = URL(string: "https://www.kaggle.com/competitions/digit-recognizer")!

    var body: some View {
        NavigationStack {
            VStack(spacing: 5) {
                Form {
                    Section("Data") {
                        Link("Download CSV Dataset", destination: datasetURL)
                            .font(.body)
                            .foregroundColor(.blue)
                            .underline()
                            .padding(.bottom, 5)
                            .focusable(false)

                        HStack {
                            TextField("Row Limit", value: $viewModel.storedRowLimit, format: .number)
                                .textFieldStyle(.roundedBorder)
                                .frame(width: 200)
                                .disabled(viewModel.isAllRowsChecked)

                            Toggle("All Rows", isOn: $viewModel.isAllRowsChecked)
                        }

                        HStack {
                            Button("Load Data") {
                                fileLoadIsPresented.toggle()
                            }
                            .fileImporter(isPresented: $fileLoadIsPresented, allowedContentTypes: [.commaSeparatedText]) { result in
                                switch result {
                                    case .success(let url):
                                        Task {
                                            await viewModel.loadTrainData(fromURL: url)
                                        }
                                    case .failure(let error):
                                        print("Error loading file: \(error.localizedDescription)")
                                }
                            }
                            .disabled(viewModel.isLoadingData || viewModel.isTraining)

                            ProgressView()
                                .opacity(viewModel.isLoadingData ? 1 : 0)
                        }
                    }
                    .padding(.bottom, 5)

                    Section("Training parameters") {
                        // Model selection picker
                        Picker("Model", selection: $viewModel.selectedModel) {
                            ForEach(ViewModel.ModelType.allCases) { model in
                                Text(model.rawValue).tag(model)
                            }
                        }
                        .pickerStyle(.menu)

                        TextField("Batch Size", value: $viewModel.batchSize, format: .number)
                            .textFieldStyle(.roundedBorder)

                        TextField("Epochs", value: $viewModel.epochs, format: .number)
                            .textFieldStyle(.roundedBorder)

                        TextField("Learning Rate", value: $viewModel.learningRate, format: .number)
                            .textFieldStyle(.roundedBorder)

                        HStack {
                            Button("Train Model") {
                                Task {
                                    await viewModel.train()
                                }
                            }
                            .disabled(viewModel.isLoadingData || viewModel.isTraining)

                            ProgressView()
                                .opacity(viewModel.isTraining ? 1 : 0)
                        }
                    }
                }
                .padding(.vertical, 5)

                TrainingChartView(viewModel: viewModel)
            }
            .padding()
        }
        .navigationTitle("MLX Swift MNIST Training Example")
    }
}

// we're not modifying any data in the dataframe outside MainActor
extension DataFrame: @retroactive @unchecked Sendable {}
extension AnyColumn: @retroactive @unchecked Sendable {}

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
    var batchSize: Int = 256
    var epochs: Int = 10
    var learningRate: Float = 0.1
    var selectedModel: ModelType = .leNet  // Default model selection

    var trainingProgress: [(epoch: Int, trainingLoss: Float, testLoss: Float, accuracy: Float)] = []
    private var trainer = Trainer()

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
        trainingProgress = []

        let model: TrainableModel = selectedModel == .leNet ? LeNet() : MLP()
        await trainer.setModel(model)
        await trainer.setOptimizer(SGD(learningRate: learningRate))
        await trainer.train(viewModel: self, epochs: epochs, batchSize: batchSize)
        isTraining = false
    }

    static var preview: ViewModel {
        let vm = ViewModel()
        vm.trainingProgress = [
            (epoch: 1, trainingLoss: 0.9, testLoss: 0.8, accuracy: 0.1),
            (epoch: 2, trainingLoss: 0.7, testLoss: 0.6, accuracy: 0.2),
            (epoch: 3, trainingLoss: 0.5, testLoss: 0.8, accuracy: 0.4),
            (epoch: 4, trainingLoss: 0.3, testLoss: 0.5, accuracy: 0.6),
            (epoch: 5, trainingLoss: 0.2, testLoss: 0.4, accuracy: 0.75),
            (epoch: 6, trainingLoss: 0.1, testLoss: 0.4, accuracy: 0.85)
        ]
        return vm
    }
}





#Preview {
    ContentView()
}
