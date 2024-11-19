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
import UniformTypeIdentifiers


struct ContentView: View {
    @Bindable var viewModel = ViewModel()
    @State private var isDataFileLoaPresented: Bool = false

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
                                isDataFileLoaPresented.toggle()
                            }
                            .fileImporter(isPresented: $isDataFileLoaPresented, allowedContentTypes: [.commaSeparatedText]) { result in
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
                            Button("Train") {
                                Task {
                                    await viewModel.train()
                                }
                            }
                            .disabled(viewModel.isLoadingData || viewModel.isLoading || viewModel.isSaving || viewModel.isTraining)

                            Button("Reset Model") {
                                Task {
                                    await viewModel.resetModel()
                                }
                            }
                            .disabled(viewModel.isLoading || viewModel.isSaving || viewModel.isTraining)

                            ProgressView()
                                .opacity((viewModel.isTraining || viewModel.isLoading || viewModel.isSaving) ? 1 : 0)

                            Button("Save") {
                                showSavePanel()
                            }
                            .disabled(viewModel.isLoading || viewModel.isSaving || viewModel.isTraining)

                            Button("Load") {
                                showOpenPanel()
                            }
                            .disabled(viewModel.isLoading || viewModel.isSaving || viewModel.isTraining)

                            NavigationLink("Test", destination: PredictionView(viewModel: viewModel))
                                .disabled(viewModel.isLoading || viewModel.isTraining)
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

    func showSavePanel() {
        let savePanel = NSSavePanel()
        savePanel.title = "Save Model"
        savePanel.allowedContentTypes = [.safetensors] // Specify your file type
        savePanel.nameFieldStringValue = "model" // Default file name

        savePanel.begin { response in
            if response == .OK, let url = savePanel.url {
                Task {
                    await viewModel.save(to: url)
                }
            }
        }
    }

    func showOpenPanel() {
        let openPanel = NSOpenPanel()
        openPanel.title = "Select a Model File"
        openPanel.allowedContentTypes = [.safetensors] // Specify your file type
        openPanel.allowsMultipleSelection = false // Allow one file

        openPanel.begin { response in
            if response == .OK, let url = openPanel.url {
                Task {
                    await viewModel.load(from: url)
                }
            }
        }
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
        await trainer.setOptimizer(SGD(learningRate: learningRate))
        await trainer.train(viewModel: self, epochs: epochs, batchSize: batchSize)
        isTraining = false
    }

    func resetModel() async {
        let model: TrainableModel = selectedModel == .leNet ? LeNet() : MLP()
        await trainer.setModel(model)
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

extension UTType {
    static var safetensors: UTType { .init("com.iliasazonov.safetensors")! }
}





#Preview {
    ContentView()
}
