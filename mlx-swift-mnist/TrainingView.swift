//
//  TrainingView.swift
//  mlx-swift-mnist
//
//  Created by Ilia Sazonov on 11/24/24.
//

import SwiftUI


struct TrainingView: View {
    @Bindable var viewModel: ViewModel
    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            Text("Training a Model")
                .font(.title.bold())
                .padding(.vertical)
            Form {
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

            Spacer()
        }
        .padding()
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


#Preview {
    @Previewable @State var viewModel = ViewModel()
    TrainingView(viewModel: viewModel)
        .frame(width: 1000, height: 800)
}
