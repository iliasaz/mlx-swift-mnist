//
//  DataView.swift
//  mlx-swift-mnist
//
//  Created by Ilia Sazonov on 11/24/24.
//

import SwiftUI

struct DataView: View {
    @Bindable var viewModel: ViewModel
    @State private var isDataFileLoaPresented: Bool = false

    private let datasetURL = URL(string: "https://www.kaggle.com/competitions/digit-recognizer")!

    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            Text("Loading Dataset")
                .font(.title.bold())
                .padding(.vertical)
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
            }
            Spacer()
        }
        .padding()
    }
}

#Preview {
    @Previewable @State var viewModel = ViewModel()
    DataView(viewModel: viewModel)
}
