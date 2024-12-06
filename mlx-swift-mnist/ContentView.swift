//
//  ContentView.swift
//  mlx-swift-mnist
//
//  Created by Ilia Sazonov on 11/11/24.
//

import SwiftUI

struct ContentView: View {
    @Bindable var viewModel = ViewModel()

    var body: some View {
        NavigationSplitView {
            List {
                NavigationLink {
                    DataView(viewModel: viewModel)
                        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
                        .padding()
                } label: {
                    Label("Data", systemImage: "camera.macro.circle")
                        .font(.title2)
                        .labelStyle(.titleAndIcon)
                }

                NavigationLink {
                    TrainingView(viewModel: viewModel)
                        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
                        .padding()
                } label: {
                    Label("Training", systemImage: "figure.archery.circle")
                        .font(.title2)
                        .labelStyle(.titleAndIcon)
                }

                NavigationLink {
                    PredictionView(viewModel: viewModel)
                        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
                        .padding()
                } label: {
                    Label("Testing", systemImage: "checkmark.circle")
                        .font(.title2)
                        .labelStyle(.titleAndIcon)
                }
            }
            .padding(.vertical)
            .navigationTitle("MLX Swift MNIST Training Example")
            .navigationSplitViewColumnWidth(180)
        } detail: {
            Text("Select a section to view details")
        }
    }
}



#Preview {
    ContentView()
        .frame(width: 1200, height: 800)
}
