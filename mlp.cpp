/**
 * mlp.cpp - Implementation of the Multilayer Perceptron (MLP) Neural Network
 * 
 * This class implements a fully-connected feed-forward neural network with
 * OpenMP parallelization for matrix operations.
 */

 #include "mlp.h"
 #include <iostream>
 #include <fstream>
 #include <cmath>
 #include <algorithm>
 #include <omp.h>
 #include <cassert>
 #include <random>
 
 using namespace std;
 
 MLP::MLP(int inputSize, const vector<int>& hiddenSizes, int outputSize) 
     : inputSize(inputSize), hiddenSizes(hiddenSizes), outputSize(outputSize) {
     
     // Initialize random number generator
     random_device rd;
     rng = mt19937(rd());
     uniform_real_distribution<float> dist(-0.5, 0.5);
     
     // Setup network architecture
     vector<int> layerSizes;
     layerSizes.push_back(inputSize);
     layerSizes.insert(layerSizes.end(), hiddenSizes.begin(), hiddenSizes.end());
     layerSizes.push_back(outputSize);
     
     int numLayers = layerSizes.size();
     
     // Initialize weights and biases
     weights.resize(numLayers-1);
     biases.resize(numLayers-1);
     activations.resize(numLayers);
     deltas.resize(numLayers);
     
     // For each layer (except input)
     for (int i = 0; i < numLayers-1; ++i) {
         int currentLayerSize = layerSizes[i];
         int nextLayerSize = layerSizes[i+1];
         
         // Initialize weights with small random values
         weights[i].resize(nextLayerSize);
         for (int j = 0; j < nextLayerSize; ++j) {
             weights[i][j].resize(currentLayerSize);
             for (int k = 0; k < currentLayerSize; ++k) {
                 weights[i][j][k] = dist(rng) / sqrt(currentLayerSize);
             }
         }
         
         // Initialize biases with zeros
         biases[i].resize(nextLayerSize, 0.0f);
         
         // Initialize activations
         activations[i].resize(currentLayerSize, 0.0f);
         
         // Initialize deltas
         deltas[i].resize(currentLayerSize, 0.0f);
     }
     
     // Initialize final layer activations and deltas
     activations[numLayers-1].resize(layerSizes[numLayers-1], 0.0f);
     deltas[numLayers-1].resize(layerSizes[numLayers-1], 0.0f);
     
     cout << "Initialized MLP with structure: ";
     for (size_t i = 0; i < layerSizes.size(); ++i) {
         cout << layerSizes[i];
         if (i < layerSizes.size() - 1) cout << "->";
     }
     cout << endl;
 }
 
 float MLP::sigmoid(float x) {
     return 1.0f / (1.0f + exp(-x));
 }
 
 float MLP::sigmoidDerivative(float x) {
     float s = sigmoid(x);
     return s * (1.0f - s);
 }
 
 void MLP::forwardPass(const vector<float>& input) {
     // Set input layer activations
     for (int i = 0; i < inputSize; ++i) {
         activations[0][i] = input[i];
     }
     
     // For each layer (except input)
     for (size_t layer = 0; layer < weights.size(); ++layer) {
         int numNeurons = weights[layer].size();
         
         // Parallel computation of activations for each neuron in the current layer
         #pragma omp parallel for
         for (int j = 0; j < numNeurons; ++j) {
             float sum = biases[layer][j];
             
             // Sum of (weight * prev_activation) for each input to this neuron
             for (size_t k = 0; k < weights[layer][j].size(); ++k) {
                 sum += weights[layer][j][k] * activations[layer][k];
             }
             
             // Apply activation function
             activations[layer+1][j] = sigmoid(sum);
         }
     }
 }
 
 vector<float> MLP::oneHotEncode(int label, int numClasses) {
     vector<float> encoded(numClasses, 0.0f);
     if (label >= 0 && label < numClasses) {
         encoded[label] = 1.0f;
     }
     return encoded;
 }
 
 void MLP::backwardPass(const vector<float>& input, const vector<float>& target) {
     // Compute output layer deltas
     int outputLayer = weights.size();
     
     #pragma omp parallel for
     for (int i = 0; i < outputSize; ++i) {
         float error = activations[outputLayer][i] - target[i];
         deltas[outputLayer][i] = error * activations[outputLayer][i] * (1.0f - activations[outputLayer][i]);
     }
     
     // Compute hidden layer deltas
     for (int layer = outputLayer - 1; layer > 0; --layer) {
         int numNeurons = activations[layer].size();
         int nextLayerSize = weights[layer].size();
         
         #pragma omp parallel for
         for (int i = 0; i < numNeurons; ++i) {
             float errorSum = 0.0f;
             for (int j = 0; j < nextLayerSize; ++j) {
                 errorSum += weights[layer][j][i] * deltas[layer+1][j];
             }
             deltas[layer][i] = errorSum * activations[layer][i] * (1.0f - activations[layer][i]);
         }
     }
 }
 
 void MLP::updateWeights(const vector<float>& input, float learningRate) {
     // Update weights and biases for each layer
     for (size_t layer = 0; layer < weights.size(); ++layer) {
         int numNeurons = weights[layer].size();
         int prevLayerSize = activations[layer].size();
         
         #pragma omp parallel for collapse(2)
         for (int j = 0; j < numNeurons; ++j) {
             for (int i = 0; i < prevLayerSize; ++i) {
                 weights[layer][j][i] -= learningRate * deltas[layer+1][j] * activations[layer][i];
             }
             biases[layer][j] -= learningRate * deltas[layer+1][j];
         }
     }
 }
 
 void MLP::train(const vector<float>& X, const vector<int>& y, int numSamples, int numFeatures, 
                int epochs, float learningRate) {
     cout << "Starting MLP training with " << numSamples << " samples..." << endl;
     
     vector<int> indices(numSamples);
     iota(indices.begin(), indices.end(), 0);
     
     for (int epoch = 0; epoch < epochs; ++epoch) {
         // Shuffle indices for stochastic gradient descent
         shuffle(indices.begin(), indices.end(), rng);
         
         float epochLoss = 0.0f;
         
         for (int idx : indices) {
             // Extract the current sample and its label
             vector<float> input(numFeatures);
             for (int j = 0; j < numFeatures; ++j) {
                 input[j] = X[idx * numFeatures + j];
             }
             
             // Convert label to one-hot encoding
             vector<float> target = oneHotEncode(y[idx], outputSize);
             
             // Forward pass
             forwardPass(input);
             
             // Compute loss (cross-entropy for classification)
             int outputLayer = weights.size();
             float sampleLoss = 0.0f;
             for (int j = 0; j < outputSize; ++j) {
                 if (target[j] > 0) {
                     sampleLoss -= log(max(activations[outputLayer][j], 1e-7f));
                 }
             }
             epochLoss += sampleLoss;
             
             // Backward pass
             backwardPass(input, target);
             
             // Update weights
             updateWeights(input, learningRate);
         }
         
         // Print progress every 10 epochs
         if ((epoch + 1) % 10 == 0 || epoch == 0 || epoch == epochs - 1) {
             cout << "MLP Epoch " << (epoch + 1) << "/" << epochs 
                  << ", Loss: " << (epochLoss / numSamples) << endl;
         }
     }
     
     cout << "MLP training completed." << endl;
 }
 
 vector<int> MLP::predict(const vector<float>& X, int numSamples, int numFeatures) {
     vector<int> predictions(numSamples);
     
     for (int i = 0; i < numSamples; ++i) {
         // Extract current sample
         vector<float> input(numFeatures);
         for (int j = 0; j < numFeatures; ++j) {
             input[j] = X[i * numFeatures + j];
         }
         
         // Forward pass
         forwardPass(input);
         
         // Get the class with highest probability
         int outputLayer = weights.size();
         int predictedClass = 0;
         float maxProb = activations[outputLayer][0];
         
         for (int j = 1; j < outputSize; ++j) {
             if (activations[outputLayer][j] > maxProb) {
                 maxProb = activations[outputLayer][j];
                 predictedClass = j;
             }
         }
         
         predictions[i] = predictedClass;
     }
     
     return predictions;
 }
 
 void MLP::saveModel(const string& filename) {
     ofstream outFile(filename, ios::binary);
     
     if (!outFile.is_open()) {
         cerr << "Error: Could not open file " << filename << " for writing." << endl;
         return;
     }
     
     // Write network architecture
     outFile.write(reinterpret_cast<char*>(&inputSize), sizeof(inputSize));
     
     int hiddenLayersSize = hiddenSizes.size();
     outFile.write(reinterpret_cast<char*>(&hiddenLayersSize), sizeof(hiddenLayersSize));
     
     for (int size : hiddenSizes) {
         outFile.write(reinterpret_cast<char*>(&size), sizeof(size));
     }
     
     outFile.write(reinterpret_cast<char*>(&outputSize), sizeof(outputSize));
     
     // Write weights and biases
     for (size_t layer = 0; layer < weights.size(); ++layer) {
         int numNeurons = weights[layer].size();
         int prevLayerSize = layer == 0 ? inputSize : hiddenSizes[layer-1];
         
         // Write weights
         for (int j = 0; j < numNeurons; ++j) {
             for (int i = 0; i < prevLayerSize; ++i) {
                 outFile.write(reinterpret_cast<char*>(&weights[layer][j][i]), sizeof(float));
             }
         }
         
         // Write biases
         for (int j = 0; j < numNeurons; ++j) {
             outFile.write(reinterpret_cast<char*>(&biases[layer][j]), sizeof(float));
         }
     }
     
     outFile.close();
     cout << "MLP model saved to " << filename << endl;
 }
 
 void MLP::loadModel(const string& filename) {
     ifstream inFile(filename, ios::binary);
     
     if (!inFile.is_open()) {
         cerr << "Error: Could not open file " << filename << " for reading." << endl;
         return;
     }
     
     // Read network architecture
     inFile.read(reinterpret_cast<char*>(&inputSize), sizeof(inputSize));
     
     int hiddenLayersSize;
     inFile.read(reinterpret_cast<char*>(&hiddenLayersSize), sizeof(hiddenLayersSize));
     
     hiddenSizes.resize(hiddenLayersSize);
     for (int i = 0; i < hiddenLayersSize; ++i) {
         inFile.read(reinterpret_cast<char*>(&hiddenSizes[i]), sizeof(int));
     }
     
     inFile.read(reinterpret_cast<char*>(&outputSize), sizeof(outputSize));
     
     // Setup network architecture
     vector<int> layerSizes;
     layerSizes.push_back(inputSize);
     layerSizes.insert(layerSizes.end(), hiddenSizes.begin(), hiddenSizes.end());
     layerSizes.push_back(outputSize);
     
     int numLayers = layerSizes.size();
     
     // Initialize weights and biases
     weights.resize(numLayers-1);
     biases.resize(numLayers-1);
     activations.resize(numLayers);
     deltas.resize(numLayers);
     
     // For each layer (except input)
     for (int i = 0; i < numLayers-1; ++i) {
         int currentLayerSize = layerSizes[i];
         int nextLayerSize = layerSizes[i+1];
         
         weights[i].resize(nextLayerSize);
         for (int j = 0; j < nextLayerSize; ++j) {
             weights[i][j].resize(currentLayerSize);
         }
         
         biases[i].resize(nextLayerSize);
         
         activations[i].resize(currentLayerSize, 0.0f);
         deltas[i].resize(currentLayerSize, 0.0f);
     }
     
     // Initialize final layer activations and deltas
     activations[numLayers-1].resize(layerSizes[numLayers-1], 0.0f);
     deltas[numLayers-1].resize(layerSizes[numLayers-1], 0.0f);
     
     // Read weights and biases
     for (size_t layer = 0; layer < weights.size(); ++layer) {
         int numNeurons = weights[layer].size();
         int prevLayerSize = layer == 0 ? inputSize : hiddenSizes[layer-1];
         
         // Read weights
         for (int j = 0; j < numNeurons; ++j) {
             for (int i = 0; i < prevLayerSize; ++i) {
                 inFile.read(reinterpret_cast<char*>(&weights[layer][j][i]), sizeof(float));
             }
         }
         
         // Read biases
         for (int j = 0; j < numNeurons; ++j) {
             inFile.read(reinterpret_cast<char*>(&biases[layer][j]), sizeof(float));
         }
     }
     
     inFile.close();
     cout << "MLP model loaded from " << filename << endl;
 }