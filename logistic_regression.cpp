/**
 * logistic_regression.cpp - Implementation of the Logistic Regression classifier
 * 
 * This class implements logistic regression with OpenMP parallelization for
 * gradient computation.
 */

 #include "logistic_regression.h"
 #include <iostream>
 #include <fstream>
 #include <cmath>
 #include <omp.h>
 #include <algorithm>
 #include <cassert>
 
 using namespace std;
 
 LogisticRegression::LogisticRegression(int numFeatures, float learningRate, int maxIterations)
     : numFeatures(numFeatures), learningRate(learningRate), maxIterations(maxIterations), bias(0.0f) {
     
     // Initialize random number generator
     random_device rd;
     rng = mt19937(rd());
     uniform_real_distribution<float> dist(-0.1, 0.1);
     
     // Initialize weights
     weights.resize(numFeatures);
     for (int i = 0; i < numFeatures; ++i) {
         weights[i] = dist(rng);
     }
     
     cout << "Initialized Logistic Regression with " << numFeatures << " features." << endl;
 }
 
 float LogisticRegression::sigmoid(float x) {
     return 1.0f / (1.0f + exp(-x));
 }
 
 vector<float> LogisticRegression::computeGradient(const vector<float>& X, const vector<int>& y, 
                                                  int numSamples, int numFeatures) {
     vector<float> gradient(numFeatures, 0.0f);
     float biasGradient = 0.0f;
     
     // Parallelize the gradient computation over samples
     #pragma omp parallel
     {
         vector<float> threadGradient(numFeatures, 0.0f);
         float threadBiasGradient = 0.0f;
         
         #pragma omp for
         for (int i = 0; i < numSamples; ++i) {
             // Compute prediction
             float logit = bias;
             for (int j = 0; j < numFeatures; ++j) {
                 logit += weights[j] * X[i * numFeatures + j];
             }
             float prediction = sigmoid(logit);
             
             // Compute error
             float error = prediction - y[i];
             
             // Accumulate gradients
             threadBiasGradient += error;
             for (int j = 0; j < numFeatures; ++j) {
                 threadGradient[j] += error * X[i * numFeatures + j];
             }
         }
         
         // Merge thread-local gradients
         #pragma omp critical
         {
             biasGradient += threadBiasGradient;
             for (int j = 0; j < numFeatures; ++j) {
                 gradient[j] += threadGradient[j];
             }
         }
     }
     
     // Normalize by number of samples
     biasGradient /= numSamples;
     for (int j = 0; j < numFeatures; ++j) {
         gradient[j] /= numSamples;
     }
     
     // Update bias (store in the last element of gradient)
     gradient.push_back(biasGradient);
     
     return gradient;
 }
 
 float LogisticRegression::computeLoss(const vector<float>& X, const vector<int>& y, 
                                      int numSamples, int numFeatures) {
     float loss = 0.0f;
     
     #pragma omp parallel for reduction(+:loss)
     for (int i = 0; i < numSamples; ++i) {
         // Compute logit
         float logit = bias;
         for (int j = 0; j < numFeatures; ++j) {
             logit += weights[j] * X[i * numFeatures + j];
         }
         
         // Compute binary cross-entropy loss
         if (y[i] == 1) {
             loss -= log(max(sigmoid(logit), 1e-7f));
         } else {
             loss -= log(max(1.0f - sigmoid(logit), 1e-7f));
         }
     }
     
     return loss / numSamples;
 }
 
 void LogisticRegression::train(const vector<float>& X, const vector<int>& y, 
                               int numSamples, int numFeatures) {
     cout << "Starting Logistic Regression training with " << numSamples << " samples..." << endl;
     
     for (int iter = 0; iter < maxIterations; ++iter) {
         // Compute gradient
         vector<float> gradient = computeGradient(X, y, numSamples, numFeatures);
         
         // Update weights
         for (int j = 0; j < numFeatures; ++j) {
             weights[j] -= learningRate * gradient[j];
         }
         
         // Update bias (last element of gradient)
         bias -= learningRate * gradient[numFeatures];
         
         // Print progress
         if ((iter + 1) % 10 == 0 || iter == 0 || iter == maxIterations - 1) {
             float loss = computeLoss(X, y, numSamples, numFeatures);
             cout << "Logistic Regression Iteration " << (iter + 1) << "/" << maxIterations 
                  << ", Loss: " << loss << endl;
         }
     }
     
     cout << "Logistic Regression training completed." << endl;
 }
 
 vector<int> LogisticRegression::predict(const vector<float>& X, int numSamples, int numFeatures) {
     vector<int> predictions(numSamples);
     
     #pragma omp parallel for
     for (int i = 0; i < numSamples; ++i) {
         float logit = bias;
         for (int j = 0; j < numFeatures; ++j) {
             logit += weights[j] * X[i * numFeatures + j];
         }
         
         predictions[i] = sigmoid(logit) >= 0.5 ? 1 : 0;
     }
     
     return predictions;
 }
 
 vector<float> LogisticRegression::predictProbabilities(const vector<float>& X, int numSamples, int numFeatures) {
     vector<float> probabilities(numSamples);
     
     #pragma omp parallel for
     for (int i = 0; i < numSamples; ++i) {
         float logit = bias;
         for (int j = 0; j < numFeatures; ++j) {
             logit += weights[j] * X[i * numFeatures + j];
         }
         
         probabilities[i] = sigmoid(logit);
     }
     
     return probabilities;
 }
 
 void LogisticRegression::saveModel(const string& filename) {
     ofstream outFile(filename, ios::binary);
     
     if (!outFile.is_open()) {
         cerr << "Error: Could not open file " << filename << " for writing." << endl;
         return;
     }
     
     // Write model parameters
     outFile.write(reinterpret_cast<char*>(&numFeatures), sizeof(numFeatures));
     outFile.write(reinterpret_cast<char*>(&bias), sizeof(bias));
     
     // Write weights
     for (int i = 0; i < numFeatures; ++i) {
         outFile.write(reinterpret_cast<char*>(&weights[i]), sizeof(float));
     }
     
     outFile.close();
     cout << "Logistic Regression model saved to " << filename << endl;
 }
 
 void LogisticRegression::loadModel(const string& filename) {
     ifstream inFile(filename, ios::binary);
     
     if (!inFile.is_open()) {
         cerr << "Error: Could not open file " << filename << " for reading." << endl;
         return;
     }
     
     // Read model parameters
     inFile.read(reinterpret_cast<char*>(&numFeatures), sizeof(numFeatures));
     inFile.read(reinterpret_cast<char*>(&bias), sizeof(bias));
     
     // Read weights
     weights.resize(numFeatures);
     for (int i = 0; i < numFeatures; ++i) {
         inFile.read(reinterpret_cast<char*>(&weights[i]), sizeof(float));
     }
     
     inFile.close();
     cout << "Logistic Regression model loaded from " << filename << endl;
 }