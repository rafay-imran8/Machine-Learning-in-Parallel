// #ifndef MLP_H
// #define MLP_H

// #include <vector>
// #include <string>
// #include <random>
// #include <memory>
// #include "evaluate.h"  // For ModelInterface

// /**
//  * mlp.h - Definition of the Multilayer Perceptron (MLP) Neural Network
//  *
//  * This class implements a fully-connected feed-forward neural network with
//  * OpenMP parallelization for matrix operations.
//  */

// class MLP : public ModelInterface {
// private:
//     // Network architecture
//     int inputSize;
//     std::vector<int> hiddenSizes;
//     int outputSize;
    
//     // Weights and biases
//     std::vector<std::vector<std::vector<float>>> weights; // [layer][neuron][input]
//     std::vector<std::vector<float>> biases;                // [layer][neuron]
    
//     // Activations and deltas for training
//     std::vector<std::vector<float>> activations; // [layer][neuron]
//     std::vector<std::vector<float>> deltas;      // [layer][neuron]
    
//     // Random number generator
//     std::mt19937 rng;
    
//     // Helper functions
//     float sigmoid(float x);
//     float sigmoidDerivative(float x);
//     void forwardPass(const std::vector<float>& input);
//     void backwardPass(const std::vector<float>& input, const std::vector<float>& target);
//     void updateWeights(const std::vector<float>& input, float learningRate);
//     std::vector<float> oneHotEncode(int label, int numClasses);

// public:
//     MLP(int inputSize = 0, const std::vector<int>& hiddenSizes = {}, int outputSize = 0);
//     ~MLP() override = default;
    
//     // Original training and batch prediction methods
//     void train(const std::vector<float>& X, const std::vector<int>& y, 
//                int numSamples, int numFeatures, int epochs = 100, float learningRate = 0.01);
//     std::vector<int> predict(const std::vector<float>& X, int numSamples, int numFeatures);
    
//     // Model saving methods (original)
//     void saveModel(const std::string& filename);
    
//     // Implementation of ModelInterface methods
//     void loadModel(const std::string& path) override;
//     int predict(const std::vector<float>& features) override; // Single sample prediction

//     // Clone method for thread-safe evaluation
//     std::unique_ptr<ModelInterface> clone() const override {
//         return std::make_unique<MLP>(*this);
//     }
// };

// #endif // MLP_H


#ifndef MLP_H
#define MLP_H

#include <vector>
#include <string>
#include <random>
#include <memory>
#include "evaluate.h"  // For ModelInterface

/**
 * mlp.h - Definition of the Multilayer Perceptron (MLP) Neural Network
 *
 * This class implements a fully-connected feed-forward neural network with
 * OpenMP parallelization for matrix operations.
 */

class MLP : public ModelInterface {
private:
    // Network architecture
    int inputSize;
    std::vector<int> hiddenSizes;
    int outputSize;
    
    // Weights and biases
    std::vector<std::vector<std::vector<float>>> weights; // [layer][neuron][input]
    std::vector<std::vector<float>> biases;                // [layer][neuron]
    
    // Activations and deltas for training
    std::vector<std::vector<float>> activations; // [layer][neuron]
    std::vector<std::vector<float>> deltas;      // [layer][neuron]
    
    // Random number generator
    std::mt19937 rng;
    
    // Helper functions
    float sigmoid(float x);
    float sigmoidDerivative(float x);
    void forwardPass(const std::vector<float>& input);
    void backwardPass(const std::vector<float>& input, const std::vector<float>& target);
    void updateWeights(const std::vector<float>& input, float learningRate);
    std::vector<float> oneHotEncode(int label, int numClasses);

public:
    MLP(int inputSize = 0, const std::vector<int>& hiddenSizes = {}, int outputSize = 0);
    ~MLP() override = default;
    
    // Original training and batch prediction methods
    void train(const std::vector<float>& X, const std::vector<int>& y, 
               int numSamples, int numFeatures, int epochs = 100, float learningRate = 0.01);
    std::vector<int> predict(const std::vector<float>& X, int numSamples, int numFeatures);
    
    // Model saving methods (original)
    void saveModel(const std::string& filename);
    
    // Implementation of ModelInterface methods
    void loadModel(const std::string& path) override;
    int predict(const std::vector<float>& features) override; // Single sample prediction

    // Clone method for thread-safe evaluation
    std::unique_ptr<ModelInterface> clone() const override {
        return std::make_unique<MLP>(*this);
    }
};

#endif // MLP_H