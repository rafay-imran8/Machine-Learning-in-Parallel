/**
 * mlp.h - Definition of the Multilayer Perceptron (MLP) Neural Network
 * 
 * This class implements a fully-connected feed-forward neural network with
 * OpenMP parallelization for matrix operations.
 */

 #ifndef MLP_H
 #define MLP_H
 
 #include <vector>
 #include <string>
 #include <random>
 
 class MLP {
 private:
     // Network architecture
     int inputSize;
     std::vector<int> hiddenSizes;
     int outputSize;
     
     // Weights and biases
     std::vector<std::vector<std::vector<float>>> weights; // [layer][neuron][input]
     std::vector<std::vector<float>> biases; // [layer][neuron]
     
     // Activations and deltas for training
     std::vector<std::vector<float>> activations; // [layer][neuron]
     std::vector<std::vector<float>> deltas; // [layer][neuron]
     
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
     MLP(int inputSize, const std::vector<int>& hiddenSizes, int outputSize);
     void train(const std::vector<float>& X, const std::vector<int>& y, 
                int numSamples, int numFeatures, int epochs = 100, float learningRate = 0.01);
     std::vector<int> predict(const std::vector<float>& X, int numSamples, int numFeatures);
     void saveModel(const std::string& filename);
     void loadModel(const std::string& filename);
 };
 
 #endif // MLP_H