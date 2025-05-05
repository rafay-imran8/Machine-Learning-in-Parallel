/**
 * logistic_regression.h - Definition of the Logistic Regression classifier
 * 
 * This class implements logistic regression with OpenMP parallelization for
 * gradient computation.
 */

 #ifndef LOGISTIC_REGRESSION_H
 #define LOGISTIC_REGRESSION_H
 
 #include <vector>
 #include <string>
 #include <random>
 
 class LogisticRegression {
 private:
     int numFeatures;
     float learningRate;
     int maxIterations;
     
     // Model parameters
     std::vector<float> weights;
     float bias;
     
     // Random number generator
     std::mt19937 rng;
     
     // Helper functions
     float sigmoid(float x);
     std::vector<float> computeGradient(const std::vector<float>& X, const std::vector<int>& y, 
                                       int numSamples, int numFeatures);
     float computeLoss(const std::vector<float>& X, const std::vector<int>& y, 
                      int numSamples, int numFeatures);
 
 public:
     LogisticRegression(int numFeatures, float learningRate = 0.01, int maxIterations = 100);
     void train(const std::vector<float>& X, const std::vector<int>& y, 
                int numSamples, int numFeatures);
     std::vector<int> predict(const std::vector<float>& X, int numSamples, int numFeatures);
     std::vector<float> predictProbabilities(const std::vector<float>& X, int numSamples, int numFeatures);
     void saveModel(const std::string& filename);
     void loadModel(const std::string& filename);
 };
 
 #endif // LOGISTIC_REGRESSION_H