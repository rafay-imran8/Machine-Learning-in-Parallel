/**
 * logistic_regression.h - Definition of the Logistic Regression classifier
 * 
 * This class implements logistic regression with OpenMP parallelization for
 * gradient computation and integrates with the evaluation framework.
 */

 #ifndef LOGISTIC_REGRESSION_H
 #define LOGISTIC_REGRESSION_H
 
 #include <vector>
 #include <string>
 #include <random>
 #include <memory>
 #include "evaluate.h"  // For ModelInterface
 
 class LogisticRegression : public ModelInterface {
 public:
     LogisticRegression(int numFeatures = 0,
                        float learningRate = 0.01f,
                        int maxIterations = 100);
     ~LogisticRegression() override = default;
 
     // ModelInterface methods
     void loadModel(const std::string& filename) override;
     int predict(const std::vector<float>& features) override;
     std::unique_ptr<ModelInterface> clone() const override;
 
     // Batch operations
     void train(const std::vector<float>& X,
                const std::vector<int>& y,
                int numSamples,
                int numFeatures);
     std::vector<int> predict(const std::vector<float>& X,
                              int numSamples,
                              int numFeatures);
     std::vector<float> predictProbabilities(const std::vector<float>& X,
                                             int numSamples,
                                             int numFeatures);
     void saveModel(const std::string& filename);
 
 private:
     int numFeatures;
     float learningRate;
     int maxIterations;
 
     // Model parameters
     std::vector<float> weights;
     float bias;
 
     // RNG
     std::mt19937 rng;
 
     // Helpers
     float sigmoid(float x);
     std::vector<float> computeGradient(const std::vector<float>& X,
                                        const std::vector<int>& y,
                                        int numSamples,
                                        int numFeatures);
     float computeLoss(const std::vector<float>& X,
                       const std::vector<int>& y,
                       int numSamples,
                       int numFeatures);
 };
 
 #endif // LOGISTIC_REGRESSION_H