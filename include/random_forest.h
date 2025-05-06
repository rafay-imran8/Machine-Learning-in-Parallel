#ifndef RANDOM_FOREST_H
#define RANDOM_FOREST_H

#include <vector>
#include <string>
#include <memory>
#include <random>
#include <fstream>
#include <utility>
#include <omp.h>
#include "evaluate.h"  // For ModelInterface

/**
 * random_forest.h - Definition of the Random Forest classifier
 *
 * Implements a Random Forest classifier with OpenMP parallelization
 * and integrates with the evaluation framework.
 */

// Tree node for classification
class Node {
public:
    bool isLeaf = false;
    int featureIndex = -1;
    float threshold = 0.0f;
    int classLabel = -1;
    Node* left = nullptr;
    Node* right = nullptr;
};

// Single decision tree
class DecisionTree {
public:
    DecisionTree(int maxDepth, int minSamplesLeaf, int numFeatures, unsigned int seed);
    ~DecisionTree();

    void train(const std::vector<float>& X,
               const std::vector<int>& y,
               int numSamples,
               int numFeatures);
    int predict(const std::vector<float>& x);
    void saveTree(const std::string& filename);
    void loadTree(const std::string& filename);

private:
    Node* root;
    int maxDepth;
    int minSamplesLeaf;
    int numFeatures;
    int mtry;
    std::mt19937 rng;

    Node* buildTree(const std::vector<float>& X,
                    const std::vector<int>& y,
                    const std::vector<int>& sampleIndices,
                    int depth);
    std::pair<int, float> findBestSplit(const std::vector<float>& X,
                                        const std::vector<int>& y,
                                        const std::vector<int>& sampleIndices,
                                        const std::vector<int>& featureIndices);
    float calculateGini(const std::vector<int>& y,
                        const std::vector<int>& sampleIndices);
    void predict(const std::vector<float>& x,
                 Node* node,
                 int& prediction);
    void saveTreeRecursive(Node* node,
                           std::ofstream& file);
    Node* loadTreeRecursive(std::ifstream& file);
};

// Random forest ensemble
class RandomForest : public ModelInterface {
public:
    RandomForest(int numTrees = 10,
                 int maxDepth = 5,
                 int minSamplesLeaf = 1,
                 int numFeatures = 0);
    ~RandomForest() override;

    // Load and save
    void loadModel(const std::string& path) override;
    void saveModel(const std::string& prefix);
    void loadModel(const std::string& prefix, int numTrees);

    // Training
    void train(const std::vector<float>& X,
               const std::vector<int>& y,
               int numSamples,
               int numFeatures);

    // Single-sample API
    int predict(const std::vector<float>& features) override;
    std::unique_ptr<ModelInterface> clone() const override;

private:
    std::vector<std::shared_ptr<DecisionTree>> trees;
    int numTrees;
    int maxDepth;
    int minSamplesLeaf;
    int numFeatures;
    std::string modelPath;
};

#endif // RANDOM_FOREST_H