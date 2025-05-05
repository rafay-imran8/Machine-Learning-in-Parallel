/**
 * random_forest.h - Definition of the Random Forest classifier
 * 
 * This class implements a Random Forest classifier from scratch with OpenMP parallelization
 * for both tree-level and feature-split calculations.
 */

 #ifndef RANDOM_FOREST_H
 #define RANDOM_FOREST_H
 
 #include <vector>
 #include <random>
 #include <string>
 #include <fstream>
 #include <cmath>
 #include <algorithm>
 #include <omp.h>
 
 class Node {
 public:
     bool isLeaf;
     int featureIndex;
     float threshold;
     int classLabel;
     Node* left;
     Node* right;
 
     Node() : isLeaf(false), featureIndex(-1), threshold(0.0f), classLabel(-1), left(nullptr), right(nullptr) {}
     ~Node() {
         delete left;
         delete right;
     }
 };
 
 class DecisionTree {
 private:
     Node* root;
     int maxDepth;
     int minSamplesLeaf;
     int numFeatures;
     int mtry;  // Number of features to consider at each split
     std::mt19937 rng;
 
     // Helper functions
     Node* buildTree(const std::vector<float>& X, const std::vector<int>& y, 
                     const std::vector<int>& sampleIndices, int depth);
     std::pair<int, float> findBestSplit(const std::vector<float>& X, const std::vector<int>& y, 
                                         const std::vector<int>& sampleIndices, const std::vector<int>& featureIndices);
     float calculateGini(const std::vector<int>& y, const std::vector<int>& sampleIndices);
     void predict(const std::vector<float>& x, Node* node, int& prediction);
     void saveTreeRecursive(Node* node, std::ofstream& file);
     Node* loadTreeRecursive(std::ifstream& file);
 
 public:
     DecisionTree(int maxDepth, int minSamplesLeaf, int numFeatures, unsigned int seed);
     ~DecisionTree();
     void train(const std::vector<float>& X, const std::vector<int>& y, int numSamples, int numFeatures);
     int predict(const std::vector<float>& x);
     void saveTree(const std::string& filename);
     void loadTree(const std::string& filename);
 };
 
 class RandomForest {
 private:
     std::vector<DecisionTree*> trees;
     int numTrees;
     int maxDepth;
     int minSamplesLeaf;
     int numFeatures;
 
 public:
     RandomForest(int numTrees, int maxDepth, int minSamplesLeaf, int numFeatures);
     ~RandomForest();
     void train(const std::vector<float>& X, const std::vector<int>& y, int numSamples, int numFeatures);
     int predict(const std::vector<float>& x);
     void saveModel(const std::string& prefix);
     void loadModel(const std::string& prefix, int numTrees);
 };
 
 #endif // RANDOM_FOREST_H