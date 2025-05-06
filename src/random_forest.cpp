/**
 * random_forest.cpp - Implementation of the Random Forest classifier
 */

 #include "include/random_forest.h"
 #include <iostream>
 #include <unordered_map>
 #include <limits>
 #include <cmath>
 #include <algorithm>
 #include <numeric>
 #include <fstream>
 #include "include/omp_config.h"
 #include <chrono>
 
 // Decision Tree implementation
 DecisionTree::DecisionTree(int maxDepth, int minSamplesLeaf, int numFeatures, unsigned int seed) 
     : root(nullptr), maxDepth(maxDepth), minSamplesLeaf(minSamplesLeaf), numFeatures(numFeatures), rng(seed) {
     // Calculate mtry (number of features to consider at each split)
     mtry = static_cast<int>(std::sqrt(numFeatures));
     mtry = std::max(1, mtry);
 }
 
 DecisionTree::~DecisionTree() {
     delete root;
 }
 
 void DecisionTree::train(const std::vector<float>& X, const std::vector<int>& y, int numSamples, int numFeatures) {
     this->numFeatures = numFeatures;
     
     // Create bootstrap sample indices
     std::vector<int> sampleIndices(numSamples);
     std::iota(sampleIndices.begin(), sampleIndices.end(), 0);
     
     std::uniform_int_distribution<int> dist(0, numSamples - 1);
     for (int i = 0; i < numSamples; ++i) {
         sampleIndices[i] = dist(rng);
     }
     
     // Build the tree recursively
     root = buildTree(X, y, sampleIndices, 0);
 }
 
 Node* DecisionTree::buildTree(const std::vector<float>& X, const std::vector<int>& y, 
                            const std::vector<int>& sampleIndices, int depth) {
     Node* node = new Node();
     
     // Check stopping criteria - using static_cast to avoid signed/unsigned comparison
     if (depth >= maxDepth || sampleIndices.size() <= static_cast<size_t>(minSamplesLeaf)) {
         node->isLeaf = true;
         
         // Determine the most common class
         std::unordered_map<int, int> classCounts;
         for (int idx : sampleIndices) {
             classCounts[y[idx]]++;
         }
         
         int maxCount = -1;
         for (const auto& pair : classCounts) {
             if (pair.second > maxCount) {
                 maxCount = pair.second;
                 node->classLabel = pair.first;
             }
         }
         
         return node;
     }
     
     // Select a random subset of features to consider
     std::vector<int> featureIndices(numFeatures);
     std::iota(featureIndices.begin(), featureIndices.end(), 0);
     std::shuffle(featureIndices.begin(), featureIndices.end(), rng);
     featureIndices.resize(mtry);
     
     // Find the best split
     auto [featureIndex, threshold] = findBestSplit(X, y, sampleIndices, featureIndices);
     
     // If no good split was found, make this a leaf node
     if (featureIndex == -1) {
         node->isLeaf = true;
         
         // Determine the most common class
         std::unordered_map<int, int> classCounts;
         for (int idx : sampleIndices) {
             classCounts[y[idx]]++;
         }
         
         int maxCount = -1;
         for (const auto& pair : classCounts) {
             if (pair.second > maxCount) {
                 maxCount = pair.second;
                 node->classLabel = pair.first;
             }
         }
         
         return node;
     }
     
     // Set the split information
     node->featureIndex = featureIndex;
     node->threshold = threshold;
     
     // Split the samples
     std::vector<int> leftIndices, rightIndices;
     for (int idx : sampleIndices) {
         if (X[idx * numFeatures + featureIndex] <= threshold) {
             leftIndices.push_back(idx);
         } else {
             rightIndices.push_back(idx);
         }
     }
     
     // If one of the splits is empty, make this a leaf node
     if (leftIndices.empty() || rightIndices.empty()) {
         node->isLeaf = true;
         
         // Determine the most common class
         std::unordered_map<int, int> classCounts;
         for (int idx : sampleIndices) {
             classCounts[y[idx]]++;
         }
         
         int maxCount = -1;
         for (const auto& pair : classCounts) {
             if (pair.second > maxCount) {
                 maxCount = pair.second;
                 node->classLabel = pair.first;
             }
         }
         
         return node;
     }
     
     // Recursively build the left and right subtrees
     node->left = buildTree(X, y, leftIndices, depth + 1);
     node->right = buildTree(X, y, rightIndices, depth + 1);
     
     return node;
 }
 
 std::pair<int, float> DecisionTree::findBestSplit(const std::vector<float>& X, const std::vector<int>& y, 
                                                const std::vector<int>& sampleIndices, const std::vector<int>& featureIndices) {
     float bestGini = std::numeric_limits<float>::max();
     int bestFeatureIndex = -1;
     float bestThreshold = 0.0f;
     
     // Removing unused parentGini calculation
     // float parentGini = calculateGini(y, sampleIndices);
     
     // Parallelized search for the best split across features and thresholds
     // Using size_t for loop counters to avoid signed/unsigned comparison warnings
     #pragma omp parallel for collapse(2) schedule(dynamic) shared(bestGini, bestFeatureIndex, bestThreshold)
     for (size_t f = 0; f < featureIndices.size(); ++f) {
         for (size_t s = 0; s < sampleIndices.size(); ++s) {
             int featureIndex = featureIndices[f];
             int sampleIndex = sampleIndices[s];
             float threshold = X[sampleIndex * numFeatures + featureIndex];
             
             // Split the samples
             std::vector<int> leftIndices, rightIndices;
             for (int idx : sampleIndices) {
                 if (X[idx * numFeatures + featureIndex] <= threshold) {
                     leftIndices.push_back(idx);
                 } else {
                     rightIndices.push_back(idx);
                 }
             }
             
             // Skip if the split is too unbalanced - using static_cast for comparison
             if (leftIndices.size() < static_cast<size_t>(minSamplesLeaf) || 
                 rightIndices.size() < static_cast<size_t>(minSamplesLeaf)) {
                 continue;
             }
             
             // Calculate the weighted gini impurity
             float leftGini = calculateGini(y, leftIndices);
             float rightGini = calculateGini(y, rightIndices);
             float weightedGini = (leftIndices.size() * leftGini + rightIndices.size() * rightGini) / sampleIndices.size();
             
             // Update the best split if this one is better
             #pragma omp critical
             {
                 if (weightedGini < bestGini) {
                     bestGini = weightedGini;
                     bestFeatureIndex = featureIndex;
                     bestThreshold = threshold;
                 }
             }
         }
     }
     
     return {bestFeatureIndex, bestThreshold};
 }
 
 float DecisionTree::calculateGini(const std::vector<int>& y, const std::vector<int>& sampleIndices) {
     if (sampleIndices.empty()) {
         return 0.0f;
     }
     
     std::unordered_map<int, int> classCounts;
     for (int idx : sampleIndices) {
         classCounts[y[idx]]++;
     }
     
     float gini = 1.0f;
     for (const auto& pair : classCounts) {
         float probability = static_cast<float>(pair.second) / sampleIndices.size();
         gini -= probability * probability;
     }
     
     return gini;
 }
 
 int DecisionTree::predict(const std::vector<float>& x) {
     int prediction = -1;
     predict(x, root, prediction);
     return prediction;
 }
 
 void DecisionTree::predict(const std::vector<float>& x, Node* node, int& prediction) {
     if (node->isLeaf) {
         prediction = node->classLabel;
         return;
     }
     
     if (x[node->featureIndex] <= node->threshold) {
         predict(x, node->left, prediction);
     } else {
         predict(x, node->right, prediction);
     }
 }
 
 void DecisionTree::saveTree(const std::string& filename) {
     std::ofstream file(filename, std::ios::binary);
     saveTreeRecursive(root, file);
     file.close();
 }
 
 void DecisionTree::saveTreeRecursive(Node* node, std::ofstream& file) {
     // Write if it's a leaf node
     file.write(reinterpret_cast<const char*>(&node->isLeaf), sizeof(bool));
     
     if (node->isLeaf) {
         // Write class label
         file.write(reinterpret_cast<const char*>(&node->classLabel), sizeof(int));
     } else {
         // Write split information
         file.write(reinterpret_cast<const char*>(&node->featureIndex), sizeof(int));
         file.write(reinterpret_cast<const char*>(&node->threshold), sizeof(float));
         
         // Recursively save left and right subtrees
         saveTreeRecursive(node->left, file);
         saveTreeRecursive(node->right, file);
     }
 }
 
 void DecisionTree::loadTree(const std::string& filename) {
     std::ifstream file(filename, std::ios::binary);
     delete root;  // Delete the existing tree
     root = loadTreeRecursive(file);
     file.close();
 }
 
 Node* DecisionTree::loadTreeRecursive(std::ifstream& file) {
     Node* node = new Node();
     
     // Read if it's a leaf node
     file.read(reinterpret_cast<char*>(&node->isLeaf), sizeof(bool));
     
     if (node->isLeaf) {
         // Read class label
         file.read(reinterpret_cast<char*>(&node->classLabel), sizeof(int));
     } else {
         // Read split information
         file.read(reinterpret_cast<char*>(&node->featureIndex), sizeof(int));
         file.read(reinterpret_cast<char*>(&node->threshold), sizeof(float));
         
         // Recursively load left and right subtrees
         node->left = loadTreeRecursive(file);
         node->right = loadTreeRecursive(file);
     }
     
     return node;
 }
 
 // Random Forest implementation
 RandomForest::RandomForest(int numTrees, int maxDepth, int minSamplesLeaf, int numFeatures)
     : numTrees(numTrees), maxDepth(maxDepth), minSamplesLeaf(minSamplesLeaf), numFeatures(numFeatures) {
     // Setup OpenMP threads according to configuration
     setup_openmp_threads();
     trees.resize(numTrees);
 }
 
 RandomForest::~RandomForest() {
     // No need to manually delete shared_ptr objects
 }
 
 void RandomForest::train(const std::vector<float>& X, const std::vector<int>& y, int numSamples, int numFeatures) {
     std::cout << "Training Random Forest with " << numTrees << " trees, " 
               << numSamples << " samples, and " << numFeatures << " features..." << std::endl;
     
     // Using OpenMP to parallelize tree training
     #pragma omp parallel for schedule(dynamic)
     for (int i = 0; i < numTrees; ++i) {
         // Each tree gets a different random seed
         unsigned int seed = std::chrono::system_clock::now().time_since_epoch().count() + i;
         
         // Use make_shared instead of new
         trees[i] = std::make_shared<DecisionTree>(maxDepth, minSamplesLeaf, numFeatures, seed);
         trees[i]->train(X, y, numSamples, numFeatures);
         
         #pragma omp critical
         {
             std::cout << "Tree " << i + 1 << "/" << numTrees << " trained." << std::endl;
         }
     }
     
     std::cout << "Random Forest training completed." << std::endl;
 }
 
 int RandomForest::predict(const std::vector<float>& x) {
     std::unordered_map<int, int> votes;
     
     // Each tree votes for a class
     for (auto tree : trees) {
         int prediction = tree->predict(x);
         votes[prediction]++;
     }
     
     // Find the class with the most votes
     int maxVotes = -1;
     int prediction = -1;
     
     for (const auto& pair : votes) {
         if (pair.second > maxVotes) {
             maxVotes = pair.second;
             prediction = pair.first;
         }
     }
     
     return prediction;
 }
 
 void RandomForest::saveModel(const std::string& prefix) {
     // Save each tree to a separate file
     for (int i = 0; i < numTrees; ++i) {
         std::string filename = prefix + "_tree_" + std::to_string(i) + ".bin";
         trees[i]->saveTree(filename);
     }
     
     // Save the forest metadata
     std::ofstream metafile(prefix + "_meta.txt");
     metafile << numTrees << " " << maxDepth << " " << minSamplesLeaf << " " << numFeatures << std::endl;
     metafile.close();
     
     std::cout << "Random Forest model saved with prefix: " << prefix << std::endl;
 }
 
 void RandomForest::loadModel(const std::string& path) {
     // Implementing the interface method - use the prefix-based implementation
     modelPath = path;
     loadModel(path, numTrees);
 }
 
 void RandomForest::loadModel(const std::string& prefix, int numTrees) {
     // First, clear the existing trees
     trees.clear();
     
     // Load the metadata
     std::ifstream metafile(prefix + "_meta.txt");
     metafile >> this->numTrees >> maxDepth >> minSamplesLeaf >> numFeatures;
     metafile.close();
     
     // Load each tree
     trees.resize(numTrees);
     for (int i = 0; i < numTrees; ++i) {
         std::string filename = prefix + "_tree_" + std::to_string(i) + ".bin";
         trees[i] = std::make_shared<DecisionTree>(maxDepth, minSamplesLeaf, numFeatures, i);
         trees[i]->loadTree(filename);
     }
     
     std::cout << "Random Forest model loaded from prefix: " << prefix << std::endl;
 }
 
 std::unique_ptr<ModelInterface> RandomForest::clone() const {
     return std::make_unique<RandomForest>(*this);
 }