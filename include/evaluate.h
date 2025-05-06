// #ifndef EVALUATION_H
// #define EVALUATION_H

// #include <vector>
// #include <string>
// #include <mpi.h>
// #include <omp.h>
// #include <memory>

// // Forward declarations for model classes
// class RandomForest;
// class MLP;
// class logistic_regression;

// // Define the ModelInterface that all models should implement
// class ModelInterface {
// public:
//     virtual ~ModelInterface() = default;
//     virtual void loadModel(const std::string& path) = 0;
//     virtual int predict(const std::vector<float>& features) = 0;
//     // make a full copy, so each thread can have its own instance
//    virtual std::unique_ptr<ModelInterface> clone() const = 0;
// };

// // Struct to hold evaluation metrics
// // Make sure this is properly aligned for MPI communication
// #pragma pack(push, 1)
// struct Metrics {
//     double accuracy;
//     double precision;
//     double recall;
// };
// #pragma pack(pop)

// // Load test data from CSV: fills X (row-major N*D), y (size N), and sets N (#samples) and D (#features)
// void loadTestData(const std::string& filename,
//                   std::vector<float>& X,
//                   std::vector<int>& y,
//                   int& N,
//                   int& D);

// // Evaluate a model at modelPath on dataset (X,y) with dimensions N x D
// // Uses OpenMP to parallelize predictions and accumulate TP, FP, TN, FN
// Metrics evaluateModel(const std::string& modelPath,
//                       const std::vector<float>& X,
//                       const std::vector<int>& y,
//                       int N,
//                       int D);

// // Gather metrics from all ranks and print summary on rank 0
// void gatherAndPrintMetrics(const Metrics& localMetrics,
//                           int rank,
//                           int size);

// #endif // EVALUATION_H

// #ifndef EVALUATION_H
// #define EVALUATION_H

// #include <vector>
// #include <string>
// #include <memory>
// #include <iostream>
// #include <mpi.h>

// // Forward declarations for model classes
// class RandomForest;
// class MLP;
// class logistic_regression;

// // Define the ModelInterface that all models should implement
// class ModelInterface {
// public:
//     virtual ~ModelInterface() = default;
//     virtual void loadModel(const std::string& path) = 0;
//     virtual int predict(const std::vector<float>& features) const = 0;
//     // make a full copy, so each thread can have its own instance
//     virtual std::unique_ptr<ModelInterface> clone() const = 0;
// };

// // Struct to hold evaluation metrics
// // Make sure this is properly aligned for MPI communication
// #pragma pack(push, 1)
// struct Metrics {
//     double accuracy;
//     double precision;
//     double recall;
// };
// #pragma pack(pop)

// // Load test data from CSV: fills X (row-major N*D), y (size N), and sets N (#samples) and D (#features)
// void loadTestData(const std::string& filename,
//                   std::vector<float>& X,
//                   std::vector<int>& y,
//                   int& N,
//                   int& D);

// // Evaluate a model at modelPath on dataset (X,y) with dimensions N x D
// // Uses OpenMP to parallelize predictions and accumulate TP, FP, TN, FN
// Metrics evaluateModel(const std::string& modelPath,
//                       const std::vector<float>& X,
//                       const std::vector<int>& y,
//                       int N,
//                       int D);

// // Function to evaluate any model that implements ModelInterface
// Metrics evaluate(const ModelInterface& prototype,
//                  const std::vector<float>& X,
//                  const std::vector<int>& y,
//                  int N,
//                  int D);

// // Gather metrics from all ranks and print summary on rank 0
// void gatherAndPrintMetrics(const Metrics& localMetrics,
//                           int rank,
//                           int size);

// #endif // EVALUATION_H
#ifndef EVALUATION_H
#define EVALUATION_H

#include <vector>
#include <string>
#include <memory>
#include <iostream>
#include <mpi.h>

// Forward declarations for model classes
class RandomForest;
class MLP;
class LogisticRegression;  // Updated to match your actual class name

// Define the ModelInterface that all models should implement
class ModelInterface {
public:
    virtual ~ModelInterface() = default;
    virtual void loadModel(const std::string& path) = 0;
    virtual int predict(const std::vector<float>& features) = 0;  // Note: Not const to match your implementation
    // make a full copy, so each thread can have its own instance
    virtual std::unique_ptr<ModelInterface> clone() const = 0;
};

// Struct to hold evaluation metrics
// Make sure this is properly aligned for MPI communication
#pragma pack(push, 1)
struct Metrics {
    double accuracy;
    double precision;
    double recall;
};
#pragma pack(pop)

// Load test data from CSV: fills X (row-major N*D), y (size N), and sets N (#samples) and D (#features)
void loadTestData(const std::string& filename,
                  std::vector<float>& X,
                  std::vector<int>& y,
                  int& N,
                  int& D);

// Evaluate a model at modelPath on dataset (X,y) with dimensions N x D
// Uses OpenMP to parallelize predictions and accumulate TP, FP, TN, FN
Metrics evaluateModel(const std::string& modelPath,
                      const std::vector<float>& X,
                      const std::vector<int>& y,
                      int N,
                      int D);

// Function to evaluate any model that implements ModelInterface
Metrics evaluate(const ModelInterface& prototype,
                 const std::vector<float>& X,
                 const std::vector<int>& y,
                 int N,
                 int D);

// Gather metrics from all ranks and print summary on rank 0
void gatherAndPrintMetrics(const Metrics& localMetrics,
                          int rank,
                          int size);

#endif // EVALUATION_H