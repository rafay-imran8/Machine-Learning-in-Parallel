// #include "./include/evaluate.h"
// #include <fstream>
// #include <sstream>
// #include <iostream>
// #include <mpi.h>
// #include <omp.h>

// // Declarations for your model interfaces
// #include "./include/random_forest.h"
// #include "./include/mlp.h"
// #include "./include/logistic_regression.h"

// void loadTestData(const std::string& filename,
//                   std::vector<float>& X,
//                   std::vector<int>& y,
//                   int& N,
//                   int& D) {
//     std::ifstream file(filename);
//     std::string line;
//     // Read header
//     std::getline(file, line);

//     std::vector<std::vector<float>> data;
//     std::vector<int> labels;

//     while (std::getline(file, line)) {
//         std::stringstream ss(line);
//         std::string val;
//         std::vector<float> row;
//         int col = 0;
//         while (std::getline(ss, val, ',')) {
//             if (ss.peek() == EOF) {
//                 // last column is label
//                 labels.push_back(std::stoi(val));
//             } else {
//                 row.push_back(std::stof(val));
//             }
//             ++col;
//         }
//         data.push_back(row);
//     }

//     N = data.size();
//     D = data[0].size();
//     X.resize(N * D);
//     y = labels;

//     for (int i = 0; i < N; ++i) {
//         for (int j = 0; j < D; ++j) {
//             X[i*D + j] = data[i][j];
//         }
//     }
// }

// Metrics evaluateModel(const std::string& modelPath,
//                       const std::vector<float>& X,
//                       const std::vector<int>& y,
//                       int N,
//                       int D) {
//     // // Load the appropriate model based on path
//     Metrics m{0,0,0};
   
//     std::unique_ptr<ModelInterface> model;
//     if (modelPath.find("random_forest") != std::string::npos)
//         model = std::make_unique<RandomForest>();
//     else if (modelPath.find("mlp") != std::string::npos)
//         model = std::make_unique<MLP>();
//     else
//         model = std::make_unique<logistic_regression>();
//     model->loadModel(modelPath);

//     // Now delegate to a pure‐interface evaluator
//     return evaluate(*model, X, y, N, D);
//  }

// // New: evaluate against any loaded model, thread‐safe via clone()
// Metrics evaluate(ModelInterface& prototype,
//                  const std::vector<float>& X,
//                  const std::vector<int>& y,
//                  int N,
//                  int D) {
//     int TP=0, FP=0, TN=0, FN=0;

//     #pragma omp parallel reduction(+:TP,FP,TN,FN)
//     {
//         // each thread gets its own model copy
//         auto model = prototype.clone();
//         #pragma omp for
//         for (int i = 0; i < N; ++i) {
//             std::vector<float> feat(X.begin() + i*D, X.begin() + i*D + D);
//             int pred = model->predict(feat);
//             int actual = y[i];
//             if      (pred==1 && actual==1) ++TP;
//             else if (pred==1 && actual==0) ++FP;
//             else if (pred==0 && actual==0) ++TN;
//             else if (pred==0 && actual==1) ++FN;
//         }
//     }

//     Metrics m;
//     m.accuracy  = double(TP + TN) / N;
//     m.precision = (TP + FP) ? double(TP) / (TP + FP) : 0.0;
//     m.recall    = (TP + FN) ? double(TP) / (TP + FN) : 0.0;
//     return m;


// }

// void gatherAndPrintMetrics(const Metrics& localMetrics,
//                            int rank,
//                            int size) {
//     std::vector<Metrics> all(size);
//     MPI_Gather(const_cast<Metrics*>(&localMetrics), sizeof(Metrics), MPI_BYTE,
//                all.data(), sizeof(Metrics), MPI_BYTE,
//                0, MPI_COMM_WORLD);

//     if (rank == 0) {
//         std::cout << "\n=== Evaluation Metrics ===\n";
//         for (int r = 0; r < size; ++r) {
//             std::cout << "Model (rank " << r << "): "
//                       << "Accuracy="  << all[r].accuracy
//                       << ", Precision=" << all[r].precision
//                       << ", Recall="    << all[r].recall << "\n";
//         }
//     }
// }

// #include "./include/evaluate.h"
// #include <iostream>
// #include <fstream>
// #include <sstream>
// #include <string>
// #include <memory>

// // Declarations for your model interfaces
// #include "./include/random_forest.h"
// #include "./include/mlp.h"
// #include "./include/logistic_regression.h"

// void loadTestData(const std::string& filename,
//                   std::vector<float>& X,
//                   std::vector<int>& y,
//                   int& N,
//                   int& D) {
//     std::ifstream file(filename);
//     std::string line;
//     // Read header
//     std::getline(file, line);

//     std::vector<std::vector<float>> data;
//     std::vector<int> labels;

//     while (std::getline(file, line)) {
//         std::stringstream ss(line);
//         std::string val;
//         std::vector<float> row;
//         int col = 0;
//         while (std::getline(ss, val, ',')) {
//             if (ss.peek() == EOF) {
//                 // last column is label
//                 labels.push_back(std::stoi(val));
//             } else {
//                 row.push_back(std::stof(val));
//             }
//             ++col;
//         }
//         data.push_back(row);
//     }

//     N = data.size();
//     D = data[0].size();
//     X.resize(N * D);
//     y = labels;

//     for (int i = 0; i < N; ++i) {
//         for (int j = 0; j < D; ++j) {
//             X[i*D + j] = data[i][j];
//         }
//     }
// }

// // Define the evaluate function before it's used
// Metrics evaluate(const ModelInterface& prototype,
//                  const std::vector<float>& X,
//                  const std::vector<int>& y,
//                  int N,
//                  int D) {
//     int TP=0, FP=0, TN=0, FN=0;

//     #pragma omp parallel reduction(+:TP,FP,TN,FN)
//     {
//         // each thread gets its own model copy
//         auto model = prototype.clone();
//         #pragma omp for
//         for (int i = 0; i < N; ++i) {
//             std::vector<float> feat(X.begin() + i*D, X.begin() + i*D + D);
//             int pred = model->predict(feat);
//             int actual = y[i];
//             if      (pred==1 && actual==1) ++TP;
//             else if (pred==1 && actual==0) ++FP;
//             else if (pred==0 && actual==0) ++TN;
//             else if (pred==0 && actual==1) ++FN;
//         }
//     }

//     Metrics m;
//     m.accuracy  = double(TP + TN) / N;
//     m.precision = (TP + FP) ? double(TP) / (TP + FP) : 0.0;
//     m.recall    = (TP + FN) ? double(TP) / (TP + FN) : 0.0;
//     return m;
// }

// Metrics evaluateModel(const std::string& modelPath,
//                       const std::vector<float>& X,
//                       const std::vector<int>& y,
//                       int N,
//                       int D) {
//     // Load the appropriate model based on path
//     std::unique_ptr<ModelInterface> model;
    
//     if (modelPath.find("random_forest") != std::string::npos) {
//         auto rf = std::make_unique<RandomForest>();
//         rf->loadModel(modelPath);
//         return evaluate(*rf, X, y, N, D);
//     }
//     else if (modelPath.find("mlp") != std::string::npos) {
//         auto mlp = std::make_unique<MLP>();
//         mlp->loadModel(modelPath);
//         return evaluate(*mlp, X, y, N, D);
//     }
//     else {
//         auto lr = std::make_unique<logistic_regression>();
//         lr->loadModel(modelPath);
//         return evaluate(*lr, X, y, N, D);
//     }
// }

// void gatherAndPrintMetrics(const Metrics& localMetrics,
//                            int rank,
//                            int size) {
//     std::vector<Metrics> all(size);
//     MPI_Gather(const_cast<Metrics*>(&localMetrics), sizeof(Metrics), MPI_BYTE,
//                all.data(), sizeof(Metrics), MPI_BYTE,
//                0, MPI_COMM_WORLD);

//     if (rank == 0) {
//         std::cout << "\n=== Evaluation Metrics ===\n";
//         for (int r = 0; r < size; ++r) {
//             std::cout << "Model (rank " << r << "): "
//                       << "Accuracy="  << all[r].accuracy
//                       << ", Precision=" << all[r].precision
//                       << ", Recall="    << all[r].recall << "\n";
//         }
//     }
// }

#include "./include/evaluate.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <memory>

// Declarations for your model interfaces
#include "./include/random_forest.h"
#include "./include/mlp.h"
#include "./include/logistic_regression.h"

void loadTestData(const std::string& filename,
                  std::vector<float>& X,
                  std::vector<int>& y,
                  int& N,
                  int& D) {
    std::ifstream file(filename);
    std::string line;
    // Read header
    std::getline(file, line);

    std::vector<std::vector<float>> data;
    std::vector<int> labels;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string val;
        std::vector<float> row;
        int col = 0;
        while (std::getline(ss, val, ',')) {
            if (ss.peek() == EOF) {
                // last column is label
                labels.push_back(std::stoi(val));
            } else {
                row.push_back(std::stof(val));
            }
            ++col;
        }
        data.push_back(row);
    }

    N = data.size();
    D = data[0].size();
    X.resize(N * D);
    y = labels;

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < D; ++j) {
            X[i*D + j] = data[i][j];
        }
    }
}

// Define the evaluate function before it's used
Metrics evaluate(const ModelInterface& prototype,
                 const std::vector<float>& X,
                 const std::vector<int>& y,
                 int N,
                 int D) {
    int TP=0, FP=0, TN=0, FN=0;

    #pragma omp parallel reduction(+:TP,FP,TN,FN)
    {
        // each thread gets its own model copy
        auto model = prototype.clone();
        #pragma omp for
        for (int i = 0; i < N; ++i) {
            std::vector<float> feat(X.begin() + i*D, X.begin() + i*D + D);
            int pred = model->predict(feat);
            int actual = y[i];
            if      (pred==1 && actual==1) ++TP;
            else if (pred==1 && actual==0) ++FP;
            else if (pred==0 && actual==0) ++TN;
            else if (pred==0 && actual==1) ++FN;
        }
    }

    Metrics m;
    m.accuracy  = double(TP + TN) / N;
    m.precision = (TP + FP) ? double(TP) / (TP + FP) : 0.0;
    m.recall    = (TP + FN) ? double(TP) / (TP + FN) : 0.0;
    return m;
}

Metrics evaluateModel(const std::string& modelPath,
                      const std::vector<float>& X,
                      const std::vector<int>& y,
                      int N,
                      int D) {
    // Load the appropriate model based on path
    if (modelPath.find("random_forest") != std::string::npos) {
        auto rf = std::make_unique<RandomForest>();
        rf->loadModel(modelPath);
        return evaluate(*rf, X, y, N, D);
    }
    else if (modelPath.find("mlp") != std::string::npos) {
        auto mlp = std::make_unique<MLP>();
        mlp->loadModel(modelPath);
        return evaluate(*mlp, X, y, N, D);
    }
    else {
        auto lr = std::make_unique<LogisticRegression>();
        lr->loadModel(modelPath);
        return evaluate(*lr, X, y, N, D);
    }
}

void gatherAndPrintMetrics(const Metrics& localMetrics,
                           int rank,
                           int size) {
    std::vector<Metrics> all(size);
    MPI_Gather(const_cast<Metrics*>(&localMetrics), sizeof(Metrics), MPI_BYTE,
               all.data(), sizeof(Metrics), MPI_BYTE,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "\n=== Evaluation Metrics ===\n";
        for (int r = 0; r < size; ++r) {
            std::cout << "Model (rank " << r << "): "
                      << "Accuracy="  << all[r].accuracy
                      << ", Precision=" << all[r].precision
                      << ", Recall="    << all[r].recall << "\n";
        }
    }
}