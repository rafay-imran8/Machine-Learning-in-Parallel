/**
 * model_evaluator.cpp
 * Main program to evaluate trained ML models in parallel
 */

 #include "../include/evaluate.h"
 #include "../include/common.h"
 #include "../include/csv.h"
 #include <mpi.h>
 #include <iostream>
 #include <string>
 #include <vector>
 #include <algorithm>
 
 int main(int argc, char* argv[]) {
     // Initialize MPI
     MPI_Init(&argc, &argv);
     int rank, size;
     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
     MPI_Comm_size(MPI_COMM_WORLD, &size);
 
     // Parse command line arguments
     if (argc < 3) {
         if (rank == 0) {
             std::cerr << "Usage: " << argv[0] << " <test_data.csv> <model1_path> [model2_path] ...\n";
         }
         MPI_Finalize();
         return 1;
     }
 
     const std::string testDataFile = argv[1];
     std::vector<std::string> modelPaths;
     
     // Collect all model paths
     for (int i = 2; i < argc; i++) {
         modelPaths.push_back(argv[i]);
     }
 
     // Distribute models among MPI ranks
     std::vector<std::string> localModelPaths;
     for (size_t i = rank; i < modelPaths.size(); i += size) {
         localModelPaths.push_back(modelPaths[i]);
     }
 
     if (rank == 0) {
         std::cout << "Evaluating " << modelPaths.size() << " models using " 
                   << size << " MPI processes\n";
     }
 
     // Load test data (everyone loads the same data)
     std::vector<float> X;
     std::vector<int> y;
     int N, D;
     
     if (rank == 0) {
         std::cout << "Loading test data from " << testDataFile << "...\n";
     }
     
     loadTestData(testDataFile, X, y, N, D);
     
     if (rank == 0) {
         std::cout << "Loaded " << N << " samples with " << D << " features\n";
     }
 
     // Evaluate each local model
     for (const auto& modelPath : localModelPaths) {
         if (rank == 0) {
             std::cout << "Evaluating model: " << modelPath << std::endl;
         }
         
         Metrics metrics = evaluateModel(modelPath, X, y, N, D);
         
         // Gather and print metrics from all processes
         MPI_Barrier(MPI_COMM_WORLD);
         gatherAndPrintMetrics(metrics, rank, size);
     }
 
     MPI_Finalize();
     return 0;
 }