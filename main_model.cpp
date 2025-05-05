/**
 * main.cpp - MPI orchestration for the hybrid parallel machine learning system
 * 
 * This file manages the MPI communication and coordinates the training of three
 * classification algorithms in parallel: Random Forest, MLP, and Logistic Regression.
 */

 #include <mpi.h>
 #include <iostream>
 #include <fstream>
 #include <sstream>
 #include <vector>
 #include <string>
 #include <chrono>
 #include <algorithm>
 #include <numeric>
 #include <iomanip>
 #include "include/random_forest.h"
 #include "include/mlp.h"
 #include "include/logistic_regression.h"
 
 using namespace std;
 
 // Function to load data from CSV
 void loadData(const string& filename, vector<float>& X, vector<int>& y, int& numSamples, int& numFeatures) {
     ifstream file(filename);
     if (!file.is_open()) {
         cerr << "Error: Unable to open file " << filename << endl;
         MPI_Abort(MPI_COMM_WORLD, 1);
     }
 
     string line;
     vector<vector<float>> dataList;
     vector<int> labelList;
 
     // Skip the header (if present)
     getline(file, line);
 
     // Read the data
     while (getline(file, line)) {
         stringstream ss(line);
         string value;
         vector<float> row;
         int label = -1;
         int colIndex = 0;
 
         while (getline(ss, value, ',')) {
             float val = stof(value);
             if (colIndex == 5) { // Assuming label is the 6th column
                 label = static_cast<int>(val);
             } else {
                 row.push_back(val);
             }
             colIndex++;
         }
 
         dataList.push_back(row);
         labelList.push_back(label);
     }
 
     numSamples = dataList.size();
     numFeatures = dataList[0].size();
 
     // Flatten the data
     X.resize(numSamples * numFeatures);
     y.resize(numSamples);
 
     for (int i = 0; i < numSamples; ++i) {
         for (int j = 0; j < numFeatures; ++j) {
             X[i * numFeatures + j] = dataList[i][j];
         }
         y[i] = labelList[i];
     }
 }
 
 int main(int argc, char* argv[]) {
     MPI_Init(&argc, &argv);
 
     int world_size, rank;
     MPI_Comm_size(MPI_COMM_WORLD, &world_size);
     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
 
     // Check if we have the required number of processes
     if (world_size != 3) {
         if (rank == 0) {
             cerr << "Error: This program requires exactly 3 MPI processes." << endl;
             cerr << "Please run with: mpirun -np 3 " << argv[0] << " processed_data.csv" << endl;
         }
         MPI_Finalize();
         return 1;
     }
 
     // Check if filename is provided
     if (argc != 2) {
         if (rank == 0) {
             cerr << "Usage: " << argv[0] << " <data_file.csv>" << endl;
         }
         MPI_Finalize();
         return 1;
     }
 
     string filename = argv[1];
     vector<float> X;
     vector<int> y;
     int numSamples = 0;
     int numFeatures = 0;
 
     // Rank 0 loads the data
     if (rank == 0) {
         cout << "Loading dataset from " << filename << "..." << endl;
         loadData(filename, X, y, numSamples, numFeatures);
         cout << "Dataset loaded with " << numSamples << " samples and " 
              << numFeatures << " features." << endl;
     }
 
     // Broadcast the metadata to all ranks
     MPI_Bcast(&numSamples, 1, MPI_INT, 0, MPI_COMM_WORLD);
     MPI_Bcast(&numFeatures, 1, MPI_INT, 0, MPI_COMM_WORLD);
 
     // Calculate data distribution
     vector<int> rows(world_size);
     vector<int> displs_rows(world_size);
     vector<int> counts_X(world_size);
     vector<int> displs_X(world_size);
     vector<int> counts_y(world_size);
     vector<int> displs_y(world_size);
 
     for (int i = 0; i < world_size; ++i) {
         rows[i] = numSamples / world_size + (i < numSamples % world_size ? 1 : 0);
     }
 
     displs_rows[0] = 0;
     for (int i = 1; i < world_size; ++i) {
         displs_rows[i] = displs_rows[i-1] + rows[i-1];
     }
 
     for (int i = 0; i < world_size; ++i) {
         counts_X[i] = rows[i] * numFeatures;
         displs_X[i] = displs_rows[i] * numFeatures;
         counts_y[i] = rows[i];
         displs_y[i] = displs_rows[i];
     }
 
     // Allocate local data
     vector<float> local_X(counts_X[rank]);
     vector<int> local_y(counts_y[rank]);
 
     // Scatter the data
     MPI_Scatterv(X.data(), counts_X.data(), displs_X.data(), MPI_FLOAT,
                  local_X.data(), counts_X[rank], MPI_FLOAT, 0, MPI_COMM_WORLD);
     MPI_Scatterv(y.data(), counts_y.data(), displs_y.data(), MPI_INT,
                  local_y.data(), counts_y[rank], MPI_INT, 0, MPI_COMM_WORLD);
 
     // Train the appropriate model based on rank
     double trainingTime = 0.0;
     auto startTime = chrono::high_resolution_clock::now();
 
     if (rank == 0) {
         // Random Forest
         cout << "Rank 0: Training Random Forest..." << endl;
         RandomForest rf(100, 10, 2, numFeatures);
         rf.train(local_X, local_y, rows[rank], numFeatures);
         rf.saveModel("random_forest_model.bin");
     } 
     else if (rank == 1) {
         // MLP (Neural Network)
         cout << "Rank 1: Training MLP Neural Network..." << endl;
         vector<int> hiddenLayers = {16, 8};
         MLP mlp(numFeatures, hiddenLayers, 2); // Assuming binary classification for now
         mlp.train(local_X, local_y, rows[rank], numFeatures, 100, 0.01);
         mlp.saveModel("mlp_model.bin");
     } 
     else if (rank == 2) {
         // Logistic Regression
         cout << "Rank 2: Training Logistic Regression..." << endl;
         LogisticRegression lr(numFeatures, 0.01, 100);
         lr.train(local_X, local_y, rows[rank], numFeatures);
         lr.saveModel("logistic_regression_model.bin");
     }
 
     auto endTime = chrono::high_resolution_clock::now();
     trainingTime = chrono::duration<double>(endTime - startTime).count();
 
     // Gather timing results
     vector<double> timings(world_size);
     MPI_Gather(&trainingTime, 1, MPI_DOUBLE, timings.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
 
     // Print results
     if (rank == 0) {
         cout << "==================================================" << endl;
         cout << "SUMMARY OF MODEL TRAINING:" << endl;
         cout << "--------------------------------------------------" << endl;
         cout << "Random Forest Training Time: " << timings[0] << " seconds" << endl;
         cout << "MLP Training Time: " << timings[1] << " seconds" << endl;
         cout << "Logistic Regression Training Time: " << timings[2] << " seconds" << endl;
         cout << "--------------------------------------------------" << endl;
         
         // Determine the fastest model
         int fastestIndex = min_element(timings.begin(), timings.end()) - timings.begin();
         string fastestModel;
         
         if (fastestIndex == 0) fastestModel = "Random Forest";
         else if (fastestIndex == 1) fastestModel = "MLP";
         else fastestModel = "Logistic Regression";
         
         cout << "Fastest model: " << fastestModel << " (" << timings[fastestIndex] << " seconds)" << endl;
         cout << "--------------------------------------------------" << endl;
         cout << "Generated model files:" << endl;
         cout << "1. random_forest_model.bin" << endl;
         cout << "2. mlp_model.bin" << endl;
         cout << "3. logistic_regression_model.bin" << endl;
         cout << "==================================================" << endl;
         
         // Print the required format
         cout << "Timings (RF, MLP, LR): " 
              << fixed << setprecision(1) 
              << timings[0] << "s, " 
              << timings[1] << "s, " 
              << timings[2] << "s" << endl;
     }
 
     MPI_Finalize();
     return 0;
 }