/**
 * @file evaluation_demo.cpp
 * @brief Demo showing how to integrate the evaluation module with preprocessing and training (C++ version)
 */

 #include <iostream>
 #include <cstdlib>
 #include <ctime>
 #include <mpi.h>
 #include <omp.h>
 #include "../include/common.h"
 #include "../include/evaluate.h"
 
 // Use C linkage if common.h and evaluate.h are from C libraries
 extern "C" {
     DataMatrix* create_data_matrix(int rows, int cols);
     void free_data_matrix(DataMatrix* matrix);
     Model* create_model(int num_features);
     void free_model(Model* model);
     void init_evaluation_metrics(EvaluationMetrics* metrics);
     int evaluate_model(Model* model, DataMatrix* test_data, EvaluationMetrics* metrics);
     void print_evaluation_metrics(const EvaluationMetrics* metrics);
     void save_evaluation_metrics(const EvaluationMetrics* metrics, const char* filename);
 }
 
 // Function prototypes (assumed from other modules)
 DataMatrix* load_and_preprocess(const char* filepath);
 int split_data(DataMatrix* full_data, DataMatrix** train_data, DataMatrix** test_data, float test_ratio);
 int train_model(DataMatrix* train_data, Model** model);
 
 int main(int argc, char** argv) {
     MPI_Init(&argc, &argv);
 
     int rank, size;
     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
     MPI_Comm_size(MPI_COMM_WORLD, &size);
 
     if (rank == 0) {
         std::cout << "Starting ML pipeline with evaluation..." << std::endl;
 
         // 1. Preprocessing
         DataMatrix* full_data = load_and_preprocess("data/dataset.csv");
         if (!full_data) {
             std::cerr << "Error: Failed to load and preprocess data" << std::endl;
             MPI_Abort(MPI_COMM_WORLD, 1);
             return 1;
         }
 
         DataMatrix* train_data = nullptr;
         DataMatrix* test_data = nullptr;
         if (split_data(full_data, &train_data, &test_data, 0.2f) != 0) {
             std::cerr << "Error: Failed to split data" << std::endl;
             free_data_matrix(full_data);
             MPI_Abort(MPI_COMM_WORLD, 1);
             return 1;
         }
 
         // 2. Training
         Model* model = nullptr;
         if (train_model(train_data, &model) != 0) {
             std::cerr << "Error: Failed to train model" << std::endl;
             free_data_matrix(train_data);
             free_data_matrix(test_data);
             free_data_matrix(full_data);
             MPI_Abort(MPI_COMM_WORLD, 1);
             return 1;
         }
 
         // 3. Evaluation
         EvaluationMetrics metrics;
         init_evaluation_metrics(&metrics);
 
         std::cout << "Evaluating model..." << std::endl;
         if (evaluate_model(model, test_data, &metrics) != 0) {
             std::cerr << "Error: Failed to evaluate model" << std::endl;
             free_model(model);
             free_data_matrix(train_data);
             free_data_matrix(test_data);
             free_data_matrix(full_data);
             MPI_Abort(MPI_COMM_WORLD, 1);
             return 1;
         }
 
         print_evaluation_metrics(&metrics);
         save_evaluation_metrics(&metrics, "results/evaluation_results.txt");
 
         // Cleanup
         free_model(model);
         free_data_matrix(train_data);
         free_data_matrix(test_data);
         free_data_matrix(full_data);
 
         std::cout << "ML pipeline completed successfully." << std::endl;
     }
 
     MPI_Finalize();
     return 0;
 }
 
 // Dummy function definitions for demonstration
 
 DataMatrix* load_and_preprocess(const char* filepath) {
     std::cout << "Simulating data preprocessing (will be implemented by teammate A)..." << std::endl;
     return create_data_matrix(100, 10); // Dummy data
 }
 
 int split_data(DataMatrix* full_data, DataMatrix** train_data, DataMatrix** test_data, float test_ratio) {
     std::cout << "Simulating data splitting..." << std::endl;
     int test_size = static_cast<int>(full_data->rows * test_ratio);
     int train_size = full_data->rows - test_size;
 
     *train_data = create_data_matrix(train_size, full_data->cols);
     *test_data = create_data_matrix(test_size, full_data->cols);
 
     return 0;
 }
 
 int train_model(DataMatrix* train_data, Model** model) {
     std::cout << "Simulating model training (will be implemented by teammate B)..." << std::endl;
     *model = create_model(train_data->cols);
     if (!(*model)) return 1;
 
     srand(42);
     for (int i = 0; i < train_data->cols; ++i) {
         (*model)->weights[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
     }
     (*model)->bias = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
 
     return 0;
 }
 