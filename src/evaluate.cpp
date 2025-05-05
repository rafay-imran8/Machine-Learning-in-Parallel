// evaluate.cpp
#include <iostream>
#include <fstream>
#include <cmath>
#include <cstring>
#include <omp.h>
#include "../include/evaluate.hpp"
#include "../include/common.hpp"

float sigmoid(float z) {
    return 1.0f / (1.0f + std::exp(-z));
}

int predict(const Model* model, const DataMatrix* data_matrix, int* predictions) {
    if (!model || !data_matrix || !predictions) {
        std::cerr << "Error: NULL pointer passed to predict()\n";
        return -1;
    }

    #pragma omp parallel for
    for (int i = 0; i < data_matrix->rows; ++i) {
        float z = model->bias;
        for (int j = 0; j < data_matrix->cols; ++j) {
            z += model->weights[j] * data_matrix->features[i * data_matrix->cols + j];
        }
        float probability = sigmoid(z);
        predictions[i] = (probability >= 0.5f) ? 1 : 0;
    }

    return 0;
}

float compute_accuracy(const int* actual, const int* predicted, int count) {
    if (!actual || !predicted || count <= 0) {
        std::cerr << "Error: Invalid parameters in compute_accuracy()\n";
        return 0.0f;
    }

    int correct = 0;
    #pragma omp parallel for reduction(+:correct)
    for (int i = 0; i < count; ++i) {
        if (actual[i] == predicted[i]) correct++;
    }

    return 100.0f * static_cast<float>(correct) / count;
}

void compute_confusion_matrix(const int* actual, const int* predicted, int count, int confusion_matrix[2][2]) {
    if (!actual || !predicted || !confusion_matrix) {
        std::cerr << "Error: NULL pointer passed to compute_confusion_matrix()\n";
        return;
    }

    std::memset(confusion_matrix, 0, 4 * sizeof(int));

    #pragma omp parallel
    {
        int local_cm[2][2] = {{0, 0}, {0, 0}};
        #pragma omp for
        for (int i = 0; i < count; ++i) {
            local_cm[actual[i]][predicted[i]]++;
        }
        #pragma omp critical
        {
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 2; ++j) {
                    confusion_matrix[i][j] += local_cm[i][j];
                }
            }
        }
    }
}

void compute_precision_recall_f1(const int confusion_matrix[2][2], float* precision, float* recall, float* f1_score) {
    int tn = confusion_matrix[0][0];
    int fp = confusion_matrix[0][1];
    int fn = confusion_matrix[1][0];
    int tp = confusion_matrix[1][1];

    *precision = (tp + fp > 0) ? static_cast<float>(tp) / (tp + fp) : 0.0f;
    *recall = (tp + fn > 0) ? static_cast<float>(tp) / (tp + fn) : 0.0f;

    if (*precision > 0.0f || *recall > 0.0f) {
        *f1_score = 2.0f * (*precision) * (*recall) / (*precision + *recall);
    } else {
        *f1_score = 0.0f;
    }
}

void print_confusion_matrix(const int confusion_matrix[2][2]) {
    std::cout << "             Predicted      \n";
    std::cout << "             Negative Positive\n";
    std::cout << "Actual Negative " << confusion_matrix[0][0] << "       " << confusion_matrix[0][1] << "\n";
    std::cout << "       Positive " << confusion_matrix[1][0] << "       " << confusion_matrix[1][1] << "\n";
}

void print_evaluation_metrics(const EvaluationMetrics* metrics) {
    std::cout << "===== Model Evaluation Results =====\n";
    std::cout << "Accuracy: " << metrics->accuracy << "%\n";
    std::cout << "Precision: " << metrics->precision << "\n";
    std::cout << "Recall: " << metrics->recall << "\n";
    std::cout << "F1 Score: " << metrics->f1_score << "\n";
    std::cout << "Evaluation Time: " << metrics->evaluation_time << " seconds\n\n";
    std::cout << "Confusion Matrix:\n";
    print_confusion_matrix(metrics->confusion_matrix);
}

int save_evaluation_metrics(const EvaluationMetrics* metrics, const char* filename) {
    std::ofstream file(filename);
    if (!file) {
        std::cerr << "Error opening file '" << filename << "' for writing\n";
        return -1;
    }

    file << "===== Model Evaluation Results =====\n";
    file << "Accuracy: " << metrics->accuracy << "%\n";
    file << "Precision: " << metrics->precision << "\n";
    file << "Recall: " << metrics->recall << "\n";
    file << "F1 Score: " << metrics->f1_score << "\n";
    file << "Evaluation Time: " << metrics->evaluation_time << " seconds\n\n";
    file << "Confusion Matrix:\n";
    file << "             Predicted      \n";
    file << "             Negative Positive\n";
    file << "Actual Negative " << metrics->confusion_matrix[0][0] << "       " << metrics->confusion_matrix[0][1] << "\n";
    file << "       Positive " << metrics->confusion_matrix[1][0] << "       " << metrics->confusion_matrix[1][1] << "\n";

    file.close();
    return 0;
}

int evaluate_model(const Model* model, const DataMatrix* test_data, EvaluationMetrics* metrics) {
    if (!model || !test_data || !metrics) {
        std::cerr << "Error: NULL pointer passed to evaluate_model()\n";
        return -1;
    }

    double start_time = omp_get_wtime();
    int* predictions = new(std::nothrow) int[test_data->rows];

    if (!predictions) {
        std::cerr << "Error: Memory allocation failed for predictions\n";
        return -1;
    }

    if (predict(model, test_data, predictions) != 0) {
        std::cerr << "Error: Failed to make predictions\n";
        delete[] predictions;
        return -1;
    }

    metrics->accuracy = compute_accuracy(test_data->labels, predictions, test_data->rows);
    compute_confusion_matrix(test_data->labels, predictions, test_data->rows, metrics->confusion_matrix);
    compute_precision_recall_f1(metrics->confusion_matrix, &metrics->precision, &metrics->recall, &metrics->f1_score);

    metrics->evaluation_time = omp_get_wtime() - start_time;

    delete[] predictions;
    return 0;
}
