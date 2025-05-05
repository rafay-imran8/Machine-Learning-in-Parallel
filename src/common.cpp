
#include <iostream>
#include <cstring>
#include "../include/common.hpp"

DataMatrix::DataMatrix(int rows, int cols) : rows(rows), cols(cols) {
    if (rows <= 0 || cols <= 0) {
        throw std::invalid_argument("Invalid dimensions for DataMatrix");
    }

    features = new float[rows * cols]();
    labels = new int[rows]();

    if (!features || !labels) {
        delete[] features;
        delete[] labels;
        throw std::runtime_error("Failed to allocate memory for DataMatrix");
    }
}

DataMatrix::~DataMatrix() {
    delete[] features;
    delete[] labels;
}

Model::Model(int feature_count) : feature_count(feature_count), bias(0.0f) {
    if (feature_count <= 0) {
        throw std::invalid_argument("Invalid feature count for Model");
    }

    weights = new float[feature_count]();
    if (!weights) {
        throw std::runtime_error("Failed to allocate memory for weights");
    }
}

Model::~Model() {
    delete[] weights;
}

EvaluationMetrics::EvaluationMetrics()
    : accuracy(0.0f), precision(0.0f), recall(0.0f), f1_score(0.0f), evaluation_time(0.0) {
    std::memset(confusion_matrix, 0, sizeof(confusion_matrix));
}
