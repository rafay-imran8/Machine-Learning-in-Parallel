/**
 * @file evaluate.h
 * @brief Header file for logistic regression model evaluation functions
 */

 #ifndef EVALUATE_H
 #define EVALUATE_H
 
 #include "common.h"
 
 /**
  * @brief Evaluates a trained logistic regression model on test data
  *
  * @param model The trained model containing weights
  * @param test_data Test data matrix
  * @param metrics Output struct to store evaluation metrics
  * @return int 0 on success, non-zero on failure
  */
 int evaluate_model(const Model* model, const DataMatrix* test_data, EvaluationMetrics* metrics);
 
 /**
  * @brief Predicts class labels using the trained model
  * 
  * @param model The trained model
  * @param data_matrix Input data for prediction
  * @param predictions Output array for predicted labels (must be pre-allocated)
  * @return int 0 on success, non-zero on failure
  */
 int predict(const Model* model, const DataMatrix* data_matrix, int* predictions);
 
 /**
  * @brief Calculates the sigmoid of the given value
  * 
  * @param z Input value
  * @return float Sigmoid value between 0 and 1
  */
 float sigmoid(float z);
 
 /**
  * @brief Computes accuracy of predictions
  * 
  * @param actual Actual labels
  * @param predicted Predicted labels
  * @param count Number of examples
  * @return float Accuracy as a percentage (0-100)
  */
 float compute_accuracy(const int* actual, const int* predicted, int count);
 
 /**
  * @brief Computes confusion matrix for binary classification
  * 
  * @param actual Actual labels
  * @param predicted Predicted labels
  * @param count Number of examples
  * @param confusion_matrix Output 2x2 confusion matrix (must be preallocated)
  */
 void compute_confusion_matrix(const int* actual, const int* predicted, int count, int confusion_matrix[2][2]);
 
 /**
  * @brief Computes precision, recall, and F1 score from confusion matrix
  * 
  * @param confusion_matrix 2x2 confusion matrix
  * @param precision Output precision value
  * @param recall Output recall value
  * @param f1_score Output F1 score
  */
 void compute_precision_recall_f1(const int confusion_matrix[2][2], 
                                 float* precision, 
                                 float* recall, 
                                 float* f1_score);
 
 /**
  * @brief Prints evaluation metrics to stdout
  * 
  * @param metrics The metrics to print
  */
 void print_evaluation_metrics(const EvaluationMetrics* metrics);
 
 /**
  * @brief Prints confusion matrix to stdout
  * 
  * @param confusion_matrix 2x2 confusion matrix
  */
 void print_confusion_matrix(const int confusion_matrix[2][2]);
 
 /**
  * @brief Saves evaluation metrics to a file
  * 
  * @param metrics The metrics to save
  * @param filename Output filename
  * @return int 0 on success, non-zero on failure
  */
 int save_evaluation_metrics(const EvaluationMetrics* metrics, const char* filename);
 
 #endif /* EVALUATE_H */