/**
 * @file common.h
 * @brief Common data structures shared across preprocessing, training, and evaluation
 */

 #ifndef COMMON_H
 #define COMMON_H
 
 /**
  * @brief Structure to hold dataset features and labels
  */
 typedef struct {
     float* features;    // Features matrix (rows × cols)
     int* labels;        // Labels array (rows)
     int rows;           // Number of data points
     int cols;           // Number of features
 } DataMatrix;
 
 /**
  * @brief Structure to hold logistic regression model parameters
  */
 typedef struct {
     float* weights;     // Weight vector (cols)
     float bias;         // Bias term
     int feature_count;  // Number of features
 } Model;
 
 /**
  * @brief Structure to hold evaluation metrics
  */
 typedef struct {
     float accuracy;             // Overall accuracy (%)
     int confusion_matrix[2][2]; // Confusion matrix for binary classification
     float precision;            // Precision score
     float recall;               // Recall score
     float f1_score;             // F1 score
     double evaluation_time;     // Time taken for evaluation (seconds)
 } EvaluationMetrics;
 
 /**
  * @brief Creates a new DataMatrix with allocated memory
  * 
  * @param rows Number of data points
  * @param cols Number of features
  * @return DataMatrix* Pointer to new DataMatrix, or NULL on failure
  */
 DataMatrix* create_data_matrix(int rows, int cols);
 
 /**
  * @brief Frees memory allocated for a DataMatrix
  * 
  * @param matrix Pointer to DataMatrix to free
  */
 void free_data_matrix(DataMatrix* matrix);
 
 /**
  * @brief Creates a new Model with allocated memory
  * 
  * @param feature_count Number of features
  * @return Model* Pointer to new Model, or NULL on failure
  */
 Model* create_model(int feature_count);
 
 /**
  * @brief Frees memory allocated for a Model
  * 
  * @param model Pointer to Model to free
  */
 void free_model(Model* model);
 
 /**
  * @brief Initializes the evaluation metrics structure
  * 
  * @param metrics Pointer to metrics structure to initialize
  */
 void init_evaluation_metrics(EvaluationMetrics* metrics);
 
 #endif /* COMMON_H */