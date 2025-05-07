/**
 * prediction.cpp - Loan approval prediction using trained models
 * 
 * This file implements a command-line application that loads the trained
 * models (Random Forest, MLP, and Logistic Regression) and uses them to
 * make predictions on loan approval based on user input.
 */

 #include <iostream>
 #include <fstream>
 #include <vector>
 #include <string>
 #include <algorithm>
 #include <iomanip>
 #include "./include/random_forest.h"
 #include "./include/mlp.h"
 #include "./include/logistic_regression.h"
 
 using namespace std;
 
 // Function to normalize input features based on training data statistics
 vector<float> normalizeInput(float income, float credit_score, float loan_amount, float dti_ratio, int employment_status) {
     // These values should come from your training data statistics
     // For now, using reasonable ranges as example
     // const float income_mean = 70000.0f, income_std = 30000.0f;
     // const float credit_score_mean = 650.0f, credit_score_std = 100.0f;
     // const float loan_amount_mean = 150000.0f, loan_amount_std = 100000.0f;
     // const float dti_ratio_mean = 0.35f, dti_ratio_std = 0.15f;
        const float income_mean = 110377.55f;
        const float income_std = 51729.68f;
        
        const float credit_score_mean = 575.72f;
        const float credit_score_std = 159.23f;
        
        const float loan_amount_mean = 44356.15f;
        const float loan_amount_std = 34666.60f;
        
        const float dti_ratio_mean = 34.72f;
        const float dti_ratio_std = 32.32f;

     
     vector<float> normalized(5);
     normalized[0] = (income - income_mean) / income_std;
     normalized[1] = (credit_score - credit_score_mean) / credit_score_std;
     normalized[2] = (loan_amount - loan_amount_mean) / loan_amount_std;
     normalized[3] = (dti_ratio - dti_ratio_mean) / dti_ratio_std;
     normalized[4] = static_cast<float>(employment_status); // Binary feature, no normalization needed
     
     return normalized;
 }
 
 // Function to get user input
 vector<float> getUserInput() {
     float income, credit_score, loan_amount, dti_ratio;
     int employment_status;
 
     cout << "===== Loan Approval Prediction System =====" << endl;
     cout << "Enter Income: $";
     cin >> income;
 
     cout << "Enter Credit Score (300-850): ";
     cin >> credit_score;
 
     cout << "Enter Loan Amount: $";
     cin >> loan_amount;
 
     cout << "Enter Debt-to-Income Ratio (0.0-1.0): ";
     cin >> dti_ratio;
 
     cout << "Enter Employment Status (1 = Employed, 0 = Unemployed): ";
     cin >> employment_status;
     
     return normalizeInput(income, credit_score, loan_amount, dti_ratio, employment_status);
 }
 
 // Function to make ensemble prediction
 string makePrediction(const vector<float>& features) {
     // Initialize models with proper constructors matching the training configuration
     // From the logs, Random Forest used 5 trees, not 100
     RandomForest rf(5, 10, 5, 5);  // 5 trees based on training output
     
     // From the logs, MLP used architecture 5->16->8->2
     MLP mlp(5, {16, 8}, 2);  // Match the architecture shown in training logs
     
     // For Logistic Regression, we need to specify number of features
     LogisticRegression lr(5);  // 5 input features
     
     // Load models - wrap each in separate try/catch to handle failures gracefully
     bool rf_loaded = false, mlp_loaded = false, lr_loaded = false;
     
     try {
         // RandomForest requires prefix and number of trees
         rf.loadModel("random_forest_model.bin", 5);  // Match the number of trees (5)
         rf_loaded = true;
     } catch (const exception& e) {
         cerr << "Error loading Random Forest model: " << e.what() << endl;
     }
     
     try {
         mlp.loadModel("mlp_model.bin");
         mlp_loaded = true;
     } catch (const exception& e) {
         cerr << "Error loading MLP model: " << e.what() << endl;
     }
     
     try {
         lr.loadModel("logistic_regression_model.bin");
         lr_loaded = true;
     } catch (const exception& e) {
         cerr << "Error loading Logistic Regression model: " << e.what() << endl;
     }
     
     // Check if at least one model was loaded successfully
     if (!rf_loaded && !mlp_loaded && !lr_loaded) {
         cerr << "Fatal error: No models could be loaded. Make sure you have trained the models first." << endl;
         return "Unknown";
     }
     
     // Make predictions only with successfully loaded models
     int votes_approve = 0;
     int votes_total = 0;
     
     cout << "\n===== Model Predictions =====" << endl;
     
     // Random Forest prediction
     if (rf_loaded) {
         try {
             int rf_prediction = rf.predict(features);
             votes_approve += (rf_prediction == 1);
             votes_total++;
             cout << "Random Forest: " << (rf_prediction == 1 ? "Approved" : "Not Approved") << endl;
         } catch (const exception& e) {
             cerr << "Error during Random Forest prediction: " << e.what() << endl;
         }
     } else {
         cout << "Random Forest: Model not available" << endl;
     }
     
     // MLP prediction
     if (mlp_loaded) {
         try {
             vector<int> mlp_predictions = mlp.predict(features, 1, 5);
             if (!mlp_predictions.empty()) {
                 int mlp_prediction = mlp_predictions[0];
                 votes_approve += (mlp_prediction == 1);
                 votes_total++;
                 cout << "Neural Network: " << (mlp_prediction == 1 ? "Approved" : "Not Approved") << endl;
             } else {
                 cout << "Neural Network: Failed to produce prediction" << endl;
             }
         } catch (const exception& e) {
             cerr << "Error during MLP prediction: " << e.what() << endl;
         }
     } else {
         cout << "Neural Network: Model not available" << endl;
     }
     
     // Logistic Regression prediction
     if (lr_loaded) {
         try {
             vector<int> lr_predictions = lr.predict(features, 1, 5);
             if (!lr_predictions.empty()) {
                 int lr_prediction = lr_predictions[0];
                 votes_approve += (lr_prediction == 1);
                 votes_total++;
                 cout << "Logistic Regression: " << (lr_prediction == 1 ? "Approved" : "Not Approved") << endl;
             } else {
                 cout << "Logistic Regression: Failed to produce prediction" << endl;
             }
         } catch (const exception& e) {
             cerr << "Error during Logistic Regression prediction: " << e.what() << endl;
         }
     } else {
         cout << "Logistic Regression: Model not available" << endl;
     }
     
     // Handle case where no predictions could be made
     if (votes_total == 0) {
         return "Could not make predictions with available models";
     }
     
     // Ensemble voting logic
     float approval_ratio = static_cast<float>(votes_approve) / votes_total;
     
     if (approval_ratio > 0.5f) {
         return "Approved";
     } else if (approval_ratio == 0.5f) {
         return "Borderline - Additional Review Required";
     } else {
         return "Not Approved";
     }
 }
 
 // Calculate risk score based on model confidences
 float calculateRiskScore(const vector<float>& features, LogisticRegression& lr, MLP& mlp, bool lr_loaded, bool mlp_loaded) {
     float risk_score = 50.0f;  // Default middle score
     float weight_sum = 0.0f;
     
     // Get probability from Logistic Regression if loaded
     if (lr_loaded) {
         try {
             vector<float> lr_probs = lr.predictProbabilities(features, 1, 5);
             if (!lr_probs.empty()) {
                 float lr_prob = lr_probs[0];
                 risk_score += lr_prob * 50.0f;  // 50% weight
                 weight_sum += 0.5f;
             }
         } catch (const exception& e) {
             cerr << "Warning: Could not get probabilities from Logistic Regression: " << e.what() << endl;
         }
     }
     
     // For MLP, use the binary prediction as a proxy for confidence if loaded
     if (mlp_loaded) {
         try {
             vector<int> mlp_pred = mlp.predict(features, 1, 5);
             if (!mlp_pred.empty()) {
                 float mlp_conf = (mlp_pred[0] == 1) ? 0.8f : 0.2f;
                 risk_score += mlp_conf * 50.0f;  // 50% weight
                 weight_sum += 0.5f;
             }
         } catch (const exception& e) {
             cerr << "Warning: Could not get prediction from MLP: " << e.what() << endl;
         }
     }
     
     // Adjust score based on actual weights used
     if (weight_sum > 0.0f) {
         risk_score /= weight_sum * 2.0f;  // Normalize to 0-100 scale
     }
     
     return risk_score;
 }
 
 int main() {
     vector<float> features = getUserInput();
     string prediction = makePrediction(features);
     
     cout << "\n===== Final Decision =====" << endl;
     cout << "Loan Application Status: " << prediction << endl;
     
     // Create and load models for risk score
     bool lr_loaded = false, mlp_loaded = false;
     LogisticRegression lr(5);
     // Use the correct architecture from training logs
     MLP mlp(5, {16, 8}, 2);
     
     try {
         lr.loadModel("logistic_regression_model.bin");
         lr_loaded = true;
     } catch (const exception& e) {
         cerr << "Warning: Could not load Logistic Regression model for risk scoring: " << e.what() << endl;
     }
     
     try {
         mlp.loadModel("mlp_model.bin");
         mlp_loaded = true;
     } catch (const exception& e) {
         cerr << "Warning: Could not load MLP model for risk scoring: " << e.what() << endl;
     }
     
     // Calculate and display risk score only if at least one model is loaded
     if (lr_loaded || mlp_loaded) {
         try {
             float risk_score = calculateRiskScore(features, lr, mlp, lr_loaded, mlp_loaded);
             cout << "Risk Assessment Score: " << fixed << setprecision(1) << risk_score << "/100" << endl;
         } catch (const exception& e) {
             cerr << "Warning: Could not calculate risk score: " << e.what() << endl;
         }
     } else {
         cout << "Risk Assessment: Not available (models could not be loaded)" << endl;
     }
     
     return 0;
 }
