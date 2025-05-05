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
 #include "random_forest.h"
 #include "mlp.h"
 #include "logistic_regression.h"
 
 using namespace std;
 
 // Function to normalize input features based on training data statistics
 vector<float> normalizeInput(float income, float credit_score, float loan_amount, float dti_ratio, int employment_status) {
     // These values should come from your training data statistics
     // For now, using reasonable ranges as example
     const float income_mean = 70000.0f, income_std = 30000.0f;
     const float credit_score_mean = 650.0f, credit_score_std = 100.0f;
     const float loan_amount_mean = 150000.0f, loan_amount_std = 100000.0f;
     const float dti_ratio_mean = 0.35f, dti_ratio_std = 0.15f;
     
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
     // Load models
     RandomForest rf;
     MLP mlp;
     LogisticRegression lr;
     
     bool rfLoaded = rf.loadModel("random_forest_model.bin");
     bool mlpLoaded = mlp.loadModel("mlp_model.bin");
     bool lrLoaded = lr.loadModel("logistic_regression_model.bin");
     
     if (!rfLoaded || !mlpLoaded || !lrLoaded) {
         cerr << "Error: Failed to load one or more models." << endl;
         cerr << "Make sure you have trained the models first using the hybrid_train application." << endl;
         return "Unknown";
     }
     
     // Make predictions
     int rf_prediction = rf.predict(features);
     int mlp_prediction = mlp.predict(features);
     int lr_prediction = lr.predict(features);
     
     // Ensemble voting (majority vote)
     int votes_approve = (rf_prediction == 1) + (mlp_prediction == 1) + (lr_prediction == 1);
     int votes_reject = 3 - votes_approve;
     
     cout << "\n===== Model Predictions =====" << endl;
     cout << "Random Forest: " << (rf_prediction == 1 ? "Approved" : "Not Approved") << endl;
     cout << "Neural Network: " << (mlp_prediction == 1 ? "Approved" : "Not Approved") << endl;
     cout << "Logistic Regression: " << (lr_prediction == 1 ? "Approved" : "Not Approved") << endl;
     
     if (votes_approve > votes_reject) {
         return "Approved";
     } else if (votes_approve == votes_reject) {
         // In case of tie, could use confidence scores or prefer caution
         return "Borderline - Additional Review Required";
     } else {
         return "Not Approved";
     }
 }
 
 // Calculate risk score based on model confidences
 float calculateRiskScore(const vector<float>& features, LogisticRegression& lr, MLP& mlp) {
     // Get probability from Logistic Regression (assumed to return probability)
     float lr_prob = lr.predict_proba(features);
     
     // Get confidence from MLP (assumed to return confidence)
     float mlp_conf = mlp.predict_confidence(features);
     
     // Combine into risk score (0-100, higher is safer)
     float risk_score = (lr_prob * 0.5 + mlp_conf * 0.5) * 100.0f;
     return risk_score;
 }
 
 int main() {
     vector<float> features = getUserInput();
     string prediction = makePrediction(features);
     
     cout << "\n===== Final Decision =====" << endl;
     cout << "Loan Application Status: " << prediction << endl;
     
     // Load models for risk score (if needed)
     LogisticRegression lr;
     MLP mlp;
     if (lr.loadModel("logistic_regression_model.bin") && mlp.loadModel("mlp_model.bin")) {
         float risk_score = calculateRiskScore(features, lr, mlp);
         cout << "Risk Assessment Score: " << fixed << setprecision(1) << risk_score << "/100" << endl;
     }
     
     return 0;
 }