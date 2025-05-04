#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <iostream>

using namespace cv;
using namespace cv::ml;

int main() {
    Ptr<KNearest> knn = Algorithm::load<KNearest>("trained_knn.yml");
    if (knn.empty()) {
        std::cerr << "Failed to load the model!" << std::endl;
        return -1;
    }

    float income, credit_score, loan_amount, dti_ratio;
    int employment_status;

    std::cout << "Enter Income: ";
    std::cin >> income;

    std::cout << "Enter Credit Score: ";
    std::cin >> credit_score;

    std::cout << "Enter Loan Amount: ";
    std::cin >> loan_amount;

    std::cout << "Enter DTI Ratio: ";
    std::cin >> dti_ratio;

    std::cout << "Enter Employment Status (1 = Employed, 0 = Unemployed): ";
    std::cin >> employment_status;

    // Prepare input
    Mat sample = (Mat_<float>(1, 5) << income, credit_score, loan_amount, dti_ratio, employment_status);

    // Predict
    float result = knn->predict(sample);

    std::cout << "Predicted Approval: " << (result == 1 ? "Approved" : "Not Approved") << std::endl;

    return 0;
}
