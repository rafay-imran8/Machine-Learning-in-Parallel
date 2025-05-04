#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <fstream>
#include <sstream>
#include <iostream>
#include <omp.h>  // OpenMP header

using namespace cv;
using namespace cv::ml;

int main() {
    std::ifstream file("processed_data.csv");
    std::string line;
    std::vector<std::vector<float>> dataList;
    std::vector<int> labelList;

    std::getline(file, line);

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        std::vector<float> row;
        int colIndex = 0;
        int label;

        while (std::getline(ss, value, ',')) {
            float val = std::stof(value);
            if (colIndex == 5)
                label = static_cast<int>(val);
            else
                row.push_back(val);
            colIndex++;
        }

        dataList.push_back(row);
        labelList.push_back(label);
    }

    int rows = dataList.size();
    int cols = dataList[0].size();

    Mat trainingDataMat(rows, cols, CV_32F);
    Mat labelsMat(rows, 1, CV_32S);

    #pragma omp parallel for
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            trainingDataMat.at<float>(i, j) = dataList[i][j];
        }
        labelsMat.at<int>(i, 0) = labelList[i];
    }

    std::cout << "Training kNN model using " << omp_get_max_threads() << " threads..." << std::endl;

    Ptr<KNearest> knn = KNearest::create();
    knn->train(trainingDataMat, ROW_SAMPLE, labelsMat);

    knn->save("trained_knn.yml");

    std::cout << "Model training completed and saved to 'trained_knn.yml'." << std::endl;

    return 0;
}
