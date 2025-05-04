#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <fstream>
#include <sstream>
#include <iostream>
#include <thread>
#include <chrono>
#include <mutex>
#include <future>

using namespace cv;
using namespace cv::ml;
using namespace std;

// Global mutex for thread-safe console output
mutex cout_mutex;

// Struct to hold our dataset
struct DataSet {
    Mat trainingData;
    Mat labels;
};

// Function to load and preprocess the data
DataSet loadData(const string& filename) {
    ifstream file(filename);
    string line;
    vector<vector<float>> dataList;
    vector<int> labelList;

    // Skip the header
    getline(file, line);

    while (getline(file, line)) {
        stringstream ss(line);
        string value;
        vector<float> row;
        int colIndex = 0;
        int label;

        while (getline(ss, value, ',')) {
            float val = stof(value);
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

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            trainingDataMat.at<float>(i, j) = dataList[i][j];
        }
        labelsMat.at<int>(i, 0) = labelList[i];
    }

    DataSet dataset;
    dataset.trainingData = trainingDataMat;
    dataset.labels = labelsMat;
    
    return dataset;
}

// Function to train a K-Nearest Neighbors model, returns training time in seconds
double trainKNN(const DataSet& dataset) {
    // Record start time
    auto start = chrono::high_resolution_clock::now();
    
    {
        lock_guard<mutex> lock(cout_mutex);
        cout << "Starting KNN training at " 
             << chrono::system_clock::to_time_t(chrono::system_clock::now())
             << " ..." << endl;
    }
    
    Ptr<KNearest> knn = KNearest::create();
    knn->setDefaultK(5);
    knn->setIsClassifier(true);
    knn->train(dataset.trainingData, ROW_SAMPLE, dataset.labels);
    
    // Save the model
    knn->save("trained_knn_model.yml");
    
    // Record end time
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;
    
    {
        lock_guard<mutex> lock(cout_mutex);
        cout << "-------------------------------------------" << endl;
        cout << "KNN MODEL TRAINING RESULTS:" << endl;
        cout << "   Start time: " << chrono::system_clock::to_time_t(chrono::system_clock::now() - elapsed) << endl;
        cout << "   End time: " << chrono::system_clock::to_time_t(chrono::system_clock::now()) << endl;
        cout << "   Total training time: " << elapsed.count() << " seconds" << endl;
        cout << "   Model saved to: 'trained_knn_model.yml'" << endl;
        cout << "-------------------------------------------" << endl;
    }
    
    return elapsed.count();
}

// Function to train a Support Vector Machine model, returns training time in seconds
double trainSVM(const DataSet& dataset) {
    // Record start time
    auto start = chrono::high_resolution_clock::now();
    
    {
        lock_guard<mutex> lock(cout_mutex);
        cout << "Starting SVM training at " 
             << chrono::system_clock::to_time_t(chrono::system_clock::now())
             << " ..." << endl;
    }
    
    Ptr<SVM> svm = SVM::create();
    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::RBF);
    svm->setGamma(0.1);
    svm->setC(1.0);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
    svm->train(dataset.trainingData, ROW_SAMPLE, dataset.labels);
    
    // Save the model
    svm->save("trained_svm_model.yml");
    
    // Record end time
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;
    
    {
        lock_guard<mutex> lock(cout_mutex);
        cout << "-------------------------------------------" << endl;
        cout << "SVM MODEL TRAINING RESULTS:" << endl;
        cout << "   Start time: " << chrono::system_clock::to_time_t(chrono::system_clock::now() - elapsed) << endl;
        cout << "   End time: " << chrono::system_clock::to_time_t(chrono::system_clock::now()) << endl;
        cout << "   Total training time: " << elapsed.count() << " seconds" << endl;
        cout << "   Model saved to: 'trained_svm_model.yml'" << endl;
        cout << "-------------------------------------------" << endl;
    }
    
    return elapsed.count();
}

// Function to train a Random Forest model, returns training time in seconds
double trainRandomForest(const DataSet& dataset) {
    // Record start time
    auto start = chrono::high_resolution_clock::now();
    
    {
        lock_guard<mutex> lock(cout_mutex);
        cout << "Starting Random Forest training at " 
             << chrono::system_clock::to_time_t(chrono::system_clock::now())
             << " ..." << endl;
    }
    
    Ptr<RTrees> forest = RTrees::create();
    forest->setActiveVarCount(4);
    forest->setMaxDepth(10);
    forest->setMinSampleCount(2);
    forest->setMaxCategories(10);
    forest->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 0.01));
    forest->train(dataset.trainingData, ROW_SAMPLE, dataset.labels);
    
    // Save the model
    forest->save("trained_random_forest_model.yml");
    
    // Record end time
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;
    
    {
        lock_guard<mutex> lock(cout_mutex);
        cout << "-------------------------------------------" << endl;
        cout << "RANDOM FOREST MODEL TRAINING RESULTS:" << endl;
        cout << "   Start time: " << chrono::system_clock::to_time_t(chrono::system_clock::now() - elapsed) << endl;
        cout << "   End time: " << chrono::system_clock::to_time_t(chrono::system_clock::now()) << endl;
        cout << "   Total training time: " << elapsed.count() << " seconds" << endl;
        cout << "   Model saved to: 'trained_random_forest_model.yml'" << endl;
        cout << "-------------------------------------------" << endl;
    }
    
    return elapsed.count();
}

int main() {
    auto totalStart = chrono::high_resolution_clock::now();
    
    // Load data
    cout << "Loading dataset..." << endl;
    DataSet dataset = loadData("processed_data.csv");
    cout << "Dataset loaded with " << dataset.trainingData.rows << " samples and " 
         << dataset.trainingData.cols << " features." << endl;
    
    // Create futures to store the returned times
    future<double> knnFuture = async(launch::async, trainKNN, ref(dataset));
    future<double> svmFuture = async(launch::async, trainSVM, ref(dataset));
    future<double> rfFuture = async(launch::async, trainRandomForest, ref(dataset));
    
    // Wait for all futures and get the results
    double knnTime = knnFuture.get();
    double svmTime = svmFuture.get();
    double rfTime = rfFuture.get();
    
    auto totalEnd = chrono::high_resolution_clock::now();
    chrono::duration<double> totalElapsed = totalEnd - totalStart;
    
    cout << "==================================================" << endl;
    cout << "SUMMARY OF MODEL TRAINING:" << endl;
    cout << "--------------------------------------------------" << endl;
    cout << "KNN Training Time: " << knnTime << " seconds" << endl;
    cout << "SVM Training Time: " << svmTime << " seconds" << endl;
    cout << "Random Forest Training Time: " << rfTime << " seconds" << endl;
    cout << "--------------------------------------------------" << endl;
    
    // Determine the fastest model
    string fastestModel;
    double fastestTime = min({knnTime, svmTime, rfTime});
    
    if (fastestTime == knnTime) fastestModel = "KNN";
    else if (fastestTime == svmTime) fastestModel = "SVM";
    else fastestModel = "Random Forest";
    
    cout << "Fastest model: " << fastestModel << " (" << fastestTime << " seconds)" << endl;
    cout << "--------------------------------------------------" << endl;
    cout << "Total wall clock time for all parallel training: " << totalElapsed.count() << " seconds" << endl;
    cout << "--------------------------------------------------" << endl;
    cout << "Generated model files:" << endl;
    cout << "1. trained_knn_model.yml" << endl;
    cout << "2. trained_svm_model.yml" << endl;
    cout << "3. trained_random_forest_model.yml" << endl;
    cout << "==================================================" << endl;
    
    return 0;
}