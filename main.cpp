// // main.cpp
// // Command-line interface for the loan data preprocessor library

// #include "loan_data_preprocessor.h"
// #include <iostream>
// #include <string>
// #include <chrono>
// #include <iomanip>

// using namespace std;
// using namespace loan_preprocessing;

// void print_usage() {
//     cout << "Loan Data Preprocessor" << endl;
//     cout << "Usage: loan_preprocessor [options] <input_file> <output_file>" << endl;
//     cout << endl;
//     cout << "Options:" << endl;
//     cout << "  --help             Display this help message" << endl;
//     cout << "  --include-text     Include text descriptions in the output" << endl;
//     cout << "  --sample <n>       Display a sample of n records after processing" << endl;
//     cout << "  --profile <file>   Export profiling data to the specified file" << endl;
//     cout << endl;
// }

// int main(int argc, char* argv[]) {
//     // Default parameter values
//     string input_file;
//     string output_file;
//     bool include_text = false;
//     int sample_size = 0;
//     string profile_file;
    
//     // Parse command line arguments
//     for (int i = 1; i < argc; i++) {
//         string arg = argv[i];
        
//         if (arg == "--help") {
//             print_usage();
//             return 0;
//         } else if (arg == "--include-text") {
//             include_text = true;
//         } else if (arg == "--sample") {
//             if (i + 1 < argc) {
//                 try {
//                     sample_size = stoi(argv[++i]);
//                     if (sample_size <= 0) {
//                         cerr << "Error: Sample size must be a positive integer." << endl;
//                         return 1;
//                     }
//                 } catch (const exception& e) {
//                     cerr << "Error: Invalid sample size." << endl;
//                     return 1;
//                 }
//             } else {
//                 cerr << "Error: --sample requires a numeric argument." << endl;
//                 return 1;
//             }
//         } else if (arg == "--profile") {
//             if (i + 1 < argc) {
//                 profile_file = argv[++i];
//             } else {
//                 cerr << "Error: --profile requires a filename argument." << endl;
//                 return 1;
//             }
//         } else if (input_file.empty()) {
//             input_file = arg;
//         } else if (output_file.empty()) {
//             output_file = arg;
//         } else {
//             cerr << "Error: Unexpected argument: " << arg << endl;
//             print_usage();
//             return 1;
//         }
//     }
    
//     // Check if required arguments are provided
//     if (input_file.empty() || output_file.empty()) {
//         cerr << "Error: Both input and output files must be specified." << endl;
//         print_usage();
//         return 1;
//     }
    
//     try {
//         // Record the start time for total execution
//         auto start_time = chrono::high_resolution_clock::now();
        
//         cout << "Processing loan data..." << endl;
//         cout << "Input file: " << input_file << endl;
//         cout << "Output file: " << output_file << endl;
//         cout << "Include text: " << (include_text ? "Yes" : "No") << endl;
        
//         // Set the number of OpenMP threads based on hardware
//         int num_threads = omp_get_max_threads();
//         cout << "Using " << num_threads << " threads for processing." << endl;
        
//         // Load and preprocess the data
//         unique_ptr<Dataset> dataset;
//         try {
//             dataset = load_and_preprocess(input_file, include_text);
//         } catch (const exception& e) {
//             cerr << "Error: " << e.what() << endl;
//             return 1;
//         }
        
//         // Display sample if requested
//         if (sample_size > 0) {
//             dataset->print_sample(sample_size);
//         }

        
//         // Save the preprocessed data
//         dataset->save_to_file(output_file);
        
//         // Export profiling data if requested
//         // if (!profile_file.empty()) {
//         //     dataset->export_profiling_data(profile_file);
//         // }



// // Show a sample of the preprocessed data with numeric values
// std::cout << "\nPREPROCESSED DATA SAMPLE (NUMERIC):" << std::endl;
// dataset->print_preprocessed_sample(sample_size);


// // Save the preprocessed data
// dataset->save_to_file("preprocessed_data.csv");





        
//         // Calculate and display total execution time
//         auto end_time = chrono::high_resolution_clock::now();
//         chrono::duration<double> elapsed = end_time - start_time;
        
//         cout << "\nPreprocessing completed successfully!" << endl;
//         cout << "Total execution time: " << fixed << setprecision(2) 
//              << elapsed.count() << " seconds" << endl;
        
//         return 0;
//     } catch (const exception& e) {
//         cerr << "Fatal error: " << e.what() << endl;
//         return 1;
//     }
// }

// main.cpp
// Command-line interface for the loan data preprocessor library

#include "include/loan_data_preprocessor.h"
#include <iostream>
#include <string>
#include <chrono>
#include <iomanip>

using namespace std;
using namespace loan_preprocessing;

void print_usage() {
    cout << "Loan Data Preprocessor" << endl;
    cout << "Usage: loan_preprocessor [options] <input_file> <output_file>" << endl;
    cout << endl;
    cout << "Options:" << endl;
    cout << "  --help             Display this help message" << endl;
    cout << "  --sample <n>       Display a sample of n records after processing" << endl;
    cout << "  --profile <file>   Export profiling data to the specified file" << endl;
    cout << endl;
}

int main(int argc, char* argv[]) {
    // Default parameter values
    string input_file;
    string output_file;
    int sample_size = 0;
    string profile_file;
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        
        if (arg == "--help") {
            print_usage();
            return 0;
        } else if (arg == "--sample") {
            if (i + 1 < argc) {
                try {
                    sample_size = stoi(argv[++i]);
                    if (sample_size <= 0) {
                        cerr << "Error: Sample size must be a positive integer." << endl;
                        return 1;
                    }
                } catch (const exception& e) {
                    cerr << "Error: Invalid sample size." << endl;
                    return 1;
                }
            } else {
                cerr << "Error: --sample requires a numeric argument." << endl;
                return 1;
            }
        } else if (arg == "--profile") {
            if (i + 1 < argc) {
                profile_file = argv[++i];
            } else {
                cerr << "Error: --profile requires a filename argument." << endl;
                return 1;
            }
        } else if (input_file.empty()) {
            input_file = arg;
        } else if (output_file.empty()) {
            output_file = arg;
        } else {
            cerr << "Error: Unexpected argument: " << arg << endl;
            print_usage();
            return 1;
        }
    }
    
    // Check if required arguments are provided
    if (input_file.empty() || output_file.empty()) {
        cerr << "Error: Both input and output files must be specified." << endl;
        print_usage();
        return 1;
    }
    
    try {
        // Record the start time for total execution
        auto start_time = chrono::high_resolution_clock::now();
        
        cout << "Processing loan data..." << endl;
        cout << "Input file: " << input_file << endl;
        cout << "Output file: " << output_file << endl;
        
        // Set the number of OpenMP threads based on hardware
        int num_threads = omp_get_max_threads();
        cout << "Using " << num_threads << " threads for processing." << endl;
        
        // Load and preprocess the data
        unique_ptr<Dataset> dataset;
        try {
            dataset = load_and_preprocess(input_file);
        } catch (const exception& e) {
            cerr << "Error: " << e.what() << endl;
            return 1;
        }
        
        // Display sample if requested
        if (sample_size > 0) {
            dataset->print_sample(sample_size);
        }

        // Save the preprocessed data
        dataset->save_to_file(output_file);
        
        // Export profiling data if requested
        if (!profile_file.empty()) {
            dataset->export_profiling_data(profile_file);
        }

        // Show a sample of the preprocessed data with numeric values
        if (sample_size > 0) {
            std::cout << "\nPREPROCESSED DATA SAMPLE (NUMERIC):" << std::endl;
            dataset->print_preprocessed_sample(sample_size);
        }
        
        // Calculate and display total execution time
        auto end_time = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed = end_time - start_time;
        
        cout << "\nPreprocessing completed successfully!" << endl;
        cout << "Total execution time: " << fixed << setprecision(2) 
             << elapsed.count() << " seconds" << endl;
        
        return 0;
    } catch (const exception& e) {
        cerr << "Fatal error: " << e.what() << endl;
        return 1;
    }
}