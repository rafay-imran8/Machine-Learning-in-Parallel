// loan_data_preprocessor.h
// Modern C++17 loan data preprocessing with OpenMP

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <unordered_map>
// #include <optional>
#include <omp.h>

using namespace std;

// Forward declarations
namespace csv
{
    class CSVReader; // From fast-cpp-csv-parser
}

namespace loan_preprocessing
{

    // Profiling utility
    struct ProfileMetric
    {
        string stage_name;
        double start_time;
        double end_time;
        int thread_id;
        int thread_count;

        ProfileMetric(const string &name);
        void end();
    };

    // Loan record structure - modern C++ version of the original C struct
    struct LoanRecord
    {
        string text;              // Optional loan description text
        double income{0.0};       // Annual income
        int credit_score{0};      // Credit score
        double loan_amount{0.0};  // Requested loan amount
        double dti_ratio{0.0};    // Debt-to-income ratio
        int employment_status{0}; // Encoded: 0 = unemployed, 1 = employed
        int approval{0};          // Encoded: 0 = rejected, 1 = approved

        // Additional fields for column statistics
        static vector<double> column_means;
        static vector<double> column_stddevs;
    };

    // Dataset container class
    class Dataset
    {
    public:
        Dataset(bool include_text = false);
        ~Dataset() = default; // RAII handles cleanup

        // No copies, only moves
        Dataset(const Dataset &) = delete;
        Dataset &operator=(const Dataset &) = delete;
        Dataset(Dataset &&) = default;
        Dataset &operator=(Dataset &&) = default;

        // Main processing functions
        bool load_from_file(const string &filename);
        void preprocess();
        void save_to_file(const string &filename) const;
        void print_sample(int sample_size) const;
        void export_profiling_data(const string &filename) const;

    private:
        // Processing pipeline stages
        void calculate_statistics();
        void encode_categorical_variables();
        void impute_missing_values();
        void normalize_numerical_features();

        // Helper methods
        void encode_categorical_vars(LoanRecord &record, const string &employment, const string &approval);
        bool is_missing_value(const string &value) const;

        // Data members
        vector<LoanRecord> records;
        bool include_text;
        vector<ProfileMetric> profile_data;

        // Column name mappings for categorical variables
        unordered_map<string, int> employment_status_map;
        unordered_map<string, int> approval_map;
    };

    // Factory function to create and process dataset
    unique_ptr<Dataset> load_and_preprocess(const string &filename, bool include_text = false);

} // namespace loan_preprocessing