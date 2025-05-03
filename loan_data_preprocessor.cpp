// loan_data_preprocessor.cpp
// Implementation of the loan data preprocessing library

#include "loan_data_preprocessor.h"
#include "fast-cpp-csv-parser/csv.h" // Include fast-cpp-csv-parser

#include <iostream>
#include <fstream>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <iomanip>
#include <sstream>

namespace loan_preprocessing
{

    // Initialize static members
    vector<double> LoanRecord::column_means;
    vector<double> LoanRecord::column_stddevs;

    // ProfileMetric implementation
    ProfileMetric::ProfileMetric(const string &name) : stage_name(name),
                                                       start_time(omp_get_wtime()),
                                                       thread_id(omp_get_thread_num()),
                                                       thread_count(omp_get_num_threads()) {}

    void ProfileMetric::end()
    {
        end_time = omp_get_wtime();
    }

    // Dataset implementation
    Dataset::Dataset(bool include_text) : include_text(include_text)
    {
        // Initialize categorical mappings
        employment_status_map = {{"unemployed", 0}, {"employed", 1}};
        approval_map = {{"Rejected", 0}, {"Approved", 1}};
    }

    bool Dataset::load_from_file(const string &filename)
    {
        ProfileMetric metric("load_file");

        try
        {
            // Create CSVReader instance with specifications
            csv::CSVReader reader(filename);

            // Reserve space based on an estimate (we'll read the file in chunks)
            const size_t INITIAL_CAPACITY = 100000; // Adjust based on expected size
            records.reserve(INITIAL_CAPACITY);

            // Define callbacks for error handling
            reader.set_skip_eof(true);

            // Define column indices for mapping
            int idx_text = -1;
            int idx_income = -1;
            int idx_credit_score = -1;
            int idx_loan_amount = -1;
            int idx_dti_ratio = -1;
            int idx_employment = -1;
            int idx_approval = -1;

            // Get header and determine column indices
            auto header = reader.get_header();
            for (size_t i = 0; i < header.size(); i++)
            {
                const auto &col_name = header[i];
                if (col_name == "Text")
                    idx_text = i;
                else if (col_name == "Income")
                    idx_income = i;
                else if (col_name == "Credit_Score")
                    idx_credit_score = i;
                else if (col_name == "Loan_Amount")
                    idx_loan_amount = i;
                else if (col_name == "DTI_Ratio")
                    idx_dti_ratio = i;
                else if (col_name == "Employment_Status")
                    idx_employment = i;
                else if (col_name == "Approval")
                    idx_approval = i;
            }

            // Validate that required columns exist
            if (idx_income == -1 || idx_credit_score == -1 || idx_loan_amount == -1 ||
                idx_dti_ratio == -1 || idx_employment == -1 || idx_approval == -1)
            {
                throw runtime_error("Missing required columns in input CSV");
            }

            // Check if Text column should be included but is missing
            if (include_text && idx_text == -1)
            {
                cerr << "Warning: Text column requested but not found in CSV" << endl;
                include_text = false;
            }

            // Process rows in parallel using OpenMP
            // First, read all rows into memory
            vector<vector<string>> raw_rows;
            string row_str;

            // Count rows for better allocation
            size_t row_count = 0;
            for (auto &row : reader)
            {
                row_count++;
            }
            raw_rows.reserve(row_count);

            // Reopen the file to read the data
            csv::CSVReader data_reader(filename);
            // Skip header
            data_reader.next_line();

            for (auto &row : data_reader)
            {
                vector<string> row_data;
                for (size_t i = 0; i < header.size(); i++)
                {
                    row_data.push_back(row[i].get<string>());
                }
                raw_rows.push_back(move(row_data));
            }

            // Now process all rows in parallel
            records.resize(raw_rows.size());

#pragma omp parallel for schedule(dynamic, 1000)
            for (size_t i = 0; i < raw_rows.size(); i++)
            {
                const auto &row_data = raw_rows[i];
                LoanRecord &record = records[i];

                // Read Text column if included
                if (include_text && idx_text >= 0 && idx_text < static_cast<int>(row_data.size()))
                {
                    record.text = row_data[idx_text];
                }

                // Parse numerical columns with error handling
                try
                {
                    if (idx_income >= 0 && idx_income < static_cast<int>(row_data.size()))
                        record.income = stod(row_data[idx_income]);

                    if (idx_credit_score >= 0 && idx_credit_score < static_cast<int>(row_data.size()))
                        record.credit_score = stoi(row_data[idx_credit_score]);

                    if (idx_loan_amount >= 0 && idx_loan_amount < static_cast<int>(row_data.size()))
                        record.loan_amount = stod(row_data[idx_loan_amount]);

                    if (idx_dti_ratio >= 0 && idx_dti_ratio < static_cast<int>(row_data.size()))
                        record.dti_ratio = stod(row_data[idx_dti_ratio]);

                    // Get categorical values for encoding
                    string employment_status = "";
                    string approval_status = "";

                    if (idx_employment >= 0 && idx_employment < static_cast<int>(row_data.size()))
                        employment_status = row_data[idx_employment];

                    if (idx_approval >= 0 && idx_approval < static_cast<int>(row_data.size()))
                        approval_status = row_data[idx_approval];

                    // Encode categorical variables
                    encode_categorical_vars(record, employment_status, approval_status);
                }
                catch (const exception &e)
                {
// Handle parsing errors for each row
#pragma omp critical
                    {
                        cerr << "Error parsing row " << i << ": " << e.what() << endl;
                    }
                }
            }

            cout << "Successfully loaded " << records.size() << " records" << endl;

            metric.end();
            profile_data.push_back(metric);
            return true;
        }
        catch (const exception &e)
        {
            cerr << "Error loading CSV file: " << e.what() << endl;
            metric.end();
            profile_data.push_back(metric);
            return false;
        }
    }

    void Dataset::encode_categorical_vars(LoanRecord &record, const string &employment, const string &approval)
    {
        // Encode employment status
        auto emp_it = employment_status_map.find(employment);
        record.employment_status = (emp_it != employment_status_map.end()) ? emp_it->second : -1;

        // Encode approval status
        auto app_it = approval_map.find(approval);
        record.approval = (app_it != approval_map.end()) ? app_it->second : -1;
    }

    bool Dataset::is_missing_value(const string &value) const
    {
        return value.empty() ||
               value == "NA" ||
               value == "N/A" ||
               value == "nan" ||
               value == "NaN" ||
               value == "?";
    }

    void Dataset::preprocess()
    {
        if (records.empty())
        {
            throw runtime_error("No data to preprocess. Load data first.");
        }

        // Calculate statistics (mean, std) for numerical features
        calculate_statistics();

        // Encode categorical features
        encode_categorical_variables();

        // Impute missing values
        impute_missing_values();

        // Normalize numerical features
        normalize_numerical_features();
    }

    void Dataset::calculate_statistics()
    {
        ProfileMetric metric("calculate_statistics");

        // Initialize statistics vectors
        const int NUM_FEATURES = 4; // income, credit_score, loan_amount, dti_ratio
        LoanRecord::column_means.resize(NUM_FEATURES, 0.0);
        LoanRecord::column_stddevs.resize(NUM_FEATURES, 0.0);

        vector<int> counts(NUM_FEATURES, 0);

// First pass: calculate means
#pragma omp parallel
        {
            // Thread-local sums and counts
            vector<double> local_sums(NUM_FEATURES, 0.0);
            vector<int> local_counts(NUM_FEATURES, 0);

#pragma omp for schedule(static)
            for (size_t i = 0; i < records.size(); i++)
            {
                const LoanRecord &record = records[i];

                // For each numerical feature, add to local sum if valid
                if (!isnan(record.income))
                {
                    local_sums[0] += record.income;
                    local_counts[0]++;
                }

                if (record.credit_score > 0)
                {
                    local_sums[1] += record.credit_score;
                    local_counts[1]++;
                }

                if (!isnan(record.loan_amount) && record.loan_amount > 0)
                {
                    local_sums[2] += record.loan_amount;
                    local_counts[2]++;
                }

                if (!isnan(record.dti_ratio) && record.dti_ratio > 0)
                {
                    local_sums[3] += record.dti_ratio;
                    local_counts[3]++;
                }
            }

// Combine results using reduction
#pragma omp critical
            {
                for (int j = 0; j < NUM_FEATURES; j++)
                {
                    LoanRecord::column_means[j] += local_sums[j];
                    counts[j] += local_counts[j];
                }
            }
        }

        // Calculate final means
        for (int j = 0; j < NUM_FEATURES; j++)
        {
            if (counts[j] > 0)
            {
                LoanRecord::column_means[j] /= counts[j];
            }
        }

// Second pass: calculate standard deviations
#pragma omp parallel
        {
            vector<double> local_var_sums(NUM_FEATURES, 0.0);
            vector<int> local_counts(NUM_FEATURES, 0);

#pragma omp for schedule(static)
            for (size_t i = 0; i < records.size(); i++)
            {
                const LoanRecord &record = records[i];

                if (!isnan(record.income))
                {
                    double diff = record.income - LoanRecord::column_means[0];
                    local_var_sums[0] += diff * diff;
                    local_counts[0]++;
                }

                if (record.credit_score > 0)
                {
                    double diff = record.credit_score - LoanRecord::column_means[1];
                    local_var_sums[1] += diff * diff;
                    local_counts[1]++;
                }

                if (!isnan(record.loan_amount) && record.loan_amount > 0)
                {
                    double diff = record.loan_amount - LoanRecord::column_means[2];
                    local_var_sums[2] += diff * diff;
                    local_counts[2]++;
                }

                if (!isnan(record.dti_ratio) && record.dti_ratio > 0)
                {
                    double diff = record.dti_ratio - LoanRecord::column_means[3];
                    local_var_sums[3] += diff * diff;
                    local_counts[3]++;
                }
            }

// Combine results
#pragma omp critical
            {
                for (int j = 0; j < NUM_FEATURES; j++)
                {
                    LoanRecord::column_stddevs[j] += local_var_sums[j];
                }
            }
        }

        // Calculate final standard deviations
        for (int j = 0; j < NUM_FEATURES; j++)
        {
            if (counts[j] > 1)
            {
                LoanRecord::column_stddevs[j] = sqrt(LoanRecord::column_stddevs[j] / (counts[j] - 1));
            }
            else
            {
                LoanRecord::column_stddevs[j] = 1.0; // Default to 1.0 to avoid division by zero
            }
        }

        cout << "Statistics calculation complete:" << endl;
        cout << "Column means: ";
        for (auto mean : LoanRecord::column_means)
        {
            cout << mean << " ";
        }
        cout << endl;

        cout << "Column std devs: ";
        for (auto stddev : LoanRecord::column_stddevs)
        {
            cout << stddev << " ";
        }
        cout << endl;

        metric.end();
        profile_data.push_back(metric);
    }

    void Dataset::encode_categorical_variables()
    {
        ProfileMetric metric("encode_categorical");

        // We've already encoded categories during loading,
        // but here we can verify and correct any issues

        // Check for employment status encoding
        int missing_employment = 0;
#pragma omp parallel for reduction(+ : missing_employment)
        for (size_t i = 0; i < records.size(); i++)
        {
            if (records[i].employment_status < 0)
            {
                missing_employment++;
            }
        }

        if (missing_employment > 0)
        {
            cout << "Warning: " << missing_employment
                 << " records with missing employment status" << endl;
        }

        // Check for approval encoding
        int missing_approval = 0;
#pragma omp parallel for reduction(+ : missing_approval)
        for (size_t i = 0; i < records.size(); i++)
        {
            if (records[i].approval < 0)
            {
                missing_approval++;
            }
        }

        if (missing_approval > 0)
        {
            cout << "Warning: " << missing_approval
                 << " records with missing approval status" << endl;
        }

        metric.end();
        profile_data.push_back(metric);
    }

    void Dataset::impute_missing_values()
    {
        ProfileMetric metric("impute_missing");

        // Create random number generator for imputation
        random_device rd;
        mt19937 gen(rd());

        // Create distributions for each numerical feature
        normal_distribution<> income_dist(
            LoanRecord::column_means[0],
            LoanRecord::column_stddevs[0] * 0.1); // Small variation

        normal_distribution<> credit_score_dist(
            LoanRecord::column_means[1],
            LoanRecord::column_stddevs[1] * 0.1);

        normal_distribution<> loan_amount_dist(
            LoanRecord::column_means[2],
            LoanRecord::column_stddevs[2] * 0.1);

        normal_distribution<> dti_ratio_dist(
            LoanRecord::column_means[3],
            LoanRecord::column_stddevs[3] * 0.1);

// Wait for all threads to see the updated means and stddevs
#pragma omp barrier

// Impute missing values in parallel
#pragma omp parallel
        {
            // Thread-local random engine to avoid contention
            mt19937 local_gen(rd() + omp_get_thread_num());

            // Create thread-local distributions
            normal_distribution<> local_income_dist(income_dist);
            normal_distribution<> local_credit_score_dist(credit_score_dist);
            normal_distribution<> local_loan_amount_dist(loan_amount_dist);
            normal_distribution<> local_dti_ratio_dist(dti_ratio_dist);

#pragma omp for schedule(dynamic, 1000)
            for (size_t i = 0; i < records.size(); i++)
            {
                LoanRecord &record = records[i];

                // Impute income if missing
                if (isnan(record.income) || record.income <= 0)
                {
                    record.income = max(0.0, local_income_dist(local_gen));
                }

                // Impute credit score if missing
                if (record.credit_score <= 0)
                {
                    record.credit_score = max(300, min(850, static_cast<int>(local_credit_score_dist(local_gen))));
                }

                // Impute loan amount if missing
                if (isnan(record.loan_amount) || record.loan_amount <= 0)
                {
                    record.loan_amount = max(0.0, local_loan_amount_dist(local_gen));
                }

                // Impute DTI ratio if missing
                if (isnan(record.dti_ratio) || record.dti_ratio <= 0)
                {
                    record.dti_ratio = max(0.0, local_dti_ratio_dist(local_gen));
                }

                // Impute employment status if missing
                if (record.employment_status < 0)
                {
                    record.employment_status = (local_gen() % 2); // Random 0 or 1
                }

                // Impute approval if missing
                if (record.approval < 0)
                {
                    record.approval = (local_gen() % 2); // Random 0 or 1
                }
            }
        }

        metric.end();
        profile_data.push_back(metric);
    }

    void Dataset::normalize_numerical_features()
    {
        ProfileMetric metric("normalize_features");

// Normalize numerical features using OpenMP SIMD
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < records.size(); i++)
        {
            LoanRecord &record = records[i];

            // Normalize income (z-score)
            if (LoanRecord::column_stddevs[0] > 0)
            {
                record.income = (record.income - LoanRecord::column_means[0]) / LoanRecord::column_stddevs[0];
            }

            // Normalize credit score (z-score)
            if (LoanRecord::column_stddevs[1] > 0)
            {
                record.credit_score = (record.credit_score - LoanRecord::column_means[1]) / LoanRecord::column_stddevs[1];
            }

// Use SIMD for vectorizable operations on continuous memory
#pragma omp simd
            for (int j = 0; j < 2; j++)
            {
                double *value = nullptr;
                double mean = 0.0;
                double stddev = 1.0;

                // Point to the right field and set appropriate mean/stddev
                if (j == 0)
                {
                    value = &record.loan_amount;
                    mean = LoanRecord::column_means[2];
                    stddev = LoanRecord::column_stddevs[2];
                }
                else if (j == 1)
                {
                    value = &record.dti_ratio;
                    mean = LoanRecord::column_means[3];
                    stddev = LoanRecord::column_stddevs[3];
                }

                // Normalize if we have a valid standard deviation
                if (value && stddev > 0)
                {
                    *value = (*value - mean) / stddev;
                }
            }
        }

        cout << "Feature normalization complete" << endl;

        metric.end();
        profile_data.push_back(metric);
    }

    void Dataset::save_to_file(const string &filename) const
    {
        ProfileMetric metric("save_file");

        try
        {
            ofstream file(filename);
            if (!file.is_open())
            {
                throw runtime_error("Could not open output file: " + filename);
            }

            // Write header
            file << "Income,Credit_Score,Loan_Amount,DTI_Ratio,Employment_Status,Approval";
            if (include_text)
            {
                file << ",Text";
            }
            file << endl;

            // Write data rows
            for (const auto &record : records)
            {
                file << fixed << setprecision(6)
                     << record.income << ","
                     << record.credit_score << ","
                     << record.loan_amount << ","
                     << record.dti_ratio << ","
                     << record.employment_status << ","
                     << record.approval;

                if (include_text)
                {
                    // Properly escape text field for CSV
                    string escaped_text = record.text;
                    // Replace quotes with double quotes
                    size_t pos = 0;
                    while ((pos = escaped_text.find("\"", pos)) != string::npos)
                    {
                        escaped_text.replace(pos, 1, "\"\"");
                        pos += 2;
                    }
                    file << ",\"" << escaped_text << "\"";
                }

                file << endl;
            }

            cout << "Saved " << records.size() << " records to " << filename << endl;

            metric.end();
            profile_data.push_back(metric);
        }
        catch (const exception &e)
        {
            cerr << "Error saving preprocessed data: " << e.what() << endl;
            metric.end();
            profile_data.push_back(metric);
        }
    }

    void Dataset::print_sample(int sample_size) const
    {
        if (records.empty())
        {
            cout << "No data to display." << endl;
            return;
        }

        int max_rows = min(static_cast<size_t>(sample_size), records.size());

        cout << "\nDataset Sample (first " << max_rows << " records):" << endl;
        cout << "-------------------------------------------------------------------------" << endl;
        cout << left
             << setw(12) << "Income"
             << setw(10) << "Credit"
             << setw(12) << "Loan_Amt"
             << setw(10) << "DTI"
             << setw(12) << "Employment"
             << setw(10) << "Approval";

        if (include_text)
        {
            cout << setw(30) << "Text";
        }
        cout << endl;

        cout << "-------------------------------------------------------------------------" << endl;

        for (int i = 0; i < max_rows; i++)
        {
            const auto &record = records[i];

            cout << fixed << setprecision(2)
                 << setw(12) << record.income
                 << setw(10) << record.credit_score
                 << setw(12) << record.loan_amount
                 << setw(10) << record.dti_ratio
                 << setw(12) << (record.employment_status == 1 ? "employed" : "unemployed")
                 << setw(10) << (record.approval == 1 ? "Approved" : "Rejected");

            if (include_text)
            {
                // Truncate text for display
                string display_text = record.text;
                if (display_text.length() > 27)
                {
                    display_text = display_text.substr(0, 24) + "...";
                }
                cout << setw(30) << display_text;
            }

            cout << endl;
        }
        cout << "-------------------------------------------------------------------------" << endl;
    }

    void Dataset::export_profiling_data(const string &filename) const
    {
        try
        {
            ofstream file(filename);
            if (!file.is_open())
            {
                throw runtime_error("Could not open profiling file: " + filename);
            }

            // Write header
            file << "Stage,ThreadID,ThreadCount,StartTime,EndTime,Duration" << endl;

            // Write profiling data
            for (const auto &metric : profile_data)
            {
                file << metric.stage_name << ","
                     << metric.thread_id << ","
                     << metric.thread_count << ","
                     << fixed << setprecision(6)
                     << metric.start_time << ","
                     << metric.end_time << ","
                     << (metric.end_time - metric.start_time) << endl;
            }

            cout << "Exported profiling data to " << filename << endl;
        }
        catch (const exception &e)
        {
            cerr << "Error exporting profiling data: " << e.what() << endl;
        }
    }

    // Factory function implementation
    unique_ptr<Dataset> load_and_preprocess(const string &filename, bool include_text)
    {
        // Create dataset
        auto dataset = make_unique<Dataset>(include_text);

        // Load data
        if (!dataset->load_from_file(filename))
        {
            throw runtime_error("Failed to load dataset from " + filename);
        }

        // Preprocess data
        try
        {
            dataset->preprocess();
        }
        catch (const exception &e)
        {
            throw runtime_error(string("Preprocessing failed: ") + e.what());
        }

        return dataset;
    }

} // namespace loan_preprocessing