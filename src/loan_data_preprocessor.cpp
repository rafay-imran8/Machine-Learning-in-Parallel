



// loan_data_preprocessor.cpp
#include "include/loan_data_preprocessor.h"
#include "include/csv.h" // Include fast-cpp-csv-parser

#include <iostream>
#include <fstream>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <iomanip>
#include <sstream>
using namespace io;
namespace loan_preprocessing
{
    // Initialize static members
    std::vector<double> LoanRecord::column_means;
    std::vector<double> LoanRecord::column_stddevs;

    // ProfileMetric implementation
    ProfileMetric::ProfileMetric(const std::string &name) : stage_name(name),
                                                            start_time(omp_get_wtime()),
                                                            thread_id(omp_get_thread_num()),
                                                            thread_count(omp_get_num_threads()) {}

    void ProfileMetric::end()
    {
        end_time = omp_get_wtime();
    }

    // Dataset implementation
    Dataset::Dataset()
    {
        // Initialize categorical mappings
        employment_status_map = {{"unemployed", 0}, {"employed", 1}};
        approval_map = {{"Rejected", 0}, {"Approved", 1}};
    }


bool Dataset::load_from_file(const std::string &filename)
{
    ProfileMetric metric("load_file");

    try
    {
        // Using CSVReader with correct column types
        io::CSVReader<6> reader(filename);
        reader.read_header(ignore_extra_column,
            "Income", "Credit_Score",
            "Loan_Amount", "DTI_Ratio",
            "Employment_Status", "Approval");
        

        records.clear();
        
        std::string employment_status, approval_status;
        double income, loan_amount, dti_ratio;
        int credit_score;

        // Read each row from the CSV
        while (reader.read_row(income, credit_score, loan_amount, dti_ratio, employment_status, approval_status)) {
            LoanRecord record;
            record.income = income;
            record.credit_score = credit_score;
            record.loan_amount = loan_amount;
            record.dti_ratio = dti_ratio;
        
            encode_categorical_vars(record, employment_status, approval_status);
            records.push_back(std::move(record));
        }
        

        std::cout << "Successfully loaded " << records.size() << " records\n";

        metric.end();
        profile_data.push_back(metric);
        return true;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error loading CSV file: " << e.what() << std::endl;
        metric.end();
        profile_data.push_back(metric);
        return false;
    }
}


    void Dataset::encode_categorical_vars(LoanRecord &record, const std::string &employment, const std::string &approval)
    {
        // Encode employment status
        auto emp_it = employment_status_map.find(employment);
        record.employment_status = (emp_it != employment_status_map.end()) ? emp_it->second : -1;

        // Encode approval status
        auto app_it = approval_map.find(approval);
        record.approval = (app_it != approval_map.end()) ? app_it->second : -1;
    }

    bool Dataset::is_missing_value(const std::string &value) const
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
            throw std::runtime_error("No data to preprocess. Load data first.");
        }

        // Calculate statistics (mean, std) for numerical features
        calculate_statistics();

        // Check for and fix any categorical encoding issues
        encode_categorical_variables();

        // Impute missing values
        impute_missing_values();
        
        // NO normalization step - we want to preserve original values
    }

    void Dataset::calculate_statistics()
    {
        ProfileMetric metric("calculate_statistics");

        // Initialize statistics vectors
        const int NUM_FEATURES = 4; // income, credit_score, loan_amount, dti_ratio
        LoanRecord::column_means.resize(NUM_FEATURES, 0.0);
        LoanRecord::column_stddevs.resize(NUM_FEATURES, 0.0);

        std::vector<int> counts(NUM_FEATURES, 0);

// First pass: calculate means
#pragma omp parallel
        {
            // Thread-local sums and counts
            std::vector<double> local_sums(NUM_FEATURES, 0.0);
            std::vector<int> local_counts(NUM_FEATURES, 0);

#pragma omp for schedule(static)
            for (size_t i = 0; i < records.size(); i++)
            {
                const LoanRecord &record = records[i];

                // For each numerical feature, add to local sum if valid
                if (!std::isnan(record.income) && record.income > 0)
                {
                    local_sums[0] += record.income;
                    local_counts[0]++;
                }

                if (record.credit_score > 0)
                {
                    local_sums[1] += record.credit_score;
                    local_counts[1]++;
                }

                if (!std::isnan(record.loan_amount) && record.loan_amount > 0)
                {
                    local_sums[2] += record.loan_amount;
                    local_counts[2]++;
                }

                if (!std::isnan(record.dti_ratio) && record.dti_ratio > 0)
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
            std::vector<double> local_var_sums(NUM_FEATURES, 0.0);
            std::vector<int> local_counts(NUM_FEATURES, 0);

#pragma omp for schedule(static)
            for (size_t i = 0; i < records.size(); i++)
            {
                const LoanRecord &record = records[i];

                if (!std::isnan(record.income) && record.income > 0)
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

                if (!std::isnan(record.loan_amount) && record.loan_amount > 0)
                {
                    double diff = record.loan_amount - LoanRecord::column_means[2];
                    local_var_sums[2] += diff * diff;
                    local_counts[2]++;
                }

                if (!std::isnan(record.dti_ratio) && record.dti_ratio > 0)
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
                LoanRecord::column_stddevs[j] = std::sqrt(LoanRecord::column_stddevs[j] / (counts[j] - 1));
            }
            else
            {
                LoanRecord::column_stddevs[j] = 1.0; // Default to 1.0 to avoid division by zero
            }
        }

        std::cout << "Statistics calculation complete:" << std::endl;
        std::cout << "Column means: ";
        std::cout << "Income: " << LoanRecord::column_means[0] << ", ";
        std::cout << "Credit Score: " << LoanRecord::column_means[1] << ", ";
        std::cout << "Loan Amount: " << LoanRecord::column_means[2] << ", ";
        std::cout << "DTI Ratio: " << LoanRecord::column_means[3];
        std::cout << std::endl;

        std::cout << "Column std devs: ";
        std::cout << "Income: " << LoanRecord::column_stddevs[0] << ", ";
        std::cout << "Credit Score: " << LoanRecord::column_stddevs[1] << ", ";
        std::cout << "Loan Amount: " << LoanRecord::column_stddevs[2] << ", ";
        std::cout << "DTI Ratio: " << LoanRecord::column_stddevs[3];
        std::cout << std::endl;

        metric.end();
        profile_data.push_back(metric);
    }

    void Dataset::encode_categorical_variables()
    {
        ProfileMetric metric("encode_categorical");

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
            std::cout << "Warning: " << missing_employment
                      << " records with missing employment status" << std::endl;
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
            std::cout << "Warning: " << missing_approval
                      << " records with missing approval status" << std::endl;
        }

        metric.end();
        profile_data.push_back(metric);
    }

    void Dataset::impute_missing_values()
    {
        ProfileMetric metric("impute_missing");

// Impute missing values in parallel
#pragma omp parallel for
        for (size_t i = 0; i < records.size(); i++)
        {
            LoanRecord &record = records[i];

            // Impute income if missing
            if (std::isnan(record.income) || record.income <= 0)
            {
                record.income = LoanRecord::column_means[0];
            }

            // Impute credit score if missing
            if (record.credit_score <= 0)
            {
                // Round to nearest integer since credit scores are integers
                record.credit_score = static_cast<int>(std::round(LoanRecord::column_means[1]));
                // Ensure it's in a valid range for credit scores
                record.credit_score = std::max(300, std::min(850, record.credit_score));
            }

            // Impute loan amount if missing
            if (std::isnan(record.loan_amount) || record.loan_amount <= 0)
            {
                record.loan_amount = LoanRecord::column_means[2];
            }

            // Impute DTI ratio if missing
            if (std::isnan(record.dti_ratio) || record.dti_ratio <= 0)
            {
                record.dti_ratio = LoanRecord::column_means[3];
            }

            // Impute employment status if missing - use most common value
            if (record.employment_status < 0)
            {
                record.employment_status = 1; // Default to employed (most common)
            }

            // Impute approval if missing
            if (record.approval < 0)
            {
                record.approval = 0; // Default to rejected (safer assumption)
            }
        }

        std::cout << "Missing value imputation complete" << std::endl;
        metric.end();
        profile_data.push_back(metric);
    }
    
    void Dataset::save_to_file(const std::string &filename)
    {
        ProfileMetric metric("save_file");
    
        try
        {
            std::ofstream file(filename);
            if (!file.is_open())
            {
                throw std::runtime_error("Could not open output file: " + filename);
            }
    
            // Write header
            file << "Income,Credit_Score,Loan_Amount,DTI_Ratio,Employment_Status,Approval";
            file << std::endl;
    
            // Write data rows with numeric values (ideal for model training)
            for (const auto &record : records)
            {
                file << std::fixed << std::setprecision(6)
                     << record.income << ","
                     << record.credit_score << ","
                     << record.loan_amount << ","
                     << record.dti_ratio << ","
                     << record.employment_status << ","
                     << record.approval;
                
                file << std::endl;
            }
    
            std::cout << "Saved " << records.size() << " records to " << filename << std::endl;
    
            metric.end();
            std::cout << "Save file time: " << (metric.end_time - metric.start_time) << " seconds" << std::endl;
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error saving preprocessed data: " << e.what() << std::endl;
            metric.end();
            std::cout << "Save file time: " << (metric.end_time - metric.start_time) << " seconds" << std::endl;
        }
    }

    void Dataset::print_sample(int sample_size) const
    {
        if (records.empty())
        {
            std::cout << "No data to display." << std::endl;
            return;
        }

        int max_rows = std::min(static_cast<size_t>(sample_size), records.size());

        std::cout << "\nDataset Sample (first " << max_rows << " records):" << std::endl;
        std::cout << "-------------------------------------------------------------------------" << std::endl;
        std::cout << std::left
                  << std::setw(12) << "Income"
                  << std::setw(10) << "Credit"
                  << std::setw(12) << "Loan_Amt"
                  << std::setw(10) << "DTI"
                  << std::setw(12) << "Employment"
                  << std::setw(10) << "Approval";
        std::cout << std::endl;

        std::cout << "-------------------------------------------------------------------------" << std::endl;

        for (int i = 0; i < max_rows; i++)
        {
            const auto &record = records[i];

            std::cout << std::fixed << std::setprecision(2)
                      << std::setw(12) << record.income
                      << std::setw(10) << record.credit_score
                      << std::setw(12) << record.loan_amount
                      << std::setw(10) << record.dti_ratio
                      << std::setw(12) << (record.employment_status == 1 ? "employed" : "unemployed")
                      << std::setw(10) << (record.approval == 1 ? "Approved" : "Rejected");
            std::cout << std::endl;
        }
        std::cout << "-------------------------------------------------------------------------" << std::endl;
    }
    
    void Dataset::print_preprocessed_sample(int sample_size) const
    {
        if (records.empty())
        {
            std::cout << "No data to display." << std::endl;
            return;
        }
    
        int max_rows = std::min(static_cast<size_t>(sample_size), records.size());
    
        std::cout << "\nPreprocessed Dataset Sample (first " << max_rows << " records) - NUMERIC VALUES:" << std::endl;
        std::cout << "-------------------------------------------------------------------------" << std::endl;
        
        // Print header
        std::cout << std::left
                  << std::setw(12) << "Income"
                  << std::setw(10) << "Credit"
                  << std::setw(12) << "Loan_Amt"
                  << std::setw(10) << "DTI"
                  << std::setw(12) << "Employment"
                  << std::setw(10) << "Approval";
        std::cout << std::endl;
    
        std::cout << "-------------------------------------------------------------------------" << std::endl;
    
        // Print rows with all numeric values
        for (int i = 0; i < max_rows; i++)
        {
            const auto &record = records[i];
    
            std::cout << std::fixed << std::setprecision(2)
                      << std::setw(12) << record.income
                      << std::setw(10) << record.credit_score
                      << std::setw(12) << record.loan_amount
                      << std::setw(10) << record.dti_ratio
                      << std::setw(12) << record.employment_status
                      << std::setw(10) << record.approval;
    
            std::cout << std::endl;
        }
        
        std::cout << "-------------------------------------------------------------------------" << std::endl;
        
        // Display summary of preprocessing
        std::cout << "All values shown in their numeric form after preprocessing." << std::endl;
        std::cout << "Employment Status: 0=unemployed, 1=employed" << std::endl;
        std::cout << "Approval: 0=Rejected, 1=Approved" << std::endl;
    }
    
    bool Dataset::verify_preprocessing() const
    {
        int missing_values = 0;
        int invalid_categorical = 0;
        
        // Check for missing values or invalid categorical encodings
        #pragma omp parallel for reduction(+:missing_values,invalid_categorical)
        for (size_t i = 0; i < records.size(); i++)
        {
            const auto& record = records[i];
            
            // Check numeric fields
            if (std::isnan(record.income) || record.income <= 0) missing_values++;
            if (record.credit_score <= 0) missing_values++;
            if (std::isnan(record.loan_amount) || record.loan_amount <= 0) missing_values++;
            if (std::isnan(record.dti_ratio) || record.dti_ratio <= 0) missing_values++;
            
            // Check categorical fields
            if (record.employment_status < 0 || record.employment_status > 1) invalid_categorical++;
            if (record.approval < 0 || record.approval > 1) invalid_categorical++;
        }
        
        if (missing_values > 0 || invalid_categorical > 0) {
            std::cout << "WARNING: Dataset still contains " << missing_values << " missing values and " 
                    << invalid_categorical << " invalid categorical values after preprocessing." << std::endl;
            return false;
        }
        
        std::cout << "Preprocessing verification successful - dataset is ready for model training." << std::endl;
        return true;
    }

    void Dataset::export_profiling_data(const std::string &filename) const
    {
        try
        {
            std::ofstream file(filename);
            if (!file.is_open())
            {
                throw std::runtime_error("Could not open profiling file: " + filename);
            }

            // Write header
            file << "Stage,ThreadID,ThreadCount,StartTime,EndTime,Duration" << std::endl;

            // Write profiling data
            for (const auto &metric : profile_data)
            {
                file << metric.stage_name << ","
                     << metric.thread_id << ","
                     << metric.thread_count << ","
                     << std::fixed << std::setprecision(6)
                     << metric.start_time << ","
                     << metric.end_time << ","
                     << (metric.end_time - metric.start_time) << std::endl;
            }

            std::cout << "Exported profiling data to " << filename << std::endl;
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error exporting profiling data: " << e.what() << std::endl;
        }
    }

    // Factory function implementation
    std::unique_ptr<Dataset> load_and_preprocess(const std::string &filename)
    {
        // Create a new Dataset instance
        auto dataset = std::make_unique<Dataset>();


        // Load data from file
        if (!dataset->load_from_file(filename))
        {
            throw std::runtime_error("Failed to load data from file: " + filename);
        }

        // Preprocess the data
        dataset->preprocess();

        return dataset;
    }

} // namespace loan_preprocessing