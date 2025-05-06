#include <iostream>
#include <fstream>
#include <string>
#include <vector>

// Simple function to check if file exists and is readable
bool validateModelFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open model file: " << filename << std::endl;
        return false;
    }
    
    // Get file size to verify it's not empty
    file.seekg(0, std::ios::end);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    if (size <= 0) {
        std::cerr << "Error: Model file is empty: " << filename << std::endl;
        return false;
    }
    
    // Read just the header or initial bytes to verify format
    // This is a simplification - you'll need to adapt based on your actual file format
    std::vector<char> header(std::min(static_cast<std::streamsize>(100), size));
    if (!file.read(header.data(), header.size())) {
        std::cerr << "Error: Failed to read header from: " << filename << std::endl;
        return false;
    }
    
    std::cout << "Model file " << filename << " appears valid (size: " << size << " bytes)" << std::endl;
    
    // Print first few bytes as hex for debugging
    std::cout << "First bytes: ";
    for (int i = 0; i < std::min(16, static_cast<int>(header.size())); i++) {
        printf("%02x ", static_cast<unsigned char>(header[i]));
    }
    std::cout << std::endl;
    
    return true;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " model1.bin [model2.bin ...]" << std::endl;
        return 1;
    }
    
    bool all_valid = true;
    
    // Validate each model file
    for (int i = 1; i < argc; i++) {
        std::cout << "Validating model: " << argv[i] << std::endl;
        if (!validateModelFile(argv[i])) {
            all_valid = false;
        }
        std::cout << std::endl;
    }
    
    if (all_valid) {
        std::cout << "All model files appear valid" << std::endl;
        return 0;
    } else {
        std::cerr << "Some model files failed validation" << std::endl;
        return 1;
    }
}