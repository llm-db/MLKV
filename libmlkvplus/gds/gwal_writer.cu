#include "gwal_writer.cuh"
#include <algorithm>
#include <iostream>



namespace mlkv_plus {
namespace gds {

// CUDA kernel implementations
__global__ void format_record_header_kernel(char* header_buffer, RecordType type, size_t fragment_length, size_t header_size) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Set the crc to 0 (first 4 bytes)
        header_buffer[0] = 0;
        header_buffer[1] = 0;
        header_buffer[2] = 0;
        header_buffer[3] = 0;

        // Format the length (2 bytes, little-endian)
        header_buffer[4] = static_cast<char>(fragment_length & 0xff);
        header_buffer[5] = static_cast<char>(fragment_length >> 8);
        
        // Set the record type (1 byte)
        header_buffer[6] = static_cast<char>(type);
    }
}

__global__ void fill_trailer_kernel(char* trailer_buffer, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        trailer_buffer[idx] = '\x00';
    }
}

// Function implementation moved from header file
std::string GetWALPathName(const std::string& db_path) {
    // get the wal log name by reading the database directory
    std::string current_wal_name = "00000001.log"; // default name

    // Read the database directory to find .log files
    std::vector<std::string> log_files;
    for (const auto& entry : std::filesystem::directory_iterator(db_path)) {
        if (entry.is_regular_file() && entry.path().extension() == ".log") {
            log_files.push_back(entry.path().string());
        }
    }
    
    if (!log_files.empty()) {
        // Sort files by the numeric value in filename (e.g., 000001.log, 000002.log)
        std::sort(log_files.begin(), log_files.end(), [](const std::string& a, const std::string& b) {
            // Extract filename without path
            std::string filename_a = std::filesystem::path(a).filename().string();
            std::string filename_b = std::filesystem::path(b).filename().string();
            
            // Extract the numeric part before .log
            size_t dot_pos_a = filename_a.find(".log");
            size_t dot_pos_b = filename_b.find(".log");
            
            if (dot_pos_a != std::string::npos && dot_pos_b != std::string::npos) {
                try {
                    uint64_t num_a = std::stoull(filename_a.substr(0, dot_pos_a));
                    uint64_t num_b = std::stoull(filename_b.substr(0, dot_pos_b));
                    return num_a < num_b;
                } catch (const std::exception&) {
                    // Fallback to string comparison if parsing fails
                    return filename_a < filename_b;
                }
            }
            return filename_a < filename_b;
        });
        current_wal_name = log_files.back(); // Get the last (latest) log file
        // std::cout << "Found WAL file: " << current_wal_name << std::endl;
    } else {
        std::cerr << "No .log files found" << std::endl;
        throw std::runtime_error("No .log files found");
    }
    return current_wal_name;
}

} // namespace gds


} // namespace mlkv_plus
