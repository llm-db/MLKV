#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <cuda_runtime.h>
#include <cuda.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <fstream>
#include <iomanip>
#include <filesystem>
#include <cstdio>
#include <iostream>

#include "gds_file_system.cuh"
#include "log_format.h"
#include "utils.cuh"

namespace mlkv_plus {

namespace gds {


constexpr bool kGWALDebugInfo = false;



// CUDA kernel declarations
__global__ void format_record_header_kernel(char* header_buffer, RecordType type, size_t fragment_length, size_t header_size);

__global__ void fill_trailer_kernel(char* trailer_buffer, size_t size);


class GWALWriter {
    public:
        GWALWriter(const std::string& relative_path, const std::string& db_path)
        : relative_path_(relative_path), db_path_(db_path), block_offset_(0), header_size_(kHeaderSize), 
          gpu_header_buffer_(nullptr), gpu_trailer_buffer_(nullptr),
          gpu_temp_buffer_(nullptr), gpu_buffer_size_(0) {
            db_path_ = db_path;
            if (relative_path_.empty()) {
                throw std::runtime_error("Relative path is empty");
            }
            if (relative_path_[0] == '/') {
                relative_path_ = relative_path_.substr(1);
            }
            filename_ = std::filesystem::path(db_path_).append(relative_path_).string();
            dest_ = new GDSFileSystem(filename_, GDSOptions());
            InitializeGPUBuffers();
        }
        ~GWALWriter() {
            CleanupGPUBuffers();
            delete dest_;
        };

        std::string GetWALPathName() const {
            return filename_;
        }

        std::string GetWALRelativePath() const {
            return relative_path_;
        }

        bool IsWALRelativePathEqual(const std::string& path) const {
            std::string cleaned_path = path;
            if (path[0] == '/') {
                cleaned_path = cleaned_path.substr(1);
            }
            return cleaned_path == relative_path_;
        }


        uint64_t GetWALNumber() const {
            // get the wal number from the filename
            std::string filename = std::filesystem::path(filename_).filename().string();
            return std::stoull(filename.substr(0, filename.find(".log")));
        }

        IOStatus AddRecord(const char* gpu_ptr, size_t n) {
            bool begin = true;
            size_t left = n;
            IOStatus s = IOStatus::OK();
            const char* current_gpu_ptr = gpu_ptr;

            do {
                const int64_t leftover = kBlockSize - block_offset_;
                assert(leftover >= 0);
                if (leftover < header_size_) {
                    // Switch to a new block
                    if (leftover > 0) {
                        // Fill the trailer using GPU kernel
                        assert(header_size_ <= 11);
                        s = FillTrailerOnGPU(static_cast<size_t>(leftover));
                        if (!s.ok()) {
                            break;
                        }
                    }
                    block_offset_ = 0;
                }
            
                // Invariant: we never leave < header_size bytes in a block.
                assert(static_cast<int64_t>(kBlockSize - block_offset_) >= header_size_);
            
                const size_t avail = kBlockSize - block_offset_ - header_size_;
                const size_t fragment_length = (left < avail) ? left : avail;
            
                RecordType type;
                const bool end = (left == fragment_length);
                if (begin && end) {
                    type = RecordType::kNoChecksumFullType;
                } else if (begin) {
                    type = RecordType::kNoChecksumFirstType;
                } else if (end) {
                    type = RecordType::kNoChecksumLastType;
                } else {
                    type = RecordType::kNoChecksumMiddleType;
                }
            
                s = EmitPhysicalRecordFromGPU(type, current_gpu_ptr, fragment_length);
                current_gpu_ptr += fragment_length;
                left -= fragment_length;
                begin = false;
            } while (s.ok() && (left > 0));
            return s;
        };

        // New GPU version
        IOStatus EmitPhysicalRecordFromGPU(RecordType t, const char* gpu_ptr, size_t n) {
            assert(n <= 0xffff);  // Must fit in two bytes

            // Add debug log - similar to log_writer.cc
            if (kGWALDebugInfo) {
                fprintf(stdout, "[DEBUG] Starting gwal_writer EmitPhysicalRecordFromGPU: type=%u, size=%zu\n", 
                    static_cast<unsigned>(t), n);
            }

            size_t header_size;
            if (t >= RecordType::kNoChecksumFullType && t <= RecordType::kNoChecksumLastType) {
                // Legacy record format
                assert(block_offset_ + kHeaderSize + n <= kBlockSize);
                header_size = kHeaderSize;
            } else {
                return IOStatus::UnsupportedError("Have not support recyclable record format");
            }

            // Format header on GPU to temp buffer
            IOStatus s = FormatRecordHeaderOnGPU(t, n, header_size);
            if (!s.ok()) {
                return s;
            }

            // Check if combined size fits in temp buffer
            size_t total_size = header_size + n;
            if (total_size > gpu_buffer_size_) {
                return IOStatus::IOError("Combined header and payload size (" + 
                                       std::to_string(total_size) + 
                                       ") exceeds temp buffer size (" + 
                                       std::to_string(gpu_buffer_size_) + ")");
            }

            // Combine header and payload into temp buffer for single write
            // First copy header to temp buffer
            cudaError_t err = cudaMemcpy(gpu_temp_buffer_, gpu_header_buffer_, header_size, cudaMemcpyDeviceToDevice);
            if (err != cudaSuccess) {
                return IOStatus::CudaError("Failed to copy header to temp buffer: " + 
                                         std::string(cudaGetErrorString(err)));
            }

            // Then copy payload after header
            err = cudaMemcpy(gpu_temp_buffer_ + header_size, gpu_ptr, n, cudaMemcpyDeviceToDevice);
            if (err != cudaSuccess) {
                return IOStatus::CudaError("Failed to copy payload to temp buffer: " + 
                                         std::string(cudaGetErrorString(err)));
            }

            // Write header + payload in a single operation (for 4k alignment)
            s = dest_->AppendFromGPU(gpu_temp_buffer_, total_size);
            if (!s.ok()) {
                return s;
            }

            if (kGWALDebugInfo) {
                // Print debug information about header and payload
                fprintf(stdout, "[DEBUG] header_size=%zu, payload_size=%zu, total_size=%zu\n", 
                    header_size, n, total_size);

                // Copy combined data from GPU to host for debugging
                size_t debug_len = std::min(total_size, static_cast<size_t>(64));
                char* host_buf = new char[debug_len];
                err = cudaMemcpy(host_buf, gpu_temp_buffer_, debug_len, cudaMemcpyDeviceToHost);
                if (err == cudaSuccess) {
                    fprintf(stdout, "[DEBUG] gwal_writer combined (header+payload): ");
                    for (size_t i = 0; i < debug_len; ++i) {
                        fprintf(stdout, "%02x ", static_cast<unsigned char>(host_buf[i]));
                    }
                    if (total_size > debug_len) {
                        fprintf(stdout, "...");
                    }
                    fprintf(stdout, "\n");
                } else {
                    fprintf(stdout, "[DEBUG] gwal_writer combined: failed to copy from GPU (%s)\n", 
                        cudaGetErrorString(err));
                }
                delete[] host_buf;
            }

            MLKV_CUDA_CHECK(cudaGetLastError());
            MLKV_CUDA_CHECK(cudaDeviceSynchronize());

            block_offset_ += header_size + n;
            return s;
        }

    private:
        std::string relative_path_;
        std::string db_path_;
        std::string filename_;


        int64_t block_offset_;
        int64_t header_size_;
        GDSFileSystem* dest_;

        // GPU buffers
        char* gpu_header_buffer_;
        char* gpu_trailer_buffer_;
        char* gpu_temp_buffer_;
        size_t gpu_buffer_size_;

        // GPU helper methods
        IOStatus InitializeGPUBuffers() {
            const size_t header_buffer_size = kRecyclableHeaderSize;
            const size_t trailer_buffer_size = 16; // Max trailer size
            const size_t temp_buffer_size = kBlockSize; // Temporary buffer for operations
            
            // Allocate GPU buffers
            cudaError_t err = cudaMalloc(&gpu_header_buffer_, header_buffer_size);
            if (err != cudaSuccess) {
                return IOStatus::CudaError("Failed to allocate GPU header buffer: " + 
                                         std::string(cudaGetErrorString(err)));
            }
            
            err = cudaMalloc(&gpu_trailer_buffer_, trailer_buffer_size);
            if (err != cudaSuccess) {
                cudaFree(gpu_header_buffer_);
                return IOStatus::CudaError("Failed to allocate GPU trailer buffer: " + 
                                         std::string(cudaGetErrorString(err)));
            }
            
            err = cudaMalloc(&gpu_temp_buffer_, temp_buffer_size);
            if (err != cudaSuccess) {
                cudaFree(gpu_header_buffer_);
                cudaFree(gpu_trailer_buffer_);
                return IOStatus::CudaError("Failed to allocate GPU temp buffer: " + 
                                         std::string(cudaGetErrorString(err)));
            }
            
            gpu_buffer_size_ = temp_buffer_size;
            return IOStatus::OK();
        }

        void CleanupGPUBuffers() {
            if (gpu_header_buffer_) {
                MLKV_CUDA_CHECK(cudaFree(gpu_header_buffer_));
                gpu_header_buffer_ = nullptr;
            }
            if (gpu_trailer_buffer_) {
                MLKV_CUDA_CHECK(cudaFree(gpu_trailer_buffer_));
                gpu_trailer_buffer_ = nullptr;
            }
            if (gpu_temp_buffer_) {
                MLKV_CUDA_CHECK(cudaFree(gpu_temp_buffer_));
                gpu_temp_buffer_ = nullptr;
            }
            gpu_buffer_size_ = 0;
        }

        IOStatus FormatRecordHeaderOnGPU(RecordType type, size_t fragment_length, size_t header_size) {
            // Launch CUDA kernel to format the header
            format_record_header_kernel<<<1, 1>>>(gpu_header_buffer_, type, fragment_length, header_size);
            MLKV_CUDA_CHECK(cudaGetLastError());
            MLKV_CUDA_CHECK(cudaDeviceSynchronize());
            return IOStatus::OK();
        }

        IOStatus FillTrailerOnGPU(size_t size) {
            // Launch CUDA kernel to fill the trailer
            fill_trailer_kernel<<<1, 1>>>(gpu_trailer_buffer_, size);
            
            MLKV_CUDA_CHECK(cudaGetLastError());
            
            MLKV_CUDA_CHECK(cudaDeviceSynchronize());
            
            // Write trailer to file
            return dest_->AppendFromGPU(gpu_trailer_buffer_, size);
        }

}; 

// Function declaration - implementation is in gwal_writer.cu
std::string GetWALPathName(const std::string& db_path);

} // namespace gds

} // namespace mlkv_plus