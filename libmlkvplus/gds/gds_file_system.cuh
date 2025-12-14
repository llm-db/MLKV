#pragma once


#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <string>
#include <cufile.h>
#include <fcntl.h>
#include <iostream>
#include <sys/types.h>
#include <unistd.h>
#include <sys/stat.h>
#include <cerrno>
#include <cstring>
#include "utils.cuh"


#include "rocksdb/file_system.h"
#include "format.cuh"
#include <unordered_map>
#include <list>
#include <mutex>
#include <memory>
#include <vector>
  

namespace mlkv_plus {

namespace gds {

// GPU-side block cache for GDS reads
class GPUBlockCache {
public:
    // Cache block metadata
    struct CacheBlock {
        uint64_t block_id;       // Block identifier (offset / BLOCK_SIZE)
        char* d_data;            // GPU pointer to cached data
        size_t size;             // Actual data size in this block
        uint64_t access_count;   // For LRU tracking
        
        CacheBlock() : block_id(0), d_data(nullptr), size(0), access_count(0) {}
        CacheBlock(uint64_t bid, char* data, size_t sz) 
            : block_id(bid), d_data(data), size(sz), access_count(0) {}
    };

    static constexpr size_t BLOCK_SIZE = 256 * 1024;      // 256KB per block
    static constexpr size_t MAX_CACHED_BLOCKS = 512;      // ~128MB cache
    
    GPUBlockCache() : access_counter_(0), cache_hits_(0), cache_misses_(0) {
        // Pre-allocate GPU memory pool
        size_t total_cache_size = BLOCK_SIZE * MAX_CACHED_BLOCKS;
        cudaError_t err = cudaMalloc(&d_cache_pool_, total_cache_size);
        if (err != cudaSuccess) {
            std::cerr << "Failed to allocate GPU cache pool: " 
                      << cudaGetErrorString(err) << std::endl;
            d_cache_pool_ = nullptr;
        }
    }
    
    ~GPUBlockCache() {
        if (d_cache_pool_ != nullptr) {
            cudaFree(d_cache_pool_);
        }
    }
    
    // Try to read from cache, return nullptr if miss
    // Returns GPU pointer to cached data and actual size
    // Note: Only handles single-block reads. For cross-block reads, returns nullptr.
    char* TryGet(uint64_t file_offset, size_t request_size, 
                 size_t* cached_size, cudaStream_t stream = 0) {
        uint64_t block_id = file_offset / BLOCK_SIZE;
        uint64_t offset_in_block = file_offset % BLOCK_SIZE;
        
        std::lock_guard<std::mutex> lock(mutex_);
        
        auto it = cache_map_.find(block_id);
        if (it == cache_map_.end()) {
            cache_misses_++;
            return nullptr;  // Cache miss
        }
        
        // Check if requested data fits in this block
        if (offset_in_block + request_size > it->second.size) {
            cache_misses_++;
            return nullptr;  // Cross-block read, fall back to GDS
        }
        
        // Update LRU info and record hit
        it->second.access_count = ++access_counter_;
        cache_hits_++;
        
        *cached_size = std::min(request_size, it->second.size - offset_in_block);
        return it->second.d_data + offset_in_block;
    }
    
    // Try to read from multiple cached blocks (for cross-block reads)
    // Returns true if all required blocks are cached
    bool TryGetMultiBlock(uint64_t file_offset, size_t request_size,
                         char* d_output, cudaStream_t stream = 0) {
        if (request_size == 0) return true;
        
        uint64_t start_block = file_offset / BLOCK_SIZE;
        uint64_t end_offset = file_offset + request_size - 1;
        uint64_t end_block = end_offset / BLOCK_SIZE;
        
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Check if all required blocks are cached
        std::vector<uint64_t> required_blocks;
        for (uint64_t bid = start_block; bid <= end_block; ++bid) {
            if (cache_map_.find(bid) == cache_map_.end()) {
                cache_misses_++;
                return false;  // At least one block missing
            }
            required_blocks.push_back(bid);
        }
        
        // All blocks found! Copy data from each block
        cache_hits_++;
        size_t copied = 0;
        uint64_t current_offset = file_offset;
        
        for (uint64_t bid : required_blocks) {
            auto& block = cache_map_[bid];
            block.access_count = ++access_counter_;  // Update LRU
            
            uint64_t block_start = bid * BLOCK_SIZE;
            uint64_t offset_in_block = current_offset - block_start;
            size_t bytes_in_this_block = std::min(
                request_size - copied,
                block.size - offset_in_block
            );
            
            cudaMemcpyAsync(d_output + copied, 
                          block.d_data + offset_in_block,
                          bytes_in_this_block,
                          cudaMemcpyDeviceToDevice, 
                          stream);
            
            copied += bytes_in_this_block;
            current_offset += bytes_in_this_block;
        }
        
        return true;
    }
    
    // Insert data into cache
    void Put(uint64_t file_offset, const char* d_source, size_t size, 
             cudaStream_t stream = 0) {
        if (d_cache_pool_ == nullptr) return;
        
        uint64_t block_id = file_offset / BLOCK_SIZE;
        
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Check if already cached
        if (cache_map_.find(block_id) != cache_map_.end()) {
            return;  // Already cached
        }
        
        // Evict if necessary
        if (cache_map_.size() >= MAX_CACHED_BLOCKS) {
            EvictLRU();
        }
        
        // Find free slot in pool
        size_t slot_idx = free_slots_.empty() ? cache_map_.size() : free_slots_.back();
        if (!free_slots_.empty()) {
            free_slots_.pop_back();
        }
        
        if (slot_idx >= MAX_CACHED_BLOCKS) {
            return;  // No space (shouldn't happen after eviction)
        }
        
        // Allocate block in cache pool
        char* d_block = (char*)d_cache_pool_ + (slot_idx * BLOCK_SIZE);
        
        // Copy data to cache (Device to Device)
        size_t copy_size = std::min(size, BLOCK_SIZE);
        cudaMemcpyAsync(d_block, d_source, copy_size, 
                       cudaMemcpyDeviceToDevice, stream);
        
        // Insert into cache
        CacheBlock block(block_id, d_block, copy_size);
        block.access_count = ++access_counter_;
        cache_map_[block_id] = block;
        slot_map_[block_id] = slot_idx;
    }
    
    // Get cache statistics
    void GetStats(size_t* hits, size_t* misses, size_t* cached_blocks) const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (hits) *hits = cache_hits_;
        if (misses) *misses = cache_misses_;
        if (cached_blocks) *cached_blocks = cache_map_.size();
    }
    
    void Clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        cache_map_.clear();
        slot_map_.clear();
        free_slots_.clear();
        access_counter_ = 0;
        cache_hits_ = 0;
        cache_misses_ = 0;
    }

private:
    void EvictLRU() {
        // Find block with smallest access_count
        auto lru_it = cache_map_.begin();
        uint64_t min_access = lru_it->second.access_count;
        
        for (auto it = cache_map_.begin(); it != cache_map_.end(); ++it) {
            if (it->second.access_count < min_access) {
                min_access = it->second.access_count;
                lru_it = it;
            }
        }
        
        // Return slot to free list
        uint64_t evicted_block_id = lru_it->first;
        size_t slot_idx = slot_map_[evicted_block_id];
        free_slots_.push_back(slot_idx);
        
        // Remove from cache
        slot_map_.erase(evicted_block_id);
        cache_map_.erase(lru_it);
    }
    
    void* d_cache_pool_;                              // Pre-allocated GPU memory
    std::unordered_map<uint64_t, CacheBlock> cache_map_;  // block_id -> CacheBlock
    std::unordered_map<uint64_t, size_t> slot_map_;   // block_id -> slot index
    std::vector<size_t> free_slots_;                  // Available slots after eviction
    
    mutable std::mutex mutex_;                        // Thread safety
    uint64_t access_counter_;                         // For LRU tracking
    
    size_t cache_hits_;
    size_t cache_misses_;
};

class GDSFileSystem {

    public:
        GDSFileSystem(const std::string& filename, const GDSOptions& options):
         filename_(filename), options_(options), file_fd_(-1), gds_initialized_(false), 
         file_opened_(false), file_size_(0), gpu_buffer_(nullptr), gpu_buffer_size_(0),
         write_buffer_(nullptr), write_buffer_used_(0) {

            IOStatus status = InitializeGDS();
            if (!status.ok()) {
                std::cerr << "Failed to initialize GDS: " << status.message() << std::endl;
                std::exit(EXIT_FAILURE);
            }

            // Allocate 4k write buffer
            cudaError_t err = cudaMalloc(&write_buffer_, options_.GDSAlignment);
            if (err != cudaSuccess) {
                std::cerr << "Failed to allocate write buffer: " << cudaGetErrorString(err) << std::endl;
                std::exit(EXIT_FAILURE);
            }

         };
        ~GDSFileSystem() {
            // Flush any remaining data in write buffer
            if (write_buffer_used_ > 0) {
                Flush();
            }
            
            if (gds_initialized_) {
                cuFileHandleDeregister(cu_file_handle_);
            }
            if (gpu_buffer_ != nullptr) {
                cudaFree(gpu_buffer_);
                gpu_buffer_ = nullptr;
            }
            if (write_buffer_ != nullptr) {
                cudaFree(write_buffer_);
                write_buffer_ = nullptr;
            }
            if (file_fd_ >= 0) {
                close(file_fd_);
                file_fd_ = -1;
            }
            gds_initialized_ = false;
            file_opened_ = false;
        };


        IOStatus WriteFromGPU(const char* data, size_t size, 
                        uint64_t offset) {
            if (!gds_initialized_) {
                return IOStatus::UsageError("CUDA GDS not initialized");
            }

            // Check if this is a sequential write at the current file position
            // If so, use the write buffer for better alignment handling
            if (offset == file_size_) {
                return BufferedWrite(data, size);
            }

            // For non-sequential writes, require alignment
            if (offset % options_.GDSAlignment != 0) {
                std::cerr << "Warning: offset is not aligned to GDSAlignment when writing to GDS | current offset: " << offset << " | options_.GDSAlignment: " << options_.GDSAlignment << std::endl;
            }
            if (size % options_.GDSAlignment != 0) {
                std::cerr << "Warning: size is not aligned to GDSAlignment when writing to GDS | current size: " << size << " | options_.GDSAlignment: " << options_.GDSAlignment << std::endl;
            }

            // Write directly from GPU to storage using GDS
            ssize_t cu_err = cuFileWrite(cu_file_handle_, data, size, offset, 0);

            if (cu_err < 0) {
                return IOStatus::IOError("cuFileWrite failed: " + std::to_string(cu_err));
            }

            // Update file size
            if (offset + size > file_size_) {
                file_size_ = offset + size;
            }

            return IOStatus::OK();
        }


        IOStatus AppendFromGPU(const char* data, size_t size) {
            // Use current file_size_ as offset and update it atomically
            uint64_t offset = file_size_;
            IOStatus status = WriteFromGPU(data, size, offset);
            return IOStatus::OK();
        }

        IOStatus PositionedAppendFromGPU(const char* data, size_t size, uint64_t offset) {
            return WriteFromGPU(data, size, offset);
        }

    public:
        // Flush the write buffer to disk
        IOStatus Flush() {
            if (!gds_initialized_) {
                return IOStatus::UsageError("CUDA GDS not initialized");
            }

            if (write_buffer_used_ == 0) {
                return IOStatus::OK();  // Nothing to flush
            }

            // Write the remaining data directly (GDS compat mode handles unaligned writes)
            ssize_t cu_err = cuFileWrite(cu_file_handle_, write_buffer_, write_buffer_used_, file_size_, 0);
            if (cu_err < 0) {
                return IOStatus::IOError("cuFileWrite failed during flush: " + std::to_string(cu_err));
            }

            // Update file size
            file_size_ += write_buffer_used_;
            write_buffer_used_ = 0;

            return IOStatus::OK();
        }

    private:
        std::string filename_;
        GDSOptions options_;


        int file_fd_;
        CUfileHandle_t cu_file_handle_;
        bool gds_initialized_;
        bool file_opened_;
        uint64_t file_size_;

        void* gpu_buffer_;
        size_t gpu_buffer_size_;

        // Write buffer for handling unaligned writes
        void* write_buffer_;        // 4k buffer
        size_t write_buffer_used_;  // How much of the buffer is currently used


        // Buffered write implementation
        IOStatus BufferedWrite(const char* data, size_t size) {
            size_t total_size = write_buffer_used_ + size;

            if (total_size < options_.GDSAlignment) {
                // Total data < 4k, just save to buffer
                cudaMemcpy((char*)write_buffer_ + write_buffer_used_, 
                          data, 
                          size, 
                          cudaMemcpyDeviceToDevice);
                write_buffer_used_ += size;
                return IOStatus::OK();
            }

            // Total data >= 4k, need to write
            // Calculate how much data can be written with 4k alignment
            size_t aligned_write_size = (total_size / options_.GDSAlignment) * options_.GDSAlignment;
            size_t remaining_size = total_size - aligned_write_size;  // Remaining data to put back in buffer

            // Allocate temporary buffer to merge write_buffer_ and new data
            void* temp_buffer = nullptr;
            cudaError_t cuda_err = cudaMalloc(&temp_buffer, aligned_write_size);
            if (cuda_err != cudaSuccess) {
                return IOStatus::IOError("Failed to allocate temp buffer: " + 
                                        std::string(cudaGetErrorString(cuda_err)));
            }

            // Merge data: first copy data from write_buffer_
            if (write_buffer_used_ > 0) {
                cudaMemcpy(temp_buffer, write_buffer_, write_buffer_used_, cudaMemcpyDeviceToDevice);
            }

            // Then copy the aligned portion of new data
            size_t new_data_to_write = aligned_write_size - write_buffer_used_;
            cudaMemcpy((char*)temp_buffer + write_buffer_used_, 
                      data, 
                      new_data_to_write, 
                      cudaMemcpyDeviceToDevice);

            assert(aligned_write_size % options_.GDSAlignment == 0);
            assert(file_size_ % options_.GDSAlignment == 0);

            // Write all 4k-aligned data in one operation
            ssize_t cu_err = cuFileWrite(cu_file_handle_, temp_buffer, 
                                         aligned_write_size, file_size_, 0);
            
            cudaFree(temp_buffer);  // Free temporary buffer

            if (cu_err < 0) {
                return IOStatus::IOError("cuFileWrite failed: " + std::to_string(cu_err));
            }

            // Update file size
            file_size_ += aligned_write_size;

            // Handle remaining data: put unaligned portion back to write_buffer_
            write_buffer_used_ = 0;
            if (remaining_size > 0) {
                cudaMemcpy(write_buffer_, 
                          data + new_data_to_write, 
                          remaining_size, 
                          cudaMemcpyDeviceToDevice);
                write_buffer_used_ = remaining_size;
            }

            return IOStatus::OK();
        }

        IOStatus EnsureGPUBuffer(size_t required_size) {
            // Align to GDS requirements
            size_t aligned_size = (required_size + options_.GDSAlignment - 1) & ~(options_.GDSAlignment - 1);
            
            // Check if current buffer is sufficient
            if (gpu_buffer_ != nullptr && gpu_buffer_size_ >= aligned_size) {
                return IOStatus::OK();
            }
            
            // Free existing buffer if any
            if (gpu_buffer_ != nullptr) {
                cudaFree(gpu_buffer_);
                gpu_buffer_ = nullptr;
                gpu_buffer_size_ = 0;
            }
            
            // Allocate new buffer
            cudaError_t err = cudaMalloc(&gpu_buffer_, aligned_size);
            if (err != cudaSuccess) {
              gpu_buffer_ = nullptr;
              gpu_buffer_size_ = 0;
              return IOStatus::IOError("Failed to allocate GPU buffer: " + 
                                      std::string(cudaGetErrorString(err)));
            }
          
            gpu_buffer_size_ = aligned_size;
            return IOStatus::OK();
        }

        IOStatus InitializeGDS() {
            // 1. Initialize CUDA context (assuming device 0)
            cudaError_t cuda_err = cudaSetDevice(0);
            if (cuda_err != cudaSuccess) {
              return IOStatus::CudaError("Failed to set CUDA device: " + 
                                      std::string(cudaGetErrorString(cuda_err)));
            }
          
            // 2. Open file with O_DIRECT for GDS compatibility (non-exclusive)
            // Note: O_DIRECT and O_APPEND cannot be used together on some systems
            // We'll handle append manually by tracking file size
            file_fd_ = open(filename_.c_str(), 
                            O_CREAT | O_WRONLY | O_DIRECT, 
                            0644);
            if (file_fd_ < 0) {
              return IOStatus::IOError("Failed to open file for GDS: " + 
                                      std::string(strerror(errno)));
            }
          
            // 3. Get current file size for append operations
            struct stat file_stat;
            if (fstat(file_fd_, &file_stat) == 0) {
              file_size_ = file_stat.st_size;
            } else {
              file_size_ = 0; // If we can't get file size, assume it's empty
            }
          
            // 4. Register file with cuFile
            CUfileDescr_t file_descr;
            memset(&file_descr, 0, sizeof(file_descr));
            file_descr.handle.fd = file_fd_;
            file_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
          
            CUfileError_t cu_err = cuFileHandleRegister(&cu_file_handle_, &file_descr);
            if (cu_err.err != CU_FILE_SUCCESS) {
              close(file_fd_);
              file_fd_ = -1;
              return IOStatus::IOError("Failed to register file with cuFile: " + 
                                      std::to_string(cu_err.err));
            }
          
            // 5. Allocate initial GPU buffer
            IOStatus buffer_status = EnsureGPUBuffer(options_.DefaultGPUBufferSize);
            if (!buffer_status.ok()) {
              cuFileHandleDeregister(cu_file_handle_);
              close(file_fd_);
              file_fd_ = -1;
              return buffer_status;
            }
          
            gds_initialized_ = true;
            file_opened_ = true;
            return IOStatus::OK();
        }
};



class GDSRandomAccessFile : public rocksdb::FSRandomAccessFile {
public:
    GDSRandomAccessFile(const std::string& filename, const GDSOptions& options)
        : filename_(filename), options_(options), file_fd_(-1), 
          gds_initialized_(false), file_size_(0), 
          gpu_buffer_(nullptr), gpu_buffer_size_(0),
          gpu_cache_(options.EnableGPUCache ? std::make_shared<GPUBlockCache>() : nullptr) {
        
        IOStatus status = InitializeGDS();
        if (!status.ok()) {
            std::cerr << "Failed to initialize GDS for reading: " << status.message() << std::endl;
            throw std::runtime_error("GDS initialization failed: " + status.message());
        }
        
        // if (gpu_cache_) {
        //     std::cout << "[GDS] GPU Cache ENABLED - BlockSize: " 
        //               << (options.GPUCacheBlockSize / 1024) << "KB, MaxBlocks: " 
        //               << options.GPUCacheMaxBlocks << " (~" 
        //               << (options.GPUCacheBlockSize * options.GPUCacheMaxBlocks / (1024*1024)) 
        //               << "MB)" << std::endl;
        // } else {
        //     std::cout << "[GDS] GPU Cache DISABLED" << std::endl;
        // }
    }

    ~GDSRandomAccessFile() override {
        if (gds_initialized_) {
            cuFileHandleDeregister(cu_file_handle_);
        }
        if (gpu_buffer_ != nullptr) {
            cudaFree(gpu_buffer_);
            gpu_buffer_ = nullptr;
        }
        if (file_fd_ >= 0) {
            close(file_fd_);
            file_fd_ = -1;
        }
        gds_initialized_ = false;
    }


    // GPU read method with cache support
    IOStatus Read(uint64_t offset, size_t n, 
                            const rocksdb::IOOptions& options,
                            DeviceSlice* result, char* d_scratch,
                            rocksdb::IODebugContext* dbg, cudaStream_t stream) const {
        (void)options;  // Unused parameter
        (void)dbg;      // Unused parameter

        if (!gds_initialized_) {
            return IOStatus::IOError("GDS not initialized");
        }

        // Try cache first (if enabled)
        if (gpu_cache_) {
            // Try 1: Single-block read (fast path)
            size_t cached_size = 0;
            char* cached_ptr = gpu_cache_->TryGet(offset, n, &cached_size, stream);
            
            if (cached_ptr != nullptr && cached_size >= n) {
                // Single-block cache hit!
                if (d_scratch != nullptr) {
                    cudaMemcpyAsync(d_scratch, cached_ptr, n, cudaMemcpyDeviceToDevice, stream);
                    *result = DeviceSlice(d_scratch, n);
                } else {
                    *result = DeviceSlice(cached_ptr, n);
                }
                return IOStatus::OK();
            }
            
            // Try 2: Multi-block read (cross-block case)
            if (d_scratch != nullptr) {
                if (gpu_cache_->TryGetMultiBlock(offset, n, d_scratch, stream)) {
                    // Multi-block cache hit!
                    *result = DeviceSlice(d_scratch, n);
                    return IOStatus::OK();
                }
            } else {
                // Need temp buffer for multi-block read
                IOStatus buffer_status = EnsureGPUBuffer(n);
                if (buffer_status.ok()) {
                    if (gpu_cache_->TryGetMultiBlock(offset, n, (char*)gpu_buffer_, stream)) {
                        *result = DeviceSlice((char*)gpu_buffer_, n);
                        return IOStatus::OK();
                    }
                }
            }
        }

        // Cache miss - decide whether to prefetch full block or just requested data
        
        // Calculate block-aligned range for caching
        uint64_t block_start = (offset / GPUBlockCache::BLOCK_SIZE) * GPUBlockCache::BLOCK_SIZE;
        
        off_t read_offset = offset;
        size_t read_n = n;
        uint64_t offset_in_read = 0;
        char* read_buffer = d_scratch;
        bool need_temp_buffer = false;
        bool should_prefetch = false;
        
        // Decide: prefetch full block or just read requested data?
        if (gpu_cache_ && options_.EnableCachePrefetch && 
            n < GPUBlockCache::BLOCK_SIZE && 
            n >= options_.MinReadSizeForCache) {
            // Small read + prefetch enabled: read entire cache block
            read_offset = block_start;
            read_n = std::min(GPUBlockCache::BLOCK_SIZE, (size_t)(file_size_ - block_start));
            offset_in_read = offset - block_start;
            should_prefetch = true;
            need_temp_buffer = true;
        } else {
            // Normal read: just get requested data
            bool is_aligned = (offset % options_.GDSAlignment == 0) && (n % options_.GDSAlignment == 0);
            
            if (!is_aligned) {
                // Align to GDS requirements (4K)
                read_offset = (offset / options_.GDSAlignment) * options_.GDSAlignment;
                offset_in_read = offset - read_offset;
                read_n = ((offset_in_read + n + options_.GDSAlignment - 1) / options_.GDSAlignment) * options_.GDSAlignment;
                need_temp_buffer = true;
            } else if (d_scratch == nullptr) {
                need_temp_buffer = true;
            }
        }
        
        // Align to GDS requirements
        if (read_offset % options_.GDSAlignment != 0) {
            uint64_t aligned_offset = (read_offset / options_.GDSAlignment) * options_.GDSAlignment;
            offset_in_read += (read_offset - aligned_offset);
            read_offset = aligned_offset;
        }
        if (read_n % options_.GDSAlignment != 0) {
            read_n = ((read_n + options_.GDSAlignment - 1) / options_.GDSAlignment) * options_.GDSAlignment;
        }
        
        // Get buffer
        if (need_temp_buffer) {
            IOStatus buffer_status = EnsureGPUBuffer(read_n);
            if (!buffer_status.ok()) {
                return IOStatus::IOError("Failed to ensure GPU buffer");
            }
            read_buffer = (char*)gpu_buffer_;
        }

        assert(read_n % options_.GDSAlignment == 0);
        assert(read_offset % options_.GDSAlignment == 0);

        ssize_t bytes_read = 0;
        off_t buffer_offset = 0;
        // Perform GDS read directly from storage to GPU memory
        cuFileReadAsync(cu_file_handle_, read_buffer, &read_n, &read_offset, &buffer_offset, &bytes_read, stream);
        cudaStreamSynchronize(stream);

        if (bytes_read < 0) {
            return IOStatus::IOError("cuFileRead failed with error: " + 
                                        std::to_string(bytes_read));
        }

        // Store in cache for future reuse
        if (gpu_cache_ && bytes_read > 0 && should_prefetch) {
            // We prefetched a full block - cache it!
            size_t cache_size = std::min((size_t)bytes_read, GPUBlockCache::BLOCK_SIZE);
            gpu_cache_->Put(block_start, read_buffer, cache_size, stream);
        }

        // Extract the actual requested data from read buffer
        size_t actual_bytes = std::min(n, (size_t)(bytes_read - offset_in_read));
        if (need_temp_buffer) {
            if (d_scratch != nullptr) {
                cudaMemcpyAsync(d_scratch, read_buffer + offset_in_read, actual_bytes, 
                              cudaMemcpyDeviceToDevice, stream);
                *result = DeviceSlice(d_scratch, actual_bytes);
            } else {
                *result = DeviceSlice(read_buffer + offset_in_read, actual_bytes);
            }
        } else {
            *result = DeviceSlice(read_buffer, actual_bytes);
        }

        return IOStatus::OK();
    }

    // Compatibility read method
    rocksdb::IOStatus Read(uint64_t offset, size_t n, 
                          const rocksdb::IOOptions& options,
                          rocksdb::Slice* result, char* scratch,
                          rocksdb::IODebugContext* dbg) const override {
        (void)options;  // Unused parameter
        (void)dbg;      // Unused parameter
        (void)result;   // Unused parameter
        
        if (!gds_initialized_) {
            return rocksdb::IOStatus::IOError("GDS not initialized");
        }

        if (scratch == nullptr) {
            return rocksdb::IOStatus::InvalidArgument("scratch buffer is null");
        }

        // Check if offset and n are aligned to 4K
        bool is_aligned = (offset % options_.GDSAlignment == 0) && (n % options_.GDSAlignment == 0);
        
        uint64_t read_offset = offset;
        size_t read_n = n;
        uint64_t offset_in_aligned_block = 0;
        
        if (!is_aligned) {
            // Align offset down to 4K boundary
            read_offset = (offset / options_.GDSAlignment) * options_.GDSAlignment;
            offset_in_aligned_block = offset - read_offset;
            
            // Align n up to cover the entire range
            read_n = ((offset_in_aligned_block + n + options_.GDSAlignment - 1) / options_.GDSAlignment) * options_.GDSAlignment;
        }

        // Ensure we have enough GPU buffer space for aligned read
        IOStatus buffer_status = EnsureGPUBuffer(read_n);
        if (!buffer_status.ok()) {
            return rocksdb::IOStatus::IOError("Failed to allocate GPU buffer: " + 
                                             buffer_status.message());
        }


        assert(read_n % options_.GDSAlignment == 0);
        assert(read_offset % options_.GDSAlignment == 0);

        // Perform GDS read directly from storage to GPU memory with aligned parameters
        ssize_t bytes_read = cuFileRead(cu_file_handle_, gpu_buffer_, read_n, read_offset, 0);
        
        if (bytes_read < 0) {
            return rocksdb::IOStatus::IOError("cuFileRead failed with error: " + 
                                             std::to_string(bytes_read));
        }

        // Recover result: extract the actual requested data
        size_t actual_bytes = n;
        if (!is_aligned && bytes_read > (ssize_t)offset_in_aligned_block) {
            actual_bytes = std::min(n, (size_t)(bytes_read - offset_in_aligned_block));
        } else if (is_aligned) {
            actual_bytes = std::min(n, (size_t)bytes_read);
        }

        // Copy data from GPU to host memory (scratch buffer), skipping alignment offset
        cudaError_t cuda_err = cudaMemcpy(scratch, 
                                          (char*)gpu_buffer_ + offset_in_aligned_block, 
                                          actual_bytes, cudaMemcpyDeviceToHost);
        if (cuda_err != cudaSuccess) {
            return rocksdb::IOStatus::IOError("Failed to copy data from GPU to host: " + 
                                             std::string(cudaGetErrorString(cuda_err)));
        }

        // Set result to point to the scratch buffer with actual bytes read
        *result = rocksdb::Slice(scratch, actual_bytes);


        // Print the result
        // std::cout << "Read " << bytes_read << " bytes from GDS" << std::endl;
        
        return rocksdb::IOStatus::OK();
    }

    // Prefetch support
    rocksdb::IOStatus Prefetch(uint64_t offset, size_t n,
                              const rocksdb::IOOptions& options,
                              rocksdb::IODebugContext* dbg) override {
        (void)options;
        (void)dbg;
        
        // GDS doesn't have built-in prefetch, but we can trigger a read
        // to warm up the path. For now, return NotSupported to let RocksDB
        // handle prefetching internally
        return rocksdb::IOStatus::NotSupported("Prefetch not implemented for GDS");
    }

    // MultiRead support - read multiple blocks in parallel
    rocksdb::IOStatus MultiRead(rocksdb::FSReadRequest* reqs, size_t num_reqs,
                               const rocksdb::IOOptions& options,
                               rocksdb::IODebugContext* dbg) override {
        if (!gds_initialized_) {
            return rocksdb::IOStatus::IOError("GDS not initialized");
        }

        // Process each request sequentially
        // For true parallel reads, we would need multiple GPU buffers and async operations
        for (size_t i = 0; i < num_reqs; ++i) {
            rocksdb::FSReadRequest& req = reqs[i];
            req.status = Read(req.offset, req.len, options, &req.result, 
                            req.scratch, dbg);
        }

        // Print the result
        std::cout << "MultiRead " << num_reqs << " requests from GDS" << std::endl;
        
        return rocksdb::IOStatus::OK();
    }

    // Indicate that we use direct I/O
    bool use_direct_io() const override { 
        return true; 
    }

    // Return GDS alignment requirement
    size_t GetRequiredBufferAlignment() const override { 
        return options_.GDSAlignment; 
    }

    // Get file size
    rocksdb::IOStatus GetFileSize(uint64_t* result) override {
        if (result == nullptr) {
            return rocksdb::IOStatus::InvalidArgument("result pointer is null");
        }
        *result = file_size_;
        return rocksdb::IOStatus::OK();
    }
    
    // Get file name
    const std::string& GetFileName() const {
        return filename_;
    }

    // Cache invalidation (not applicable for GDS direct I/O)
    rocksdb::IOStatus InvalidateCache(size_t offset, size_t length) override {
        (void)offset;
        (void)length;
        return rocksdb::IOStatus::OK();  // No-op for direct I/O
    }
    
    // Cache management methods
    void ClearCache() {
        if (gpu_cache_) {
            gpu_cache_->Clear();
        }
    }
    
    void GetCacheStats(size_t* hits, size_t* misses, size_t* cached_blocks) const {
        if (gpu_cache_) {
            gpu_cache_->GetStats(hits, misses, cached_blocks);
        }
    }
    
    void PrintCacheStats() const {
        if (gpu_cache_) {
            size_t hits, misses, blocks;
            gpu_cache_->GetStats(&hits, &misses, &blocks);
            float hit_rate = (hits + misses) > 0 ? 
                           (float)hits / (hits + misses) * 100.0f : 0.0f;
            std::cout << "[GDS Cache] Hits: " << hits 
                     << " Misses: " << misses 
                     << " Hit Rate: " << hit_rate << "% "
                     << " Cached Blocks: " << blocks << std::endl;
        }
    }

private:
    std::string filename_;
    GDSOptions options_;
    
    mutable int file_fd_;
    mutable CUfileHandle_t cu_file_handle_;
    mutable bool gds_initialized_;
    uint64_t file_size_;
    
    mutable void* gpu_buffer_;
    mutable size_t gpu_buffer_size_;
    
    // GPU-side block cache
    std::shared_ptr<GPUBlockCache> gpu_cache_;

    IOStatus EnsureGPUBuffer(size_t required_size) const {
        // Align to GDS requirements
        size_t aligned_size = (required_size + options_.GDSAlignment - 1) & 
                             ~(options_.GDSAlignment - 1);
        
        // Check if current buffer is sufficient
        if (gpu_buffer_ != nullptr && gpu_buffer_size_ >= aligned_size) {
            return IOStatus::OK();
        }
        
        // Free existing buffer if any
        if (gpu_buffer_ != nullptr) {
            cudaFree(gpu_buffer_);
            gpu_buffer_ = nullptr;
            gpu_buffer_size_ = 0;
        }
        
        // Allocate new buffer
        cudaError_t err = cudaMalloc(&gpu_buffer_, aligned_size);
        if (err != cudaSuccess) {
            gpu_buffer_ = nullptr;
            gpu_buffer_size_ = 0;
            return IOStatus::IOError("Failed to allocate GPU buffer: " + 
                                    std::string(cudaGetErrorString(err)));
        }
        
        gpu_buffer_size_ = aligned_size;
        return IOStatus::OK();
    }

    IOStatus InitializeGDS() {
        // 1. Initialize CUDA context
        cudaError_t cuda_err = cudaSetDevice(0);
        if (cuda_err != cudaSuccess) {
            return IOStatus::CudaError("Failed to set CUDA device: " + 
                                      std::string(cudaGetErrorString(cuda_err)));
        }
        
        // 2. Open file with O_DIRECT for GDS compatibility (read-only)
        file_fd_ = open(filename_.c_str(), O_RDONLY | O_DIRECT);
        if (file_fd_ < 0) {
            return IOStatus::IOError("Failed to open file for GDS reading: " + 
                                    std::string(strerror(errno)));
        }
        
        // 3. Get file size
        struct stat file_stat;
        if (fstat(file_fd_, &file_stat) == 0) {
            file_size_ = file_stat.st_size;
        } else {
            close(file_fd_);
            file_fd_ = -1;
            return IOStatus::IOError("Failed to get file size: " + 
                                    std::string(strerror(errno)));
        }
        
        // 4. Register file with cuFile
        CUfileDescr_t file_descr;
        memset(&file_descr, 0, sizeof(file_descr));
        file_descr.handle.fd = file_fd_;
        file_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
        
        CUfileError_t cu_err = cuFileHandleRegister(&cu_file_handle_, &file_descr);
        if (cu_err.err != CU_FILE_SUCCESS) {
            close(file_fd_);
            file_fd_ = -1;
            return IOStatus::IOError("Failed to register file with cuFile: " + 
                                    std::to_string(cu_err.err));
        }
        
        // 5. Allocate initial GPU buffer
        IOStatus buffer_status = EnsureGPUBuffer(options_.DefaultGPUBufferSize);
        if (!buffer_status.ok()) {
            cuFileHandleDeregister(cu_file_handle_);
            close(file_fd_);
            file_fd_ = -1;
            return buffer_status;
        }
        
        gds_initialized_ = true;
        return IOStatus::OK();
    }
};

} // namespace gds

} // namespace mlkv_plus

// EOF