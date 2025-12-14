#pragma once

#include <cassert>
#include <cstddef>
#include <cuda_runtime.h>
#include <stdint.h>
#include "utils.cuh"

namespace mlkv_plus {
namespace gds {


constexpr bool kOpenMemWarnings = true;
constexpr int kKeyBufferSize = 512;

// DeviceSlice: Lightweight data view on GPU (non-owning)
// Can be constructed on both host and device sides
class DeviceSlice {
public:
    const char* data_;
    size_t      size_;

    __host__ __device__ __forceinline__ DeviceSlice() : data_(nullptr), size_(0) {}
    __host__ __device__ __forceinline__ DeviceSlice(const char* d, size_t n) : data_(d), size_(n) {}

    __host__ __device__ __forceinline__ const char* data() const { return data_; }
    __host__ __device__ __forceinline__ size_t size() const { return size_; }
    __host__ __device__ __forceinline__ bool empty() const { return size_ == 0; }

    __device__ __forceinline__ char operator[](size_t n) const {
        assert(n < size_);
        return data_[n];
    }

    __host__ __device__ __forceinline__ void clear() {
        data_ = nullptr;
        size_ = 0;
    }
};

// DeviceBlockContents: GPU-side block contents representation
// 
// Corresponds to CPU-side BlockContents, but runs entirely on GPU
// - Manages GPU memory ownership
// - Supports move semantics to avoid unnecessary copies
// - Supports two modes: owning vs non-owning (e.g., GDS direct mapping)
// - Can be constructed on both host and device sides
//
// Use cases:
// 1. GDS direct read to GPU memory (owns memory)
// 2. Reference to existing GPU memory block (non-owning)
class DeviceBlockContents {
public:
    DeviceSlice data;           // Block data view (without trailer)
    char*       allocation;     // GPU memory pointer (nullptr = non-owning)
    size_t      alloc_size;     // Allocated size (for memory management)

    // Default constructor: empty block
    __host__ __device__ __forceinline__ DeviceBlockContents() 
        : data(), allocation(nullptr), alloc_size(0) {}

    // Constructor: non-owning (reference only, e.g., mmap or GDS mapped region)
    __host__ __device__ __forceinline__ explicit DeviceBlockContents(const DeviceSlice& _data)
        : data(_data), allocation(nullptr), alloc_size(0) {}

    // Constructor: owning (takes ownership)
    __host__ __device__ __forceinline__ DeviceBlockContents(char* _data, size_t _size)
        : data(_data, _size), allocation(_data), alloc_size(_size) {}

    // Move constructor: transfer ownership
    __host__ __device__ __forceinline__ DeviceBlockContents(DeviceBlockContents&& other) noexcept
        : data(other.data), allocation(other.allocation), alloc_size(other.alloc_size) {
        other.data.clear();
        other.allocation = nullptr;
        other.alloc_size = 0;
    }

    // Move assignment: transfer ownership
    __host__ __device__ __forceinline__ DeviceBlockContents& operator=(DeviceBlockContents&& other) noexcept {
        if (this != &other) {
            data = other.data;
            allocation = other.allocation;
            alloc_size = other.alloc_size;

            other.data.clear();
            other.allocation = nullptr;
            other.alloc_size = 0;
        }
        return *this;
    }

    // Delete copy to avoid accidental memory duplication
    DeviceBlockContents(const DeviceBlockContents&) = delete;
    DeviceBlockContents& operator=(const DeviceBlockContents&) = delete;

    // Query: whether this object owns the memory
    __host__ __device__ __forceinline__ bool own_bytes() const {
        return allocation != nullptr;
    }

    // Query: usable size
    __host__ __device__ __forceinline__ size_t usable_size() const {
        return allocation ? alloc_size : 0;
    }

    // Query: approximate memory usage
    __host__ __device__ __forceinline__ size_t ApproximateMemoryUsage() const {
        return usable_size() + sizeof(*this);
    }

    // Clear: release ownership without freeing memory (external memory manager is responsible)
    __host__ __device__ __forceinline__ void clear() {
        data.clear();
        allocation = nullptr;
        alloc_size = 0;
    }

    // Release ownership: return memory pointer, caller is responsible for deallocation
    __host__ __device__ __forceinline__ char* release() {
        char* ptr = allocation;
        clear();
        return ptr;
    }

    // Query: whether the block is empty
    __host__ __device__ __forceinline__ bool empty() const {
        return data.empty();
    }
};

// DeviceBlockHandle: GPU-side block handle (location info in file)
// Can be constructed on both host and device sides
struct DeviceBlockHandle {
    uint64_t offset_;  // Block offset in file
    uint64_t size_;    // Block size (without trailer)

    __host__ __device__ __forceinline__ DeviceBlockHandle() 
        : offset_(~uint64_t(0)), size_(~uint64_t(0)) {}

    __host__ __device__ __forceinline__ DeviceBlockHandle(uint64_t offset, uint64_t size)
        : offset_(offset), size_(size) {}

    __host__ __device__ __forceinline__ uint64_t offset() const { return offset_; }
    __host__ __device__ __forceinline__ uint64_t size() const { return size_; }

    __host__ __device__ __forceinline__ void set_offset(uint64_t offset) { offset_ = offset; }
    __host__ __device__ __forceinline__ void set_size(uint64_t size) { size_ = size; }

    __host__ __device__ __forceinline__ bool IsNull() const { 
        return offset_ == 0 && size_ == 0; 
    }
};



struct GDSDataBlockIterDevicePinnedBuffer {
    bool is_allocated_ = false;
    int* h_valid_flag_;
    int* d_valid_flag_;
    char* d_key_buffer_;
    int* d_key_length_;
    char* h_key_buffer_;
    int* h_key_length_;
    int* h_value_length_;
    int* d_value_length_;

    const char** d_d_value_ptr_;
    const char** h_d_value_ptr_;
};


class GDSDataBlockIterDevicePinnedBufferManager {
public:
    GDSDataBlockIterDevicePinnedBufferManager() {};
    ~GDSDataBlockIterDevicePinnedBufferManager() {};
    
    static GDSDataBlockIterDevicePinnedBuffer* Allocate() {
        GDSDataBlockIterDevicePinnedBuffer* buffer = new GDSDataBlockIterDevicePinnedBuffer();
        buffer->is_allocated_ = true;
    
        MLKV_CUDA_CHECK(cudaHostAlloc((void**)&buffer->h_valid_flag_, sizeof(int), cudaHostAllocMapped));
        MLKV_CUDA_CHECK(cudaHostGetDevicePointer(&buffer->d_valid_flag_, buffer->h_valid_flag_, 0));
        MLKV_CUDA_CHECK(cudaHostAlloc((void**)&buffer->h_key_buffer_, sizeof(char) * kKeyBufferSize, cudaHostAllocMapped));
        MLKV_CUDA_CHECK(cudaHostGetDevicePointer(&buffer->d_key_buffer_, buffer->h_key_buffer_, 0));
        MLKV_CUDA_CHECK(cudaHostAlloc((void**)&buffer->h_key_length_, sizeof(int), cudaHostAllocMapped));
        MLKV_CUDA_CHECK(cudaHostGetDevicePointer(&buffer->d_key_length_, buffer->h_key_length_, 0));
        MLKV_CUDA_CHECK(cudaHostAlloc((void**)&buffer->h_value_length_, sizeof(int), cudaHostAllocMapped));
        MLKV_CUDA_CHECK(cudaHostGetDevicePointer(&buffer->d_value_length_, buffer->h_value_length_, 0));
        MLKV_CUDA_CHECK(cudaHostAlloc((void**)&buffer->h_d_value_ptr_, sizeof(const char*), cudaHostAllocMapped));
        MLKV_CUDA_CHECK(cudaHostGetDevicePointer(&buffer->d_d_value_ptr_, buffer->h_d_value_ptr_, 0));
        return buffer;
    
    
    }
    
    static void Free(GDSDataBlockIterDevicePinnedBuffer* buffer) {
        if (!buffer->is_allocated_) {
            return;
        }
        buffer->is_allocated_ = false;
        cudaFreeHost(buffer->h_valid_flag_);
        cudaFreeHost(buffer->h_key_buffer_);
        cudaFreeHost(buffer->h_key_length_);
        delete buffer;
    }
    

    
};

}  // namespace gds
}  // namespace mlkv_plus
