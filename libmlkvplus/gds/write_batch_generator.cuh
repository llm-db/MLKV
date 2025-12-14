#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <cuda.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>

#include "utils.cuh"


namespace mlkv_plus {

namespace gds {

// WriteBatch format constants (matching RocksDB)
constexpr size_t kWriteBatchHeader = 12;  // 8-byte sequence + 4-byte count
constexpr uint8_t kTypeValue = 0x1;       // Standard Put operation

// CUDA device functions for encoding
__device__ __host__ __forceinline__ void PutFixed32(char* dst, uint32_t value) {
    dst[0] = static_cast<char>(value & 0xff);
    dst[1] = static_cast<char>((value >> 8) & 0xff);
    dst[2] = static_cast<char>((value >> 16) & 0xff);
    dst[3] = static_cast<char>((value >> 24) & 0xff);
}

__device__ __host__ __forceinline__ void PutFixed64(char* dst, uint64_t value) {
    dst[0] = static_cast<char>(value & 0xff);
    dst[1] = static_cast<char>((value >> 8) & 0xff);
    dst[2] = static_cast<char>((value >> 16) & 0xff);
    dst[3] = static_cast<char>((value >> 24) & 0xff);
    dst[4] = static_cast<char>((value >> 32) & 0xff);
    dst[5] = static_cast<char>((value >> 40) & 0xff);
    dst[6] = static_cast<char>((value >> 48) & 0xff);
    dst[7] = static_cast<char>((value >> 56) & 0xff);
}

__device__ __host__ __forceinline__ int PutVarint32(char* dst, uint32_t value) {
    char* ptr = dst;
    static const int B = 128;
    if (value < (1 << 7)) {
        *(ptr++) = value;
    } else if (value < (1 << 14)) {
        *(ptr++) = value | B;
        *(ptr++) = value >> 7;
    } else if (value < (1 << 21)) {
        *(ptr++) = value | B;
        *(ptr++) = (value >> 7) | B;
        *(ptr++) = value >> 14;
    } else if (value < (1 << 28)) {
        *(ptr++) = value | B;
        *(ptr++) = (value >> 7) | B;
        *(ptr++) = (value >> 14) | B;
        *(ptr++) = value >> 21;
    } else {
        *(ptr++) = value | B;
        *(ptr++) = (value >> 7) | B;
        *(ptr++) = (value >> 14) | B;
        *(ptr++) = (value >> 21) | B;
        *(ptr++) = value >> 28;
    }
    return ptr - dst;
}

__device__ __forceinline__ int PutLengthPrefixedSlice(char* dst, const char* data, size_t size) {
    int varint_len = PutVarint32(dst, static_cast<uint32_t>(size));
    memcpy(dst + varint_len, data, size);
    return varint_len + size;
}

// Template structure to hold typed key-value pair on GPU
template<typename Key, typename Value>
struct TypedGPUKeyValue {
    const Key* key;
    const Value* value;
    size_t value_array_size;  // For array values like vectors
};






// Template CUDA kernel to calculate record sizes for typed key-value pairs
template<typename Key, typename Value>
__global__ void CalculateTypedRecordSizesKernel(
    const TypedGPUKeyValue<Key, Value>* kv_pairs,
    size_t num_pairs,
    size_t* record_sizes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pairs) return;

    const auto& kv = kv_pairs[idx];
    
    // Calculate sizes for: type(1) + key_len_varint + key + value_len_varint + value
    size_t key_size = sizeof(Key);
    size_t value_size = sizeof(Value) * kv.value_array_size;
    
    size_t key_len_varint_size = 1;
    if (key_size >= (1 << 7)) key_len_varint_size++;
    if (key_size >= (1 << 14)) key_len_varint_size++;
    if (key_size >= (1 << 21)) key_len_varint_size++;
    if (key_size >= (1 << 28)) key_len_varint_size++;
    
    size_t value_len_varint_size = 1;
    if (value_size >= (1 << 7)) value_len_varint_size++;
    if (value_size >= (1 << 14)) value_len_varint_size++;
    if (value_size >= (1 << 21)) value_len_varint_size++;
    if (value_size >= (1 << 28)) value_len_varint_size++;
    
    record_sizes[idx] = 1 + key_len_varint_size + key_size + value_len_varint_size + value_size;
}

// Combined CUDA kernel to initialize TypedGPUKeyValue array and calculate record sizes
template<typename Key, typename Value>
__global__ void InitializeAndCalculateKernel(
    TypedGPUKeyValue<Key, Value>* kv_pairs,
    const Key* d_keys,
    const Value* d_values,
    size_t num_pairs,
    size_t value_array_size,
    size_t* record_sizes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pairs) return;

    // Initialize typed key-value pair
    kv_pairs[idx].key = &d_keys[idx];
    kv_pairs[idx].value = &d_values[idx * value_array_size];
    kv_pairs[idx].value_array_size = value_array_size;
    
    // Calculate record size for this pair
    // Calculate sizes for: type(1) + key_len_varint + key + value_len_varint + value
    size_t key_size = sizeof(Key);
    size_t value_size = sizeof(Value) * value_array_size;
    
    size_t key_len_varint_size = 1;
    if (key_size >= (1 << 7)) key_len_varint_size++;
    if (key_size >= (1 << 14)) key_len_varint_size++;
    if (key_size >= (1 << 21)) key_len_varint_size++;
    if (key_size >= (1 << 28)) key_len_varint_size++;
    
    size_t value_len_varint_size = 1;
    if (value_size >= (1 << 7)) value_len_varint_size++;
    if (value_size >= (1 << 14)) value_len_varint_size++;
    if (value_size >= (1 << 21)) value_len_varint_size++;
    if (value_size >= (1 << 28)) value_len_varint_size++;
    
    record_sizes[idx] = 1 + key_len_varint_size + key_size + value_len_varint_size + value_size;
}



// Combined CUDA kernel to generate WriteBatch header and records from typed key-value pairs
template<typename Key, typename Value>
__global__ void GenerateTypedWriteBatchKernel(
    const TypedGPUKeyValue<Key, Value>* kv_pairs,
    size_t num_pairs,
    char* output_buffer,
    const size_t* record_offsets,
    uint64_t sequence_number
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // First thread in first block handles header generation
    if (idx == 0) {
        // Clear header area
        for (int i = 0; i < kWriteBatchHeader; ++i) {
            output_buffer[i] = 0;
        }
        
        // Write sequence number (8 bytes, little-endian)
        PutFixed64(output_buffer, sequence_number);
        
        // Write count (4 bytes, little-endian)
        PutFixed32(output_buffer + 8, static_cast<uint32_t>(num_pairs));
    }
    
    // All threads handle record generation
    if (idx < num_pairs) {
        const auto& kv = kv_pairs[idx];
        char* record_start = output_buffer + record_offsets[idx];
        char* ptr = record_start;

        // Write record type
        *ptr++ = kTypeValue;

        // Write key length and key data
        size_t key_size = sizeof(Key);
        int key_len_varint_size = PutVarint32(ptr, static_cast<uint32_t>(key_size));
        ptr += key_len_varint_size;
        memcpy(ptr, kv.key, key_size);
        ptr += key_size;

        // Write value length and value data
        size_t value_size = sizeof(Value) * kv.value_array_size;
        int value_len_varint_size = PutVarint32(ptr, static_cast<uint32_t>(value_size));
        ptr += value_len_varint_size;
        memcpy(ptr, kv.value, value_size);
    }
}

class GPUWriteBatchGenerator {
public:
    GPUWriteBatchGenerator(size_t max_batch_size = 1024)
        : max_batch_size_(max_batch_size) {
        
        // Allocate GPU memory for key-value pairs and record sizes
        MLKV_CUDA_CHECK(cudaMalloc(&d_record_sizes_, max_batch_size_ * sizeof(size_t)));
        MLKV_CUDA_CHECK(cudaMalloc(&d_record_offsets_, max_batch_size_ * sizeof(size_t)));
        
    }

    ~GPUWriteBatchGenerator() {
        if (d_record_sizes_) cudaFree(d_record_sizes_);
        if (d_record_offsets_) cudaFree(d_record_offsets_);
        if (d_output_buffer_) cudaFree(d_output_buffer_);
    }

    // Template method to generate WriteBatch from typed GPU key-value pairs
    template<typename Key, typename Value>
    IOStatus GenerateTypedWriteBatch(const Key* d_keys, const Value* d_values, 
                                   size_t num_pairs, size_t value_array_size, 
                                   uint64_t latest_sequence_number) {
        if (num_pairs == 0) {
            return IOStatus::InvalidParam("No key-value pairs to process");
        }

        // Allocate GPU memory for typed key-value pairs
        TypedGPUKeyValue<Key, Value>* d_typed_kv_pairs;
        MLKV_CUDA_CHECK(cudaMalloc(&d_typed_kv_pairs, num_pairs * sizeof(TypedGPUKeyValue<Key, Value>)));
        
        // Initialize typed key-value pairs and calculate record sizes in a single kernel
        dim3 block_size(256);
        dim3 grid_size((num_pairs + block_size.x - 1) / block_size.x);
        
        InitializeAndCalculateKernel<Key, Value><<<grid_size, block_size>>>(
            d_typed_kv_pairs, d_keys, d_values, num_pairs, value_array_size, d_record_sizes_);
        MLKV_CUDA_CHECK(cudaDeviceSynchronize());

        // Calculate offsets (prefix sum) directly on GPU using Thrust
        thrust::device_ptr<size_t> d_sizes_ptr(d_record_sizes_);
        thrust::device_ptr<size_t> d_offsets_ptr(d_record_offsets_);
        
        // Perform exclusive scan (prefix sum) with initial value as header size
        thrust::exclusive_scan(d_sizes_ptr, d_sizes_ptr + num_pairs, d_offsets_ptr, kWriteBatchHeader);
        
        // Calculate total batch size on CPU (more efficient than GPU kernel + memcpy)
        size_t total_records_size = thrust::reduce(d_sizes_ptr, d_sizes_ptr + num_pairs);
        total_batch_size_ = kWriteBatchHeader + total_records_size;

        // Allocate output buffer
        if (d_output_buffer_) MLKV_CUDA_CHECK(cudaFree(d_output_buffer_));
        MLKV_CUDA_CHECK(cudaMalloc(&d_output_buffer_, total_batch_size_));

        // Generate WriteBatch header and records in a single kernel call
        GenerateTypedWriteBatchKernel<Key, Value><<<grid_size, block_size>>>(
            d_typed_kv_pairs, num_pairs, d_output_buffer_, d_record_offsets_, latest_sequence_number);
        MLKV_CUDA_CHECK(cudaDeviceSynchronize());

        // Cleanup
        MLKV_CUDA_CHECK(cudaFree(d_typed_kv_pairs));

        return IOStatus::OK();
    }

    // Clear current batch
    void Clear() {
        // do nothing currently
    }


    // Get maximum batch size
    size_t GetMaxBatchSize() const { return max_batch_size_; }

    // Get the generated WriteBatch data (after calling GenerateWriteBatch)
    const char* GetWriteBatchData() const {
        return d_output_buffer_;
    }

    // Get size of the generated WriteBatch (after calling GenerateWriteBatch)
    size_t GetWriteBatchSize() const {
        return total_batch_size_;
    }

private:
    size_t max_batch_size_;
    size_t total_batch_size_ = 0;

    // GPU memory
    size_t* d_record_sizes_ = nullptr;
    size_t* d_record_offsets_ = nullptr;
    char* d_output_buffer_ = nullptr;
};


} // namespace gds

} // namespace mlkv_plus