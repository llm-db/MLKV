#include "gds_block.cuh"
#include "utils.cuh"
#include <cuda_runtime.h>
#include <iostream>

namespace mlkv_plus {
namespace gds {

// ============================================================================
// GPU Device Utility Functions Implementation
// ============================================================================


namespace device {

__device__ __forceinline__ const char* GetVarint32Ptr(
    const char* p, const char* limit, uint32_t* value) {
    if (p >= limit) return nullptr;
    
    uint32_t result = 0;
    for (uint32_t shift = 0; shift <= 28 && p < limit; shift += 7) {
        uint32_t byte = static_cast<unsigned char>(*p);
        p++;
        if (byte & 128) {
            result |= ((byte & 127) << shift);
        } else {
            result |= (byte << shift);
            *value = result;
            return p;
        }
    }
    return nullptr;
}

__device__ __forceinline__ const char* DecodeEntry(
    const char* p, const char* limit,
    uint32_t* shared, uint32_t* non_shared, uint32_t* value_length) {
    
    if (limit - p < 3) return nullptr;
    
    *shared = static_cast<unsigned char>(p[0]);
    *non_shared = static_cast<unsigned char>(p[1]);
    *value_length = static_cast<unsigned char>(p[2]);
    
    if ((*shared | *non_shared | *value_length) < 128) {
        // Fast path: all values fit in one byte
        p += 3;
    } else {
        // Slow path: use varint decoding
        if ((p = GetVarint32Ptr(p, limit, shared)) == nullptr) return nullptr;
        if ((p = GetVarint32Ptr(p, limit, non_shared)) == nullptr) return nullptr;
        if ((p = GetVarint32Ptr(p, limit, value_length)) == nullptr) return nullptr;
    }
    
    if (static_cast<uint32_t>(limit - p) < (*non_shared + *value_length)) {
        return nullptr;
    }
    return p;
}

__device__ __forceinline__ uint32_t DecodeFixed32(const char* ptr) {
    const unsigned char* p = reinterpret_cast<const unsigned char*>(ptr);
    return (static_cast<uint32_t>(p[0])) |
           (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) |
           (static_cast<uint32_t>(p[3]) << 24);
}

__device__ __forceinline__ uint32_t GetRestartPoint(
    const char* data, uint32_t restarts, uint32_t index) {
    return DecodeFixed32(data + restarts + index * sizeof(uint32_t));
}

// Simple bytewise comparison (for non-internal keys)
__device__ __forceinline__ int CompareSlices(
    const char* a, size_t a_len, const char* b, size_t b_len) {
    size_t min_len = (a_len < b_len) ? a_len : b_len;
    for (size_t i = 0; i < min_len; i++) {
        if (static_cast<unsigned char>(a[i]) < static_cast<unsigned char>(b[i])) return -1;
        if (static_cast<unsigned char>(a[i]) > static_cast<unsigned char>(b[i])) return 1;
    }
    if (a_len < b_len) return -1;
    if (a_len > b_len) return 1;
    return 0;
}

// Internal Key Comparator - handles user_key + sequence + type format
// Internal key format: [user_key][sequence(7 bytes)][type(1 byte)]
// Comparison: user_key ascending, then sequence DESCENDING, then type
__device__ __forceinline__ int CompareInternalKeys(
    const char* a, size_t a_len, const char* b, size_t b_len) {
    
    // Internal keys must be at least 8 bytes (sequence + type)
    constexpr size_t kInternalKeyTail = 8;
    
    if (a_len < kInternalKeyTail || b_len < kInternalKeyTail) {
        // Fallback to simple comparison if keys are too short
        return CompareSlices(a, a_len, b, b_len);
    }
    
    // Extract user_key portions
    size_t a_user_len = a_len - kInternalKeyTail;
    size_t b_user_len = b_len - kInternalKeyTail;
    
    // Compare user_key parts (ascending order)
    size_t min_user_len = (a_user_len < b_user_len) ? a_user_len : b_user_len;
    for (size_t i = 0; i < min_user_len; i++) {
        if (static_cast<unsigned char>(a[i]) < static_cast<unsigned char>(b[i])) return -1;
        if (static_cast<unsigned char>(a[i]) > static_cast<unsigned char>(b[i])) return 1;
    }
    
    // If one user_key is prefix of another
    if (a_user_len < b_user_len) return -1;
    if (a_user_len > b_user_len) return 1;
    
    // User keys are equal, compare sequence numbers (DESCENDING order!)
    // Sequence is stored in last 8 bytes: 7 bytes sequence + 1 byte type
    // We need to extract and compare sequence in descending order
    
    // Extract sequence + type (last 8 bytes) as uint64_t
    uint64_t a_num = 0, b_num = 0;
    for (size_t i = 0; i < kInternalKeyTail; i++) {
        a_num |= (static_cast<uint64_t>(static_cast<unsigned char>(a[a_user_len + i])) << (i * 8));
        b_num |= (static_cast<uint64_t>(static_cast<unsigned char>(b[b_user_len + i])) << (i * 8));
    }
    
    // Sequence is in descending order, so we reverse the comparison
    if (a_num > b_num) return -1;  // Note: reversed!
    if (a_num < b_num) return 1;   // Note: reversed!
    return 0;
}

} // namespace device

// ============================================================================
// GPU Kernel: Initialize Block Metadata
// ============================================================================

// Parse block footer and calculate metadata on GPU
// This replaces the CPU-side InitializeRestartInfo() to avoid GPU->CPU copies
__global__ void InitializeBlockMetadataKernel(
    const char* d_data,
    uint32_t block_size,
    DeviceBlockMetadata* d_metadata) {
    
    // Only one thread does the work (simple parsing, not parallelizable)
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }
    
    // Check block size
    if (block_size < sizeof(uint32_t)) {
        // Block too small - set error state
        d_metadata->num_restarts_ = 0;
        d_metadata->restart_offset_ = block_size;
        d_metadata->index_type_ = 0;  // kDataBlockBinarySearch
        d_metadata->block_size_ = block_size;
        return;
    }
    
    // Read block footer from last 4 bytes
    uint32_t block_footer = device::DecodeFixed32(d_data + block_size - sizeof(uint32_t));
    
    // Decode footer based on block size
    constexpr uint32_t kMaxBlockSizeSupportedByHashIndex = 1u << 16;  // 64KiB
    constexpr uint32_t kNumRestartsMask = 0x7FFFFFFF;  // Mask for bits 0-30
    constexpr int kDataBlockIndexTypeBitShift = 31;
    
    uint32_t num_restarts;
    uint32_t index_type;
    
    if (block_size > kMaxBlockSizeSupportedByHashIndex) {
        // Large blocks don't use packed format
        num_restarts = block_footer;
        index_type = 0;  // kDataBlockBinarySearch
    } else {
        // Unpack num_restarts and index type
        num_restarts = block_footer & kNumRestartsMask;
        
        // Extract index type from bit 31
        if (block_footer & (1u << kDataBlockIndexTypeBitShift)) {
            index_type = 1;  // kDataBlockBinaryAndHash
        } else {
            index_type = 0;  // kDataBlockBinarySearch
        }
    }
    
    // Calculate restart array offset
    uint32_t restart_offset;
    if (num_restarts > 0) {
        restart_offset = block_size - (1 + num_restarts) * sizeof(uint32_t);
    } else {
        restart_offset = block_size;
    }
    
    // Write results to output structure
    d_metadata->num_restarts_ = num_restarts;
    d_metadata->restart_offset_ = restart_offset;
    d_metadata->index_type_ = index_type;
    d_metadata->block_size_ = block_size;
    
    // Validate index type on GPU using assert (zero DtoH cost!)
    // GDS GPU implementation only supports binary search, not hash index
    // If assertion fails, cudaGetLastError() will detect it on host
    assert(index_type != 1 && 
           "[GDS ERROR] Block uses kDataBlockBinaryAndHash (hash index) which is NOT supported!");
}

// ============================================================================
// GPU Kernel: Initialize Iterator State
// ============================================================================

// Initialize iterator state from block metadata (pure GPU operation)
// This avoids GPU->CPU copies by reading metadata directly on GPU
__global__ void InitializeIteratorStateKernel(
    GDSDataBlockIterState* d_state,
    const char* d_data,
    const DeviceBlockMetadata* d_metadata) {
    
    // Only one thread does the work
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }
    
    // Read metadata from GPU memory
    uint32_t num_restarts = d_metadata->num_restarts_;
    uint32_t restart_offset = d_metadata->restart_offset_;
    uint32_t block_size = d_metadata->block_size_;
    
    // Initialize iterator state
    d_state->data_ = d_data;
    d_state->size_ = block_size;
    d_state->num_restarts_ = num_restarts;
    d_state->restarts_ = restart_offset;
    d_state->current_ = restart_offset;
    d_state->restart_index_ = 0;
    d_state->error_code_ = 0;
}

// ============================================================================
// GDSDataBlockIterState Implementation
// ============================================================================

__host__ __device__ __forceinline__ GDSDataBlockIterState::GDSDataBlockIterState()
    : data_(nullptr), size_(0), num_restarts_(0), restarts_(0),
      current_(0), restart_index_(0), error_code_(0) {}

// ============================================================================
// CUDA Kernel Implementations
// ============================================================================

__global__ void SeekToFirstKernel(GDSDataBlockIterState* state, int* valid_flag_, int* key_length_, char* key_buffer_, int* value_length_, const char** value_ptr_) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    
    state->restart_index_ = 0;
    state->current_ = device::GetRestartPoint(state->data_, state->restarts_, 0);
    *key_length_ = 0;
    
    // Parse first entry
    const char* p = state->data_ + state->current_;
    const char* limit = state->data_ + state->restarts_;
    
    uint32_t shared, non_shared, value_length;
    p = device::DecodeEntry(p, limit, &shared, &non_shared, &value_length);
    
    if (p == nullptr || shared != 0) {
        // state->valid_ = false;
        *valid_flag_ = 0;
        state->error_code_ = 1;
        return;
    }
    
    // Copy key (no sharing for restart point)
    for (uint32_t i = 0; i < non_shared && i < kKeyBufferSize * sizeof(key_buffer_); i++) {
        key_buffer_[i] = p[i];
    }
    *key_length_ = non_shared;
    *value_ptr_ = p + non_shared;
    *value_length_ = value_length;
    *valid_flag_ = 1;
    state->error_code_ = 0;
}

__global__ void SeekKernel(
    GDSDataBlockIterState* state,
    int* valid_flag_,
    const char* target_key,
    uint32_t target_len,
    int* key_length_,
    char* key_buffer_,
    int* value_length_,
    const char** value_ptr_) {
    
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    
    int64_t left = -1;
    int64_t right = state->num_restarts_ - 1;
    
    // Binary search in restart points
    while (left < right) {
        int64_t mid = left + (right - left + 1) / 2;
        uint32_t region_offset = device::GetRestartPoint(state->data_, state->restarts_, mid);
        
        const char* p = state->data_ + region_offset;
        const char* limit = state->data_ + state->restarts_;
        
        uint32_t shared, non_shared, value_length;
        p = device::DecodeEntry(p, limit, &shared, &non_shared, &value_length);
        
        if (p == nullptr || shared != 0) {
            // state->valid_ = false;
            *valid_flag_ = 0;
            state->error_code_ = 1;
            return;
        }
        
        int cmp = device::CompareInternalKeys(p, non_shared, target_key, target_len);
        
        if (cmp < 0) {
            left = mid;
        } else if (cmp > 0) {
            right = mid - 1;
        } else {
            // Exact match at restart point
            left = right = mid;
            break;
        }
    }
    
    uint32_t restart_idx = (left == -1) ? 0 : static_cast<uint32_t>(left);
    
    // Seek to restart point and linear scan to target
    state->restart_index_ = restart_idx;
    state->current_ = device::GetRestartPoint(state->data_, state->restarts_, restart_idx);
    *key_length_ = 0;
    
    // Linear scan within restart interval to find first key >= target
    while (state->current_ < state->restarts_) {
        const char* p = state->data_ + state->current_;
        const char* limit = state->data_ + state->restarts_;
        
        uint32_t shared, non_shared, value_length;
        const char* key_ptr = device::DecodeEntry(p, limit, &shared, &non_shared, &value_length);
        
        if (key_ptr == nullptr) {
            // state->valid_ = false;
            *valid_flag_ = 0;
            state->error_code_ = 1;
            return;
        }
        
        // Reconstruct key using delta encoding
        if (shared == 0) {
            // No sharing, copy directly
            for (uint32_t i = 0; i < non_shared && i < sizeof(key_buffer_) * kKeyBufferSize; i++) {
                key_buffer_[i] = key_ptr[i];
            }
            *key_length_ = non_shared;
        } else {
            // Delta encoding: keep first `shared` bytes, append non_shared bytes
            for (uint32_t i = 0; i < non_shared && (shared + i) < sizeof(key_buffer_) * kKeyBufferSize; i++) {
                key_buffer_[shared + i] = key_ptr[i];
            }
            *key_length_ = shared + non_shared;
        }
        
        *value_ptr_ = key_ptr + non_shared;
        *value_length_ = value_length;
        
        // Compare with target (use internal key comparator)
        int cmp = device::CompareInternalKeys(key_buffer_, *key_length_, target_key, target_len);
        
        if (cmp >= 0) {
            // Found first key >= target
            // state->valid_ = true;
            *valid_flag_ = 1;
            state->error_code_ = 0;
            return;
        }
        
        // Move to next entry
        state->current_ = static_cast<uint32_t>((*value_ptr_ + *value_length_) - state->data_);
    }
    
    // Reached end of block without finding key >= target
    state->current_ = state->restarts_;
    // state->valid_ = false;
    *valid_flag_ = 0;
    state->error_code_ = 0;
}

__global__ void NextKernel(GDSDataBlockIterState* state, 
    int* valid_flag_, 
    int* key_length_, 
    char* key_buffer_,
    int* value_length_,
    const char** value_ptr_) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    if (!*valid_flag_) return;
    
    // Move to next entry
    state->current_ = static_cast<uint32_t>((*value_ptr_ + *value_length_) - state->data_);
    
    if (state->current_ >= state->restarts_) {
        // state->valid_ = false;
        *valid_flag_ = 0;
        state->error_code_ = 0;
        return;
    }
    
    const char* p = state->data_ + state->current_;
    const char* limit = state->data_ + state->restarts_;
    
    uint32_t shared, non_shared, value_length;
    const char* key_ptr = device::DecodeEntry(p, limit, &shared, &non_shared, &value_length);
    
    if (key_ptr == nullptr) {
        // state->valid_ = false;
        *valid_flag_ = 0;
        state->error_code_ = 1;
        return;
    }
    
    // Reconstruct key using delta encoding
    if (shared == 0) {
        for (uint32_t i = 0; i < non_shared && i < kKeyBufferSize * sizeof(char); i++) {
            key_buffer_[i] = key_ptr[i];
        }
        *key_length_ = non_shared;
    } else {
        for (uint32_t i = 0; i < non_shared && (shared + i) < kKeyBufferSize * sizeof(char); i++) {
            key_buffer_[shared + i] = key_ptr[i];
        }
        *key_length_ = shared + non_shared;
    }
    
    *value_ptr_ = key_ptr + non_shared;
    *value_length_ = value_length;
    // state->valid_ = true;
    *valid_flag_ = 1;
    state->error_code_ = 0;
}

// ============================================================================
// GDSDataBlockIter Implementation
// ============================================================================

// Constructor
GDSDataBlockIter::GDSDataBlockIter() 
    : BlockIter<Slice>(), d_state_(nullptr), comparator_(nullptr), buffer_(nullptr) {}

// Destructor
GDSDataBlockIter::~GDSDataBlockIter() {
    Cleanup();
}

void GDSDataBlockIter::Initialize(
    const Comparator* raw_ucmp, 
    const char* d_data,  // GPU data pointer
    const DeviceBlockMetadata* d_metadata,  // GPU metadata pointer (avoids DtoH!)
    SequenceNumber global_seqno,
    void* read_amp_bitmap,  // Ignored on GPU
    bool block_contents_pinned,
    bool user_defined_timestamps_persisted,
    GDSDataBlockIterDevicePinnedBuffer* buffer,
    cudaStream_t stream) {
    
    Cleanup();
    
    // Save comparator for validation
    comparator_ = raw_ucmp;
    stream_ = stream;
    
    // Validate comparator type
    // GPU kernels only support bytewise comparison
    if (raw_ucmp != nullptr) {
        const char* comparator_name = raw_ucmp->Name();
        // BytewiseComparator name is "leveldb.BytewiseComparator"
        if (std::string(comparator_name).find("Bytewise") == std::string::npos) {
            status_ = Status::NotSupported(
                "GDS GPU iterator only supports BytewiseComparator. "
                "Current comparator: " + std::string(comparator_name));
            return;
        }
    }

    buffer_ = buffer;
 
    
    // Allocate GPU state
    MLKV_CUDA_CHECK(cudaMallocAsync(&d_state_, sizeof(GDSDataBlockIterState), stream_));

    InitializeIteratorStateKernel<<<1, 1, 0, stream_>>>(d_state_, d_data, d_metadata);
    MLKV_CUDA_CHECK(cudaStreamSynchronize(stream_));
    
    
    status_ = Status::OK();
}

// Override: Returns current value
Slice GDSDataBlockIter::key() {
    assert(Valid());
    if (buffer_ == nullptr || buffer_->h_key_buffer_ == nullptr || buffer_->h_key_length_ == nullptr) {
        return Slice();
    }
    return Slice(buffer_->h_key_buffer_, *buffer_->h_key_length_);
}

// Override: Invalidate iterator with error status
void GDSDataBlockIter::Invalidate(const Status& s) {
    status_ = s;
    // Only set valid flag if buffer is initialized
    if (buffer_ != nullptr && buffer_->h_valid_flag_ != nullptr) {
        *buffer_->h_valid_flag_ = 0;
    }
}



bool GDSDataBlockIter::Valid() {
    if (!d_state_) return false;
    if (buffer_ == nullptr || buffer_->h_valid_flag_ == nullptr) return false;

    return *buffer_->h_valid_flag_ == 1;
}

DeviceSlice GDSDataBlockIter::DKey() const {
    if (!d_state_ || buffer_ == nullptr || buffer_->h_valid_flag_ == nullptr || *buffer_->h_valid_flag_ == 0) {
        return DeviceSlice();
    }
    // Return pointer to key_buffer_ inside d_state_
    return DeviceSlice(
        buffer_->d_key_buffer_,
        *buffer_->h_key_length_
    );
}

DeviceSlice GDSDataBlockIter::DValue() const {
    if (buffer_ == nullptr || buffer_->h_valid_flag_ == nullptr || *buffer_->h_valid_flag_ == 0) {
        return DeviceSlice();
    }
    // Value pointer is directly in the GPU block data
    return DeviceSlice(*buffer_->h_d_value_ptr_, *buffer_->h_value_length_);
}


void GDSDataBlockIter::SeekToFirstImpl() {
    if (!d_state_) {
        status_ = Status::NotSupported("Iterator not initialized");
        return;
    }
    
    SeekToFirstKernel<<<1, 1, 0, stream_>>>(d_state_, buffer_->d_valid_flag_, buffer_->d_key_length_, buffer_->d_key_buffer_, buffer_->d_value_length_, buffer_->d_d_value_ptr_);
    MLKV_CUDA_CHECK(cudaStreamSynchronize(stream_));
    
}

void GDSDataBlockIter::SeekToLastImpl() {
    if (!d_state_) {
        status_ = Status::NotSupported("Iterator not initialized");
        return;
    }
    
    SeekToFirstImpl();
    if (*buffer_->h_valid_flag_ == 0) return;

    while (true) {
        NextImpl();
        if (*buffer_->h_valid_flag_ == 0) {
            break;
        }
    }
}

void GDSDataBlockIter::SeekImpl(const DeviceSlice& d_target) {
    if (!d_state_) {
        status_ = Status::NotSupported("Iterator not initialized");
        return;
    }
    
    SeekKernel<<<1, 1, 0, stream_>>>(d_state_, buffer_->d_valid_flag_, d_target.data(), static_cast<uint32_t>(d_target.size()), buffer_->d_key_length_, buffer_->d_key_buffer_, buffer_->d_value_length_, buffer_->d_d_value_ptr_);
    MLKV_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void GDSDataBlockIter::NextImpl() {
    if (!d_state_ || *buffer_->h_valid_flag_ == 0) return;
    
    NextKernel<<<1, 1, 0, stream_>>>(d_state_, buffer_->d_valid_flag_, buffer_->d_key_length_, buffer_->d_key_buffer_, buffer_->d_value_length_, buffer_->d_d_value_ptr_);
    MLKV_CUDA_CHECK(cudaStreamSynchronize(stream_));
    
}

void GDSDataBlockIter::PrevImpl() {
    // Not implemented - would require caching or backward scan
    status_ = Status::NotSupported("Prev not supported in GPU iterator");
    *buffer_->h_valid_flag_ = 0;
}


void GDSDataBlockIter::Cleanup() {
    if (d_state_) {
        cudaFreeAsync(d_state_, stream_);
        d_state_ = nullptr;
    }

}

// ============================================================================
// DeviceBlock Implementation
// ============================================================================

// Constructor: Takes ownership of GPU memory via DeviceBlockContents
DeviceBlock::DeviceBlock(DeviceBlockContents&& contents, cudaStream_t stream)
    : contents_(std::move(contents)),
      d_metadata_(nullptr),
      metadata_cached_(false),
      stream_(stream) {
    
    assert(!contents_.empty());
    InitializeMetadataGPU();
}

// Destructor: Frees GPU memory if owned
DeviceBlock::~DeviceBlock() {
    // Free GPU metadata
    if (d_metadata_ != nullptr) {
        cudaFree(d_metadata_);
        d_metadata_ = nullptr;
    }
    
    // GPU memory cleanup is handled by DeviceBlockContents destructor
    // However, DeviceBlockContents doesn't have destructor, so we need to free here
    if (contents_.own_bytes() && contents_.allocation) {
        cudaFree(contents_.allocation);
        contents_.allocation = nullptr;
    }
}

// Initialize block metadata on GPU (replaces CPU-side InitializeRestartInfo)
// Parses block footer on GPU to avoid GPU->CPU copies
void DeviceBlock::InitializeMetadataGPU() {


    MLKV_CUDA_CHECK(cudaGetLastError());
    const size_t block_size = contents_.data.size();
    
    if (block_size < sizeof(uint32_t)) {
        // Block too small to contain footer
        throw std::runtime_error(
            "[GDS ERROR] Block size " + std::to_string(block_size) + 
            " is too small to contain footer");
    }
    
    // Allocate GPU memory for metadata
    MLKV_CUDA_CHECK(cudaMallocAsync(&d_metadata_, sizeof(DeviceBlockMetadata), stream_));
    
    // Launch kernel to parse block footer on GPU
    InitializeBlockMetadataKernel<<<1, 1, 0, stream_>>>(
        contents_.data.data(),
        static_cast<uint32_t>(block_size),
        d_metadata_);
    
    MLKV_CUDA_CHECK(cudaGetLastError());

    MLKV_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

// Sync metadata from GPU to host cache (lazy, called on-demand)
void DeviceBlock::SyncMetadataToHost() const {
    if (metadata_cached_) {
        return;  // Already synced
    }
    
    if (d_metadata_ == nullptr) {
        throw std::runtime_error("[GDS ERROR] GPU metadata not initialized");
    }
    
    cudaError_t err = cudaMemcpyAsync(
        &h_metadata_cache_,
        d_metadata_,
        sizeof(DeviceBlockMetadata),
        cudaMemcpyDeviceToHost,
        stream_);
    if (kOpenMemWarnings) {
        std::cerr << "Warning: cudaMemcpyDeviceToHost called in " << std::string(__FILE__) << ":" << std::to_string(__LINE__) << std::endl;
    }
    
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("[GDS ERROR] Failed to sync block metadata from GPU: ") + 
            cudaGetErrorString(err));
    }
    
    MLKV_CUDA_CHECK(cudaStreamSynchronize(stream_));

    metadata_cached_ = true;
}

// Returns approximate memory usage (GPU + host metadata)
size_t DeviceBlock::ApproximateMemoryUsage() const {
    return contents_.ApproximateMemoryUsage() + sizeof(*this) + sizeof(DeviceBlockMetadata);
}

// Returns number of restart points (lazy GPU->CPU copy if not cached)
uint32_t DeviceBlock::NumRestarts() const {
    SyncMetadataToHost();
    return h_metadata_cache_.num_restarts_;
}

// Returns block index type (lazy GPU->CPU copy if not cached)
ROCKSDB_NAMESPACE::BlockBasedTableOptions::DataBlockIndexType DeviceBlock::IndexType() const {
    SyncMetadataToHost();
    return static_cast<ROCKSDB_NAMESPACE::BlockBasedTableOptions::DataBlockIndexType>(
        h_metadata_cache_.index_type_);
}

// Returns restart array offset (lazy GPU->CPU copy if not cached)
uint32_t DeviceBlock::GetRestartOffset() const {
    SyncMetadataToHost();
    return h_metadata_cache_.restart_offset_;
}

// Creates a new GPU-accelerated data block iterator
// Compatible with CPU Block::NewDataIterator interface
// OPTIMIZED: Zero DtoH copy - passes d_metadata_ directly to GPU kernel
GDSDataBlockIter* DeviceBlock::NewDataIterator(
    const Comparator* raw_ucmp,
    SequenceNumber global_seqno,
    GDSDataBlockIterDevicePinnedBuffer* buffer,
    GDSDataBlockIter* iter,
    Statistics* stats,  // Ignored on GPU
    bool block_contents_pinned,
    bool user_defined_timestamps_persisted,
    cudaStream_t stream) {
    
    // Allocate new iterator if not provided
    if (iter == nullptr) {
        iter = new GDSDataBlockIter();
    }
    
    // Initialize iterator with GPU block data and metadata
    // Uses GPU-native Initialize - NO DtoH copy!
    // Metadata stays on GPU, InitializeIteratorStateKernel reads it directly
    iter->Initialize(
        raw_ucmp,
        contents_.data.data(),  // GPU data pointer
        d_metadata_,            // GPU metadata pointer (no sync needed!)
        global_seqno,
        nullptr,                // read_amp_bitmap (ignored on GPU)
        block_contents_pinned,
        user_defined_timestamps_persisted,
        buffer,
        stream);
    
    return iter;
}


// Reads restart point at given index (involves small GPU-to-host copy)
uint32_t DeviceBlock::GetRestartPoint(uint32_t index) const {
    // Sync metadata to get num_restarts and restart_offset
    SyncMetadataToHost();
    
    if (index >= h_metadata_cache_.num_restarts_) {
        return h_metadata_cache_.restart_offset_;  // Invalid index, return end
    }
    
    // Calculate restart point offset in block
    uint32_t restart_array_offset = h_metadata_cache_.restart_offset_ + index * sizeof(uint32_t);
    
    // Read 4 bytes from GPU
    uint32_t restart_value = 0;
    cudaError_t err = cudaMemcpy(
        &restart_value,
        contents_.data.data() + restart_array_offset,
        sizeof(uint32_t),
        cudaMemcpyDeviceToHost);
    if (kOpenMemWarnings) {
        std::cerr << "Warning: cudaMemcpyDeviceToHost called in " << std::string(__FILE__) << ":" << std::to_string(__LINE__) << std::endl;
    }
    
    if (err != cudaSuccess) {
        return h_metadata_cache_.restart_offset_;  // Error, return end
    }
    
    return restart_value;
}





} // namespace gds
} // namespace mlkv_plus


