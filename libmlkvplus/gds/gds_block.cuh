#pragma once

#include <cstring>
#include <stdexcept>
#include <vector>
#include <cuda_runtime.h>

#include "format.cuh"
#include "rocksdb/slice.h"
#include "rocksdb/status.h"
#include "table/block_based/block.h"

using ROCKSDB_NAMESPACE::Slice;
using ROCKSDB_NAMESPACE::Status;
using ROCKSDB_NAMESPACE::SequenceNumber;
using ROCKSDB_NAMESPACE::Comparator;
using ROCKSDB_NAMESPACE::BlockIter;
using ROCKSDB_NAMESPACE::Statistics;
using ROCKSDB_NAMESPACE::BlockType;
using ROCKSDB_NAMESPACE::CacheEntryRole;

namespace mlkv_plus {
namespace gds {

// ============================================================================
// Forward Declarations
// ============================================================================

class GDSDataBlockIter;

// ============================================================================
// GPU Device Utility Functions (inline in header for __device__)
// ============================================================================

namespace device {

// Decode varint32 from GPU memory
__device__ __forceinline__ const char* GetVarint32Ptr(
    const char* p, const char* limit, uint32_t* value);

// Decode block entry: (shared, non_shared, value_length) + key_delta + value
__device__ __forceinline__ const char* DecodeEntry(
    const char* p, const char* limit,
    uint32_t* shared, uint32_t* non_shared, uint32_t* value_length);

// Decode fixed32 from block data
__device__ __forceinline__ uint32_t DecodeFixed32(const char* ptr);

// Get restart point offset
__device__ __forceinline__ uint32_t GetRestartPoint(
    const char* data, uint32_t restarts, uint32_t index);

// Simple bytewise comparison on GPU
__device__ __forceinline__ int CompareSlices(
    const char* a, size_t a_len, const char* b, size_t b_len);

// Internal Key Comparator - handles user_key + sequence + type format
__device__ __forceinline__ int CompareInternalKeys(
    const char* a, size_t a_len, const char* b, size_t b_len);

} // namespace device

// ============================================================================
// Forward Declarations for GPU Block Classes
// ============================================================================

class DeviceBlock;
class DeviceBlock_kData;
class DeviceBlock_kIndex;

// ============================================================================
// GPU-side Data Structures
// ============================================================================

// GPU-side block metadata (parsed from block footer)
// Stored in GPU memory to avoid CPU-GPU transfers
struct DeviceBlockMetadata {
    uint32_t num_restarts_;      // Number of restart points
    uint32_t restart_offset_;    // Offset to restart array
    uint32_t index_type_;        // DataBlockIndexType (0=binary search, 1=binary+hash)
    uint32_t block_size_;        // Total block size
    
    __host__ __device__ DeviceBlockMetadata() 
        : num_restarts_(0), restart_offset_(0), index_type_(0), block_size_(0) {}
};

// GPU-side data block iterator state
struct GDSDataBlockIterState {
    const char* data_;              // Block data pointer in GPU memory
    uint32_t size_;                 // Block size
    uint32_t num_restarts_;         // Number of restart points
    uint32_t restarts_;             // Offset to restart array
    uint32_t current_;              // Current entry offset
    uint32_t restart_index_;        // Current restart interval index
    
    int error_code_;                // 0 = OK, non-zero = error
    
    __host__ __device__ __forceinline__ GDSDataBlockIterState();
};

// ============================================================================
// CUDA Kernel Declarations
// ============================================================================

// Kernel: Parse block footer and initialize metadata (replaces CPU-side InitializeRestartInfo)
__global__ void InitializeBlockMetadataKernel(
    const char* d_data,
    uint32_t block_size,
    DeviceBlockMetadata* d_metadata);

// Kernel: Initialize iterator state from block metadata (GPU-only, no CPU involvement)
__global__ void InitializeIteratorStateKernel(
    GDSDataBlockIterState* d_state,
    const char* d_data,
    const DeviceBlockMetadata* d_metadata);

// Kernel: Initialize iterator to first entry
__global__ void SeekToFirstKernel(GDSDataBlockIterState* state, 
    int* valid_flag_, 
    int* key_length_, 
    char* key_buffer_, 
    int* value_length_, 
    const char** value_ptr_);
// Kernel: Binary search in restart points and seek to target
__global__ void SeekKernel(
    GDSDataBlockIterState* state,
    int* valid_flag_,
    const char* target_key,
    uint32_t target_len,
    int* key_length_,
    char* key_buffer_,
    int* value_length_,
    const char** value_ptr_);

// Kernel: Move to next entry
__global__ void NextKernel(GDSDataBlockIterState* state, 
    int* valid_flag_, 
    int* key_length_, 
    char* key_buffer_,
    int* value_length_,
    const char** value_ptr_);
// ============================================================================
// GDS Data Block Iterator Class
// ============================================================================

// GDS Data Block Iterator - inherits from BlockIter<Slice> interface
// Provides GPU-accelerated block iteration with GDS direct memory access
// Compatible with RocksDB DataBlockIter interface
//
// IMPORTANT LIMITATIONS:
// - Only supports BytewiseComparator (leveldb.BytewiseComparator)
// - GPU kernels use bytewise comparison (memcmp semantics)
// - Custom comparators will result in NotSupported error


class GDSDataBlockIter : public BlockIter<Slice> {
public:
    
    GDSDataBlockIter();
    ~GDSDataBlockIter();
    
    
    void Initialize(const Comparator* raw_ucmp, 
                    const char* d_data,  // GPU data pointer
                    const DeviceBlockMetadata* d_metadata,  // GPU metadata pointer (avoids DtoH!)
                    SequenceNumber global_seqno,
                    void* read_amp_bitmap,  // Ignored on GPU
                    bool block_contents_pinned,
                    bool user_defined_timestamps_persisted,
                    GDSDataBlockIterDevicePinnedBuffer* buffer,
                    cudaStream_t stream=nullptr);
    
    

    // Override: Returns current key
    Slice key();

    DeviceSlice DKey() const;

    DeviceSlice DValue() const;
    
    // Override: Invalidate iterator with error status
    void Invalidate(const Status& s) override;
    
    // Compatibility: SeekForGet
    inline bool SeekForGet(const DeviceSlice& d_target) {
        SeekImpl(d_target);
        return true;
    }
    
    // Check validity on GPU (requires sync)
    bool Valid();

    GDSDataBlockIterState* d_state_;       // Device state pointer
    
    // Comparator (for validation and potential future use)
    const Comparator* comparator_;         // Saved for validation

    GDSDataBlockIterDevicePinnedBuffer* buffer_;

    
    // Cleanup GPU resources
    void Cleanup();
    
    // Make these members accessible (inherited from BlockIter)
    using BlockIter<Slice>::status_;


protected:
    
    // Override: Seek to first entry
    void SeekToFirstImpl() override;
    
    // Override: Seek to last entry (not commonly used, simple implementation)
    void SeekToLastImpl() override;
    
    // Override: Seek to target key (first key >= target)
    void SeekImpl(const DeviceSlice& d_target);
    void SeekImpl(const Slice& target) override {
        throw std::runtime_error("SeekImpl(const Slice& target) is not supported in GDSDataBlockIter");
    }

    // Override: Seek for previous key (first key <= target)
    void SeekForPrevImpl(const Slice& target) override {
        throw std::runtime_error("SeekForPrevImpl(const Slice& target) is not supported in GDSDataBlockIter");
    }

    Slice value() const override {
        throw std::runtime_error("value() is not supported in GDSDataBlockIter");
    };
    
    // Override: Move to next entry
    void NextImpl() override;
    
    // Override: Move to previous entry (complex, not commonly used)
    void PrevImpl() override;



    cudaStream_t stream_;

};

// ============================================================================
// GPU Block Classes
// ============================================================================

// DeviceBlock - GPU version of RocksDB's Block class
// Holds block data in GPU memory, supports creating GPU-accelerated iterators
// 
// Key differences from CPU Block:
// - Data resides in GPU memory (CUDA allocated)
// - No compression support (data is always uncompressed)
// - No checksum verification
// - No cache integration (managed via CachableEntry ownership)
// - Optimized for GDS direct I/O
class DeviceBlock {
public:
    // Constructor: takes ownership of GPU memory via DeviceBlockContents
    // Contents must contain valid GPU-allocated block data
    explicit DeviceBlock(DeviceBlockContents&& contents, cudaStream_t stream);
    
    // Destructor: frees GPU memory if owned
    virtual ~DeviceBlock();
    
    // No copying allowed (move-only type due to GPU memory ownership)
    DeviceBlock(const DeviceBlock&) = delete;
    DeviceBlock& operator=(const DeviceBlock&) = delete;
    
    // ========================================================================
    // Query Methods - Compatible with CPU Block interface
    // ========================================================================
    
    // Returns block size (excluding trailer)
    size_t size() const { return contents_.data.size(); }
    
    // Returns GPU memory pointer (device pointer, not accessible from host!)
    const char* data() const { return contents_.data.data(); }
    
    // Returns allocated memory size
    size_t usable_size() const { return contents_.usable_size(); }
    
    // Returns number of restart points in block (involves small GPU-to-host copy)
    uint32_t NumRestarts() const;
    
    // Returns block index type (binary search or binary+hash) (involves small GPU-to-host copy)
    ROCKSDB_NAMESPACE::BlockBasedTableOptions::DataBlockIndexType IndexType() const;
    
    // Returns whether this block owns the GPU memory
    bool own_bytes() const { return contents_.own_bytes(); }
    
    // Returns restart array offset (involves small GPU-to-host copy)
    uint32_t GetRestartOffset() const;
    
    // Returns GPU pointer to block metadata (for GPU-side operations)
    const DeviceBlockMetadata* GetDeviceMetadata() const { return d_metadata_; }
    
    // Returns approximate memory usage (GPU + host metadata)
    size_t ApproximateMemoryUsage() const;
    
    // ========================================================================
    // Iterator Creation Methods
    // ========================================================================
    
    // Creates a new GPU-accelerated data block iterator
    // If iter is null, allocates new iterator; otherwise reuses existing one
    // 
    // Parameters match CPU Block::NewDataIterator signature for compatibility
    // Note: read_amp_bitmap and stats are ignored (not applicable for GPU)
    GDSDataBlockIter* NewDataIterator(
        const Comparator* raw_ucmp,
        SequenceNumber global_seqno,
        GDSDataBlockIterDevicePinnedBuffer* buffer,
        GDSDataBlockIter* iter = nullptr,
        Statistics* stats = nullptr,  // Ignored on GPU
        bool block_contents_pinned = false,
        bool user_defined_timestamps_persisted = true,
        cudaStream_t stream=nullptr);
    
    
    // Reads restart point at given index (involves small GPU-to-host copy)
    uint32_t GetRestartPoint(uint32_t index) const;

protected:
    DeviceBlockContents contents_;         // GPU memory ownership
    DeviceBlockMetadata* d_metadata_;      // GPU-side block metadata (num_restarts, restart_offset, etc.)
    mutable DeviceBlockMetadata h_metadata_cache_;  // Host-side cache for lazy access
    mutable bool metadata_cached_;         // Whether h_metadata_cache_ is valid
    cudaStream_t stream_;
    
    // Helper: Initialize block metadata on GPU (parses footer)
    void InitializeMetadataGPU();
    
    // Helper: Sync metadata from GPU to host cache (lazy)
    void SyncMetadataToHost() const;
};

// ============================================================================
// Wrapper Classes for Type Distinction (metaprogramming pattern)
// ============================================================================
// These classes enable compile-time type distinction without runtime overhead
// Used by CachableEntry and template instantiation (e.g., IterTraits)

class DeviceBlock_kData : public DeviceBlock {
public:
    using DeviceBlock::DeviceBlock;  // Inherit constructors
    
    static constexpr BlockType kBlockType = BlockType::kData;
    static constexpr CacheEntryRole kCacheEntryRole = CacheEntryRole::kDataBlock;
};

class DeviceBlock_kIndex : public DeviceBlock {
public:
    using DeviceBlock::DeviceBlock;  // Inherit constructors
    
    static constexpr BlockType kBlockType = BlockType::kIndex;
    static constexpr CacheEntryRole kCacheEntryRole = CacheEntryRole::kIndexBlock;
};

class DeviceBlock_kFilter : public DeviceBlock {
public:
    using DeviceBlock::DeviceBlock;  // Inherit constructors
    
    static constexpr BlockType kBlockType = BlockType::kFilter;
    static constexpr CacheEntryRole kCacheEntryRole = CacheEntryRole::kFilterBlock;
};

class DeviceBlock_kRangeDeletion : public DeviceBlock {
public:
    using DeviceBlock::DeviceBlock;  // Inherit constructors
    
    static constexpr BlockType kBlockType = BlockType::kRangeDeletion;
    static constexpr CacheEntryRole kCacheEntryRole = CacheEntryRole::kOtherBlock;
};

} // namespace gds
} // namespace mlkv_plus
