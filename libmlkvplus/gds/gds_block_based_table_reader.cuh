#pragma once


#include <memory>

#include "rocksdb/options.h"
#include "rocksdb/table/block_based/block_based_table_reader.h"
#include "db/table_cache.h"
#include "utils.cuh"
#include "gds_file_system.cuh"
#include "format.cuh"
#include "gds_block.cuh"



namespace mlkv_plus {

namespace gds {



using namespace ROCKSDB_NAMESPACE;

// GPU-side structure for parsed internal key
// This is a device-compatible version of ParsedInternalKey
struct DeviceParsedInternalKey {
  DeviceSlice user_key;
  SequenceNumber sequence;
  ValueType type;
  
  __host__ __device__ __forceinline__ DeviceParsedInternalKey()
    : sequence(kMaxSequenceNumber), type(kTypeDeletion) {}
  
  __host__ __device__ __forceinline__ DeviceParsedInternalKey(const DeviceSlice& u, 
                                                                const SequenceNumber& seq, 
                                                                ValueType t)
    : user_key(u), sequence(seq), type(t) {}
  
  __host__ __device__ __forceinline__ void clear() {
    user_key = DeviceSlice(nullptr, 0);
    sequence = 0;
    type = kTypeDeletion;
  }
};

// Device function to parse internal key on GPU
// This can be called from any device code (kernels or device functions)
// Returns true if successful, false if key is corrupted
__device__ bool ParseInternalKeyOnDevice(
    const DeviceSlice& internal_key,
    DeviceParsedInternalKey* result);

// CUDA kernel to parse a single internal key on GPU
// Can be launched from host code
__global__ void ParseInternalKeyKernel(
    const DeviceSlice internal_key,
    DeviceParsedInternalKey* d_result,
    bool* d_success);

// CUDA kernel to parse multiple internal keys in batch on GPU
// Can be launched from host code for better efficiency
__global__ void ParseInternalKeyBatchKernel(
    const DeviceSlice* d_internal_keys,
    DeviceParsedInternalKey* d_results,
    bool* d_success_flags,
    int num_keys);

// Helper function to parse internal key on GPU (can be called from host)
// This is a convenience wrapper that handles memory management
Status ParseInternalKeyGPU(const DeviceSlice& d_internal_key, 
                           DeviceParsedInternalKey* h_result);


class GDSGetContext : public GetContext {

public:

    GDSGetContext(const Comparator* ucmp, const MergeOperator* merge_operator,
        Logger* logger, Statistics* statistics, GetState init_state,
        const Slice& user_key, DeviceSlice* value,
        PinnableWideColumns* columns, std::string* timestamp,
        bool* value_found, MergeContext* merge_context, bool do_merge,
        SequenceNumber* max_covering_tombstone_seq, SystemClock* clock,
        SequenceNumber* seq = nullptr,
        PinnedIteratorsManager* _pinned_iters_mgr = nullptr,
        ReadCallback* callback = nullptr, bool* is_blob_index = nullptr,
        uint64_t tracing_get_id = 0, BlobFetcher* blob_fetcher = nullptr);


    bool SaveValue(const ParsedInternalKey& parsed_key, const DeviceSlice& value, bool* matched, Status* read_status, Cleanable* value_pinner = nullptr, cudaStream_t stream = nullptr);


private:
    DeviceSlice* device_value_;
    
    
};


class GDSBlockBasedTableReader : public BlockBasedTable {

public:
static Status Open(
    const ReadOptions& read_options, const ImmutableOptions& ioptions,
    const EnvOptions& env_options,
    const BlockBasedTableOptions& table_options,
    const InternalKeyComparator& internal_comparator,
    std::unique_ptr<RandomAccessFileReader>&& file, 
    std::unique_ptr<GDSRandomAccessFile>&& gds_file, uint64_t file_size,
    uint8_t block_protection_bytes_per_key,
    std::unique_ptr<TableReader>* table_reader, uint64_t tail_size,
    std::shared_ptr<CacheReservationManager> table_reader_cache_res_mgr =
        nullptr,
    const std::shared_ptr<const SliceTransform>& prefix_extractor = nullptr,
    UnownedPtr<CompressionManager> compression_manager = nullptr,
    bool prefetch_index_and_filter_in_cache = true, bool skip_filters = false,
    int level = -1, const bool immortal_table = false,
    const SequenceNumber largest_seqno = 0,
    bool force_direct_prefetch = false,
    TailPrefetchStats* tail_prefetch_stats = nullptr,
    BlockCacheTracer* const block_cache_tracer = nullptr,
    size_t max_file_size_for_l0_meta_pin = 0,
    const std::string& cur_db_session_id = "", uint64_t cur_file_num = 0,
    UniqueId64x2 expected_unique_id = {},
    const bool user_defined_timestamps_persisted = true,
    const GDSOptions& gds_options = GDSOptions());    


Status Get(const ReadOptions& read_options, const Slice& key, const DeviceSlice& d_key,
    GDSGetContext* get_context,
    const SliceTransform* prefix_extractor,
    bool skip_filters, GDSDataBlockIterDevicePinnedBuffer* buffer, cudaStream_t stream=nullptr);


// input_iter: if it is not null, update this one and return it as Iterator
template <typename TBlockIter>
TBlockIter* NewDataBlockIterator(
    const ReadOptions& ro, const BlockHandle& block_handle,
    TBlockIter* input_iter, BlockType block_type, GetContext* get_context,
    BlockCacheLookupContext* lookup_context,
    FilePrefetchBuffer* prefetch_buffer, bool for_compaction, bool async_read,
    Status& s, bool use_block_cache_for_lookup, GDSDataBlockIterDevicePinnedBuffer* buffer, cudaStream_t stream=nullptr) const;

// Similar to the above, with one crucial difference: it will retrieve the
// block from the file even if there are no caches configured (assuming the
// read options allow I/O).
template <typename TBlocklike>
WithBlocklikeCheck<Status, TBlocklike> RetrieveBlock(
    FilePrefetchBuffer* prefetch_buffer, const ReadOptions& ro,
    const BlockHandle& handle, UnownedPtr<Decompressor> decomp,
    CachableEntry<TBlocklike>* block_entry, GetContext* get_context,
    BlockCacheLookupContext* lookup_context, bool for_compaction,
    bool use_cache, bool async_read, bool use_block_cache_for_lookup, cudaStream_t stream=nullptr) const;


template <typename TBlockIter>
static TBlockIter* InitBlockIterator(const Rep* rep, DeviceBlock* block,
                                        BlockType block_type,
                                        TBlockIter* input_iter,
                                        bool block_contents_pinned,
                                        GDSDataBlockIterDevicePinnedBuffer* buffer,
                                        cudaStream_t stream=nullptr);



private:

    explicit GDSBlockBasedTableReader(Rep* rep, BlockCacheTracer* const block_cache_tracer, const GDSOptions& gds_options, std::unique_ptr<GDSRandomAccessFile>&& gds_file)
        : BlockBasedTable(rep, block_cache_tracer), gds_options_(gds_options), gds_file_(std::move(gds_file)) {}

    // No copying allowed
    explicit GDSBlockBasedTableReader(const TableReader&) = delete;
    void operator=(const TableReader&) = delete;

    const GDSOptions& gds_options_;
    std::unique_ptr<GDSRandomAccessFile> gds_file_;
};






// Read block from storage via GDS to GPU memory
Status GDSReadBlock(
    const ReadOptions& ro, 
    GDSRandomAccessFile& file,
    const BlockHandle& handle,
    DeviceBlockContents* contents,
    const Footer& footer,
    cudaStream_t stream = nullptr);

// Read and parse block from file using GDS
template <typename TBlocklike>
Status GDSReadAndParseBlockFromFile(
    GDSRandomAccessFile* file, 
    FilePrefetchBuffer* prefetch_buffer,
    const Footer& footer, 
    const ReadOptions& options, 
    const BlockHandle& handle,
    std::unique_ptr<TBlocklike>* result, 
    const ImmutableOptions& ioptions,
    BlockCreateContext& create_context, 
    bool maybe_compressed,
    UnownedPtr<Decompressor> decomp,
    const PersistentCacheOptions& cache_options,
    MemoryAllocator* memory_allocator, 
    bool for_compaction, 
    bool async_read,
    cudaStream_t stream = nullptr);


} // namespace gds

} // namespace mlkv_plus