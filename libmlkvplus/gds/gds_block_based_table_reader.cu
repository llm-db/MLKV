#include "db/dbformat.h"
#include "gds_block_based_table_reader.cuh"
#include "rocksdb/file/random_access_file_reader.h"
#include "table/format.h"
#include "table/block_based/reader_common.h"

#include <iostream>

#include <cstdint>
#include <memory>
#include <vector>


#include "cache/cache_reservation_manager.h"

#include "rocksdb/slice_transform.h"

#include "table/block_based/block.h"
#include "table/block_based/block_based_table_factory.h"
#include "table/block_based/block_type.h"
#include "table/block_based/filter_block.h"
#include "table/block_based/uncompression_dict_reader.h"
#include "table/format.h"
#include "table/table_reader.h"
#include "table/two_level_iterator.h"
#include "trace_replay/block_cache_tracer.h"
#include "util/cast_util.h"
#include "util/coro_utils.h"

#include "logging/logging.h"

#include "rocksdb/convenience.h"

#include "table/block_based/block_based_table_reader_impl.h"
#include "db/wide/wide_column_serialization.h"
#include "monitoring/perf_context_imp.h"
#include "rocksdb/merge_operator.h"

#include "gds_block.cuh"
#include "format.cuh"


namespace ROCKSDB_NAMESPACE {
// Provide inline function definition (declared in block_based_table_reader.h)
inline bool PrefixExtractorChangedHelper(
    const TableProperties* table_properties,
    const SliceTransform* prefix_extractor) {
  if (prefix_extractor == nullptr || table_properties == nullptr ||
      table_properties->prefix_extractor_name.empty()) {
    return true;
  }
  if (table_properties->prefix_extractor_name != prefix_extractor->AsString()) {
    return true;
  }
  return false;
}
}  // namespace ROCKSDB_NAMESPACE

using namespace ROCKSDB_NAMESPACE;
using ROCKSDB_NAMESPACE::WideColumnSerialization;
using ROCKSDB_NAMESPACE::ParsePackedValueForValue;


namespace mlkv_plus {

namespace gds {

/*
 * GPU-based Internal Key Parsing
 * ===============================
 * 
 * This module provides CUDA kernels for parsing RocksDB internal keys directly on the GPU.
 * 
 * Background:
 * ----------
 * RocksDB internal keys consist of:
 *   [user_key][sequence_number (7 bytes)][value_type (1 byte)]
 * 
 * The last 8 bytes contain the sequence number and value type packed together:
 *   - Lower byte: ValueType
 *   - Upper 7 bytes: SequenceNumber
 * 
 * Usage Examples:
 * --------------
 * 
 * 1. Parse a single key on GPU (from device code):
 *    __device__ void MyKernel() {
 *      DeviceSlice d_key = ...;  // Key in GPU memory
 *      DeviceParsedInternalKey parsed_key;
 *      bool success = ParseInternalKeyOnDevice(d_key, &parsed_key);
 *      if (success) {
 *        // Use parsed_key.user_key, parsed_key.sequence, parsed_key.type
 *      }
 *    }
 * 
 * 2. Parse a single key from host code:
 *    DeviceSlice d_key = biter.DKey();  // Get key from GPU memory
 *    DeviceParsedInternalKey result;
 *    Status s = ParseInternalKeyGPU(d_key, &result);
 *    if (s.ok()) {
 *      // Use result (now in host memory)
 *    }
 * 
 * 3. Parse multiple keys in batch (more efficient):
 *    int num_keys = 100;
 *    DeviceSlice* d_keys;
 *    DeviceParsedInternalKey* d_results;
 *    bool* d_success_flags;
 *    // ... allocate memory ...
 *    
 *    int blockSize = 256;
 *    int numBlocks = (num_keys + blockSize - 1) / blockSize;
 *    ParseInternalKeyBatchKernel<<<numBlocks, blockSize>>>(
 *        d_keys, d_results, d_success_flags, num_keys);
 *    cudaDeviceSynchronize();
 * 
 * Benefits:
 * --------
 * - Avoids copying keys from GPU to CPU for parsing
 * - Enables end-to-end GPU processing with GDS (GPU Direct Storage)
 * - Batch processing support for high throughput
 */

// Device function to decode fixed 64-bit value (little endian)
__device__ __forceinline__ uint64_t DeviceDecodeFixed64(const char* ptr) {
  // Assuming little endian (CUDA GPUs are little endian)
  uint64_t result;
  memcpy(&result, ptr, sizeof(result));
  return result;
}

// Device function to check if value type is valid
__device__ __forceinline__ bool DeviceIsValueType(ValueType t) {
  return t <= kTypeMerge || kTypeSingleDeletion == t || kTypeBlobIndex == t ||
         kTypeDeletionWithTimestamp == t || kTypeWideColumnEntity == t ||
         kTypeValuePreferredSeqno == t;
}

// Device function to check if value type is extended
__device__ __forceinline__ bool DeviceIsExtendedValueType(ValueType t) {
  return DeviceIsValueType(t) || t == kTypeRangeDeletion || t == kTypeMaxValid;
}

// CUDA kernel to parse internal key on GPU
// Returns true if successful, false if key is corrupted
__device__ __forceinline__ bool ParseInternalKeyOnDevice(
    const DeviceSlice& internal_key,
    DeviceParsedInternalKey* result) {
  
  const size_t n = internal_key.size();
  constexpr uint64_t kNumInternalBytes = 8;
  
  // Check if key is large enough
  if (n < kNumInternalBytes) {
    return false;  // Corrupted key
  }
  
  // Decode the last 8 bytes containing sequence number and type
  uint64_t num = DeviceDecodeFixed64(internal_key.data() + n - kNumInternalBytes);
  
  // Extract type (lower byte) and sequence (upper 7 bytes)
  unsigned char c = num & 0xff;
  result->sequence = num >> 8;
  result->type = static_cast<ValueType>(c);
  
  // Extract user key (everything except last 8 bytes)
  result->user_key = DeviceSlice(internal_key.data(), n - kNumInternalBytes);
  
  // Validate the type
  if (!DeviceIsExtendedValueType(result->type)) {
    return false;  // Invalid type
  }
  
  return true;
}

// Host-callable wrapper kernel for parsing a single internal key
__global__ void ParseInternalKeyKernel(
    const DeviceSlice internal_key,
    DeviceParsedInternalKey* d_result,
    bool* d_success) {
  
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *d_success = ParseInternalKeyOnDevice(internal_key, d_result);
  }
}

// Host-callable wrapper kernel for parsing multiple internal keys in batch
__global__ void ParseInternalKeyBatchKernel(
    const DeviceSlice* d_internal_keys,
    DeviceParsedInternalKey* d_results,
    bool* d_success_flags,
    int num_keys) {
  
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_keys) {
    d_success_flags[idx] = ParseInternalKeyOnDevice(d_internal_keys[idx], &d_results[idx]);
  }
}

// Helper function to parse internal key on GPU (can be called from host)
inline Status ParseInternalKeyGPU(const DeviceSlice& d_internal_key, 
                                   DeviceParsedInternalKey* h_result, cudaStream_t stream) {
  DeviceParsedInternalKey* d_result;
  bool* d_success;
  bool h_success;
  
  cudaMallocAsync(&d_result, sizeof(DeviceParsedInternalKey), stream);
  cudaMallocAsync(&d_success, sizeof(bool), stream);
  
  ParseInternalKeyKernel<<<1, 1, 0, stream>>>(d_internal_key, d_result, d_success);
  
  cudaMemcpyAsync(&h_success, d_success, sizeof(bool), cudaMemcpyDeviceToHost, stream);
  if (h_success && h_result) {
    cudaMemcpyAsync(h_result, d_result, sizeof(DeviceParsedInternalKey), cudaMemcpyDeviceToHost, stream);
  }
  
  cudaFreeAsync(d_result, stream);
  cudaFreeAsync(d_success, stream);
  
  if (!h_success) {
    return Status::Corruption("Corrupted Key: Failed to parse internal key on GPU");
  }
  
  return Status::OK();
}

// Implementation of the standard Open method (delegates to OpenWithGDS with default options)
Status GDSBlockBasedTableReader::Open(
  const ReadOptions& read_options, const ImmutableOptions& ioptions,
  const EnvOptions& env_options,
  const BlockBasedTableOptions& table_options,
  const InternalKeyComparator& internal_comparator,
  std::unique_ptr<RandomAccessFileReader>&& file, std::unique_ptr<GDSRandomAccessFile>&& gds_file, uint64_t file_size,
  uint8_t block_protection_bytes_per_key,
  std::unique_ptr<TableReader>* table_reader, uint64_t tail_size,
  std::shared_ptr<CacheReservationManager> table_reader_cache_res_mgr,
  const std::shared_ptr<const SliceTransform>& prefix_extractor,
  UnownedPtr<CompressionManager> compression_manager,
  bool prefetch_index_and_filter_in_cache, bool skip_filters,
  int level, const bool immortal_table,
  const SequenceNumber largest_seqno,
  bool force_direct_prefetch,
  TailPrefetchStats* tail_prefetch_stats,
  BlockCacheTracer* const block_cache_tracer,
  size_t max_file_size_for_l0_meta_pin,
  const std::string& cur_db_session_id, uint64_t cur_file_num,
  UniqueId64x2 expected_unique_id,
  const bool user_defined_timestamps_persisted,
  const GDSOptions& gds_options){

  table_reader->reset();

  Status s;
  Footer footer;
  std::unique_ptr<FilePrefetchBuffer> prefetch_buffer;

  // From read_options, retain deadline, io_timeout, rate_limiter_priority, and
  // verify_checksums. In future, we may retain more options.
  // TODO: audit more ReadOptions and do this in a way that brings attention
  // on new ReadOptions?
  ReadOptions ro;
  ro.deadline = read_options.deadline;
  ro.io_timeout = read_options.io_timeout;
  ro.rate_limiter_priority = read_options.rate_limiter_priority;
  ro.verify_checksums = read_options.verify_checksums;
  ro.io_activity = read_options.io_activity;
  ro.fill_cache = read_options.fill_cache;

  // prefetch both index and filters, down to all partitions
  const bool prefetch_all = prefetch_index_and_filter_in_cache || level == 0;
  const bool preload_all = !table_options.cache_index_and_filter_blocks;

  if (!ioptions.allow_mmap_reads && !env_options.use_mmap_reads) {
    s = PrefetchTail(ro, ioptions, file.get(), file_size, force_direct_prefetch,
                     tail_prefetch_stats, prefetch_all, preload_all,
                     &prefetch_buffer, ioptions.stats, tail_size,
                     ioptions.logger);
    // Return error in prefetch path to users.
    if (!s.ok()) {
      return s;
    }
  } else {
    // Should not prefetch for mmap mode.
    prefetch_buffer.reset(new FilePrefetchBuffer(
        ReadaheadParams(), false /* enable */, true /* track_min_offset */));
  }

  // Read in the following order:
  //    1. Footer
  //    2. [metaindex block]
  //    3. [meta block: properties]
  //    4. [meta block: range deletion tombstone]
  //    5. [meta block: compression dictionary]
  //    6. [meta block: index]
  //    7. [meta block: filter]
  IOOptions opts;
  IODebugContext dbg;
  s = file->PrepareIOOptions(ro, opts, &dbg);
  if (s.ok()) {
    s = ReadFooterFromFile(opts, file.get(), *ioptions.fs,
                           prefetch_buffer.get(), file_size, &footer,
                           kBlockBasedTableMagicNumber, ioptions.stats);
  }
  if (!s.ok()) {
    if (s.IsCorruption()) {
      RecordTick(ioptions.statistics.get(), SST_FOOTER_CORRUPTION_COUNT);
    }
    return s;
  }
  if (!IsSupportedFormatVersion(footer.format_version()) &&
      !TEST_AllowUnsupportedFormatVersion()) {
    return Status::Corruption(
        "Unknown Footer version. Maybe this file was created with newer "
        "version of RocksDB?");
  }

  BlockCacheLookupContext lookup_context{TableReaderCaller::kPrefetch};
  Rep* rep = new BlockBasedTable::Rep(
      ioptions, env_options, table_options, internal_comparator, skip_filters,
      file_size, level, immortal_table, user_defined_timestamps_persisted);
  rep->file = std::move(file);
  rep->footer = footer;

  // Some ancient versions (~2.5 - 2.7, format_version=1) could compress the
  // metaindex block, so we need to allow for that
  if (footer.format_version() < 2) {
    auto mgr = GetBuiltinCompressionManager(/*compression_format_version=*/1);
    rep->decompressor = mgr->GetDecompressor();
  }

  // For fully portable/stable cache keys, we need to read the properties
  // block before setting up cache keys. TODO: consider setting up a bootstrap
  // cache key for PersistentCache to use for metaindex and properties blocks.
  rep->persistent_cache_options = PersistentCacheOptions();

  // Meta-blocks are not dictionary compressed. Explicitly set the dictionary
  // handle to null, otherwise it may be seen as uninitialized during the below
  // meta-block reads.
  rep->compression_dict_handle = BlockHandle::NullBlockHandle();

  rep->create_context.protection_bytes_per_key = block_protection_bytes_per_key;
  // Read metaindex
  std::unique_ptr<GDSBlockBasedTableReader> new_table(
      new GDSBlockBasedTableReader(rep, block_cache_tracer, gds_options, std::move(gds_file)));
  std::unique_ptr<Block> metaindex;
  std::unique_ptr<InternalIterator> metaindex_iter;
  s = new_table->ReadMetaIndexBlock(ro, prefetch_buffer.get(), &metaindex,
                                    &metaindex_iter);
  if (!s.ok()) {
    return s;
  }

  // Populates table_properties and some fields that depend on it,
  // such as index_type.
  s = new_table->ReadPropertiesBlock(ro, prefetch_buffer.get(),
                                     metaindex_iter.get(), largest_seqno);
  if (!s.ok()) {
    return s;
  }

  // Read compression metadata and configure decompressor
  s = GetDecompressor(
      rep->table_properties ? rep->table_properties->compression_name
                            : std::string{},
      compression_manager, footer.format_version(), &rep->decompressor);
  if (!s.ok()) {
    return s;
  }

  // Populate BlockCreateContext
  rep->create_context = BlockCreateContext(
      &rep->table_options, &rep->ioptions, rep->ioptions.stats,
      rep->decompressor.get(), block_protection_bytes_per_key,
      rep->internal_comparator.user_comparator(), rep->index_value_is_full,
      rep->index_has_first_key);

  // Check expected unique id if provided
  if (expected_unique_id != kNullUniqueId64x2) {
    auto props = rep->table_properties;
    if (!props) {
      return Status::Corruption("Missing table properties on file " +
                                std::to_string(cur_file_num) +
                                " with known unique ID");
    }
    UniqueId64x2 actual_unique_id{};
    s = GetSstInternalUniqueId(props->db_id, props->db_session_id,
                               props->orig_file_number, &actual_unique_id,
                               /*force*/ true);
    assert(s.ok());  // because force=true
    if (expected_unique_id != actual_unique_id) {
      return Status::Corruption(
          "Mismatch in unique ID on table file " +
          std::to_string(cur_file_num) +
          ". Expected: " + InternalUniqueIdToHumanString(&expected_unique_id) +
          " Actual: " + InternalUniqueIdToHumanString(&actual_unique_id));
    }
  } else {
    if (ioptions.verify_sst_unique_id_in_manifest && ioptions.logger) {
      // A crude but isolated way of reporting unverified files. This should not
      // be an ongoing concern so doesn't deserve a place in Statistics IMHO.
      static std::atomic<uint64_t> unverified_count{0};
      auto prev_count =
          unverified_count.fetch_add(1, std::memory_order_relaxed);
      if (prev_count == 0) {
        ROCKS_LOG_WARN(
            ioptions.logger,
            "At least one SST file opened without unique ID to verify: %" PRIu64
            ".sst",
            cur_file_num);
      } else if (prev_count % 1000 == 0) {
        ROCKS_LOG_WARN(
            ioptions.logger,
            "Another ~1000 SST files opened without unique ID to verify");
      }
    }
  }

  // Set up prefix extracto as needed
  bool force_null_table_prefix_extractor = false;
  if (force_null_table_prefix_extractor) {
    assert(!rep->table_prefix_extractor);
  } else if (!PrefixExtractorChangedHelper(rep->table_properties.get(),
                                           prefix_extractor.get())) {
    // Establish fast path for unchanged prefix_extractor
    rep->table_prefix_extractor = prefix_extractor;
  } else {
    // Current prefix_extractor doesn't match table
    if (rep->table_properties) {
      //**TODO: If/When the DBOptions has a registry in it, the ConfigOptions
      // will need to use it
      ConfigOptions config_options;
      Status st = SliceTransform::CreateFromString(
          config_options, rep->table_properties->prefix_extractor_name,
          &(rep->table_prefix_extractor));
      if (!st.ok()) {
        //**TODO: Should this be error be returned or swallowed?
        ROCKS_LOG_ERROR(rep->ioptions.logger,
                        "Failed to create prefix extractor[%s]: %s",
                        rep->table_properties->prefix_extractor_name.c_str(),
                        st.ToString().c_str());
      }
    }
  }

  // With properties loaded, we can set up portable/stable cache keys
  SetupBaseCacheKey(rep->table_properties.get(), cur_db_session_id,
                    cur_file_num, &rep->base_cache_key);

  rep->persistent_cache_options =
      PersistentCacheOptions(rep->table_options.persistent_cache,
                             rep->base_cache_key, rep->ioptions.stats);

  s = new_table->ReadRangeDelBlock(ro, prefetch_buffer.get(),
                                   metaindex_iter.get(), internal_comparator,
                                   &lookup_context);
  if (!s.ok()) {
    return s;
  }
  rep->verify_checksum_set_on_open = ro.verify_checksums;
  s = new_table->PrefetchIndexAndFilterBlocks(
      ro, prefetch_buffer.get(), metaindex_iter.get(), new_table.get(),
      prefetch_all, table_options, level, file_size,
      max_file_size_for_l0_meta_pin, &lookup_context);

  if (s.ok()) {
    // Update tail prefetch stats
    assert(prefetch_buffer.get() != nullptr);
    if (tail_prefetch_stats != nullptr) {
      assert(prefetch_buffer->min_offset_read() < file_size);
      tail_prefetch_stats->RecordEffectiveSize(
          static_cast<size_t>(file_size) - prefetch_buffer->min_offset_read());
    }
  }

  if (s.ok() && table_reader_cache_res_mgr) {
    std::size_t mem_usage = new_table->ApproximateMemoryUsage();
    s = table_reader_cache_res_mgr->MakeCacheReservation(
        mem_usage, &(rep->table_reader_cache_res_handle));
    if (s.IsMemoryLimit()) {
      s = Status::MemoryLimit(
          "Can't allocate " +
          kCacheEntryRoleToCamelString[static_cast<std::uint32_t>(
              CacheEntryRole::kBlockBasedTableReader)] +
          " due to memory limit based on "
          "cache capacity for memory allocation");
    }
  }

  if (s.ok()) {
    *table_reader = std::move(new_table);
  }
  return s;
}




Status GDSBlockBasedTableReader::Get(const ReadOptions& read_options, const Slice& key, const DeviceSlice& d_key,
                            GDSGetContext* get_context,
                            const SliceTransform* prefix_extractor,
                            bool skip_filters, GDSDataBlockIterDevicePinnedBuffer* buffer, cudaStream_t stream) {
  // Similar to Bloom filter !may_match
  // If timestamp is beyond the range of the table, skip
  if (!TimestampMayMatch(read_options)) {
    return Status::OK();
  }
  assert(key.size() >= 8);  // key must be internal key
  assert(get_context != nullptr);
  Status s;

  FilterBlockReader* const filter =
      !skip_filters ? rep_->filter.get() : nullptr;

  // First check the full filter
  // If full filter not useful, Then go into each block
  uint64_t tracing_get_id = get_context->get_tracing_get_id();
  BlockCacheLookupContext lookup_context{
      TableReaderCaller::kUserGet, tracing_get_id,
      /*get_from_user_specified_snapshot=*/read_options.snapshot != nullptr};

  const bool may_match =
      FullFilterKeyMayMatch(filter, key, prefix_extractor, get_context,
                            &lookup_context, read_options);
  if (may_match) {
    IndexBlockIter iiter_on_stack;
    // if prefix_extractor found in block differs from options, disable
    // BlockPrefixIndex. Only do this check when index_type is kHashSearch.
    bool need_upper_bound_check = false;
    if (rep_->index_type == BlockBasedTableOptions::kHashSearch) {
      need_upper_bound_check = PrefixExtractorChanged(prefix_extractor);
    }
    auto iiter =
        NewIndexIterator(read_options, need_upper_bound_check, &iiter_on_stack,
                         get_context, &lookup_context);
    std::unique_ptr<InternalIteratorBase<IndexValue>> iiter_unique_ptr;
    if (iiter != &iiter_on_stack) {
      iiter_unique_ptr.reset(iiter);
    }

    size_t ts_sz =
        rep_->internal_comparator.user_comparator()->timestamp_size();
    bool matched = false;  // if such user key matched a key in SST
    bool done = false;
    for (iiter->Seek(key); iiter->Valid() && !done; iiter->Next()) {
      IndexValue v = iiter->value();

      // std::cout << "[GDS DEBUG] key: " << key.ToString() << std::endl;
      // std::cout << "[GDS DEBUG] iiter->value(): " << v.first_internal_key.ToString() << std::endl;
      // std::cout << "[GDS DEBUG] iiter->value().handle: " << v.handle.ToString() << std::endl;

      if (!v.first_internal_key.empty() && !skip_filters &&
          UserComparatorWrapper(rep_->internal_comparator.user_comparator())
                  .CompareWithoutTimestamp(
                      ExtractUserKey(key),
                      ExtractUserKey(v.first_internal_key)) < 0) {
        // The requested key falls between highest key in previous block and
        // lowest key in current block.
        break;
      }

      // BlockCacheLookupContext lookup_data_block_context{
      //     TableReaderCaller::kUserGet, tracing_get_id,
      //     /*get_from_user_specified_snapshot=*/read_options.snapshot !=
      //         nullptr};
      // bool does_referenced_key_exist = false;
      GDSDataBlockIter biter;
      // uint64_t referenced_data_size = 0;
      Status tmp_status;
      NewDataBlockIterator<GDSDataBlockIter>(
          read_options, v.handle, &biter, BlockType::kData, get_context,
          nullptr /* lookup_data_block_context (not used for GDS) */, /*prefetch_buffer=*/nullptr,
          /*for_compaction=*/false, /*async_read=*/false, tmp_status,
          /*use_block_cache_for_lookup=*/false, buffer, stream);

      if (read_options.read_tier == kBlockCacheTier &&
          biter.status().IsIncomplete()) {
        throw std::runtime_error("Not Supported: read_options.read_tier == kBlockCacheTier");
        // couldn't get block from block_cache
        // Update Saver.state to Found because we are only looking for
        // whether we can guarantee the key is not there when "no_io" is set
        // get_context->MarkKeyMayExist();
        // s = biter.status();
        // break;
      }
      if (!biter.status().ok()) {
        s = biter.status();
        break;
      }

      bool may_exist = biter.SeekForGet(d_key);
      // If user-specified timestamp is supported, we cannot end the search
      // just because hash index lookup indicates the key+ts does not exist.
      if (!may_exist && ts_sz == 0) {
        // HashSeek cannot find the key this block and the the iter is not
        // the end of the block, i.e. cannot be in the following blocks
        // either. In this case, the seek_key cannot be found, so we break
        // from the top level for-loop.
        done = true;
      } else {
        // Call the *saver function on each entry/block until it returns false
        for (; biter.Valid(); biter.Next()) {
          
          // DeviceParsedInternalKey* d_result;
          // bool* d_success;
          // bool h_success;
          
          // MLKV_CUDA_CHECK(cudaMalloc(&d_result, sizeof(DeviceParsedInternalKey)));
          // MLKV_CUDA_CHECK(cudaMalloc(&d_success, sizeof(bool)));
          
          // ParseInternalKeyKernel<<<1, 1>>>(biter.DKey(), d_result, d_success);

          // MLKV_CUDA_CHECK(cudaMemcpy(&h_success, d_success, sizeof(bool), cudaMemcpyDeviceToHost));
          // MLKV_CUDA_CHECK(cudaFree(d_result));
          // if (!h_success) {
          //   s = Status::Corruption("Corrupted Key: Failed to parse internal key on GPU");
          //   break;
          // }

          ParsedInternalKey parsed_key;
          Status pik_status = ParseInternalKey(biter.key(), &parsed_key, false /* log_err_key */);
          if (!pik_status.ok()) {
            s = pik_status;
            break;
          }
          // std::cout << "[GDS DEBUG] pik_status: " << pik_status.ToString() << std::endl;
          // std::cout << "[GDS DEBUG] parsed_key: " << parsed_key.user_key.ToString() << std::endl;
          // std::cout << "[GDS DEBUG] parsed_key.sequence: " << parsed_key.sequence << std::endl;
          // std::cout << "[GDS DEBUG] parsed_key.type: " << parsed_key.type << std::endl;
          // std::cout << "[GDS DEBUG] biter.key(): " << biter.key().ToString() << std::endl;
          // std::cout << "[GDS DEBUG] biter.restart_index_: " << biter.h_state_.restart_index_ << std::endl;
          // std::cout << "[GDS DEBUG] biter.num_restarts_: " << biter.h_state_.num_restarts_ << std::endl;
          // std::cout << "[GDS DEBUG] biter.current_: " << biter.h_state_.current_ << std::endl;



          Status read_status;
          bool ret = get_context->SaveValue(
              parsed_key, biter.DValue(), &matched, &read_status,
              biter.IsValuePinned() ? &biter : nullptr);


          // std::cout << "[GDS DEBUG] read_status: " << read_status.ToString() << std::endl;
          // std::cout << "[GDS DEBUG] ret: " << ret << std::endl;
          // std::cout << "[GDS DEBUG] get_context->State(): " << get_context->State() << std::endl;
          // std::cout << "[GDS DEBUG] biter.IsValuePinned(): " << biter.IsValuePinned() << std::endl;
          // std::cout << "[GDS DEBUG] matched: " << matched << std::endl;
          if (!read_status.ok()) {
            s = read_status;
            break;
          }
          if (!ret) {
            // if (get_context->State() == GetContext::GetState::kFound) {
            //   does_referenced_key_exist = true;
            //   referenced_data_size = biter.key().size() + biter.value().size();
            // }
            done = true;
            break;
          }
        }
        if (s.ok()) {
          s = biter.status();
        }
        if (!s.ok()) {
          break;
        }
      }


      if (done) {
        // Avoid the extra Next which is expensive in two-level indexes
        break;
      }
    }
    if (matched && filter != nullptr) {
      if (rep_->whole_key_filtering) {
        RecordTick(rep_->ioptions.stats, BLOOM_FILTER_FULL_TRUE_POSITIVE);
      } else {
        RecordTick(rep_->ioptions.stats, BLOOM_FILTER_PREFIX_TRUE_POSITIVE);
      }
      // Includes prefix stats
      PERF_COUNTER_BY_LEVEL_ADD(bloom_filter_full_true_positive, 1,
                                rep_->level);
    }

    if (s.ok() && !iiter->status().IsNotFound()) {
      s = iiter->status();
    }
  }

  return s;
}


template <typename TBlockIter>
struct IterTraits {};

template <>
struct IterTraits<GDSDataBlockIter> {
  using IterBlocklike = DeviceBlock_kData;
};

template <>
GDSDataBlockIter* GDSBlockBasedTableReader::InitBlockIterator<GDSDataBlockIter>(
    const Rep* rep, DeviceBlock* block, BlockType block_type,
    GDSDataBlockIter* input_iter, bool block_contents_pinned, GDSDataBlockIterDevicePinnedBuffer* buffer, cudaStream_t stream) {
  return block->NewDataIterator(rep->internal_comparator.user_comparator(),
                                rep->get_global_seqno(block_type), buffer,input_iter,
                                rep->ioptions.stats, block_contents_pinned,
                                rep->user_defined_timestamps_persisted, stream);
}

// Convert an index iterator value (i.e., an encoded BlockHandle)
// into an iterator over the contents of the corresponding block.
// If input_iter is null, new a iterator
// If input_iter is not null, update this iter and return it
template <typename TBlockIter>
TBlockIter* GDSBlockBasedTableReader::NewDataBlockIterator(
    const ReadOptions& ro, const BlockHandle& handle, TBlockIter* input_iter,
    BlockType block_type, GetContext* get_context,
    BlockCacheLookupContext* lookup_context,
    FilePrefetchBuffer* prefetch_buffer, bool for_compaction, bool async_read,
    Status& s, bool use_block_cache_for_lookup, GDSDataBlockIterDevicePinnedBuffer* buffer, cudaStream_t stream) const {
  using IterBlocklike = typename IterTraits<TBlockIter>::IterBlocklike;
  PERF_TIMER_GUARD(new_table_block_iter_nanos);

  TBlockIter* iter = input_iter != nullptr ? input_iter : new TBlockIter;
  if (!s.ok()) {
    iter->Invalidate(s);
    return iter;
  }

  CachableEntry<DeviceBlock> block;
  {
    // CachableEntry<DecompressorDict> dict;
    // Decompressor* decomp = rep_->decompressor.get();
    // if (rep_->uncompression_dict_reader && block_type == BlockType::kData) {
    //   // For async scans, don't use the prefetch buffer since an async prefetch
    //   // might already be under way and this would invalidate it. Also, the
    //   // uncompression dict is typically at the end of the file and would
    //   // most likely break the sequentiality of the access pattern.
    //   // Same is with auto_readahead_size. It iterates over index to lookup for
    //   // data blocks. And this could break the the sequentiality of the access
    //   // pattern.
    //   s = rep_->uncompression_dict_reader->GetOrReadUncompressionDictionary(
    //       ((ro.async_io || ro.auto_readahead_size) ? nullptr : prefetch_buffer),
    //       ro, get_context, lookup_context, &dict);
    //   if (!s.ok()) {
    //     iter->Invalidate(s);
    //     return iter;
    //   }
    //   assert(dict.GetValue());
    //   if (dict.GetValue()) {
    //     decomp = dict.GetValue()->decompressor_.get();
    //   }
    // }
    s = RetrieveBlock(
        prefetch_buffer, ro, handle, nullptr /* decomp (not used for GDS) */, &block.As<IterBlocklike>(),
        get_context, lookup_context, for_compaction,
        /* use_cache */ false, async_read, use_block_cache_for_lookup, stream);
  }

  if (s.IsTryAgain() && async_read) {
    return iter;
  }

  if (!s.ok()) {
    assert(block.IsEmpty());
    std::cerr << "GDSBlockBasedTableReader::NewDataBlockIterator: " << s.ToString() << std::endl;
    iter->Invalidate(s);
    return iter;
  }

  assert(block.GetValue() != nullptr);

  // Block contents are pinned and it is still pinned after the iterator
  // is destroyed as long as cleanup functions are moved to another object,
  // when:
  // 1. block cache handle is set to be released in cleanup function, or
  // 2. it's pointing to immortal source. If own_bytes is true then we are
  //    not reading data from the original source, whether immortal or not.
  //    Otherwise, the block is pinned iff the source is immortal.
  // const bool block_contents_pinned =
  //     block.IsCached() ||
  //     (!block.GetValue()->own_bytes() && rep_->immortal_table);
  iter = InitBlockIterator<TBlockIter>(rep_, block.GetValue(), block_type, iter,
                                       true /* block_contents_pinned (not used for GDS) */, buffer, stream);

  // if (!block.IsCached()) {
  //   if (!ro.fill_cache) {
  //     IterPlaceholderCacheInterface block_cache{
  //         rep_->table_options.block_cache.get()};
  //     if (block_cache) {
  //       // insert a dummy record to block cache to track the memory usage
  //       Cache::Handle* cache_handle = nullptr;
  //       CacheKey key =
  //           CacheKey::CreateUniqueForCacheLifetime(block_cache.get());
  //       s = block_cache.Insert(key.AsSlice(),
  //                              block.GetValue()->ApproximateMemoryUsage(),
  //                              &cache_handle);

  //       if (s.ok()) {
  //         assert(cache_handle != nullptr);
  //         iter->RegisterCleanup(&ForceReleaseCachedEntry, block_cache.get(),
  //                               cache_handle);
  //       }
  //     }
  //   }
  // } else {
  //   iter->SetCacheHandle(block.GetCacheHandle());
  // }

  block.TransferTo(iter);

  return iter;
}

template <typename TBlocklike /*, auto*/>
WithBlocklikeCheck<Status, TBlocklike> GDSBlockBasedTableReader::RetrieveBlock(
    FilePrefetchBuffer* prefetch_buffer, const ReadOptions& ro,
    const BlockHandle& handle, UnownedPtr<Decompressor> decomp,
    CachableEntry<TBlocklike>* out_parsed_block, GetContext* get_context,
    BlockCacheLookupContext* lookup_context, bool for_compaction,
    bool use_cache, bool async_read, bool use_block_cache_for_lookup, cudaStream_t stream) const {
  assert(out_parsed_block);
  assert(out_parsed_block->IsEmpty());


  if (use_cache) {
    return Status::NotSupported("Cache is disabled for GDS reads");
  }

  Status s;

  assert(out_parsed_block->IsEmpty());

  const bool no_io = ro.read_tier == kBlockCacheTier;
  if (no_io) {
    return Status::Incomplete("no blocking io");
  }

  const bool maybe_compressed =
      BlockTypeMaybeCompressed(TBlocklike::kBlockType) && rep_->decompressor;


  if (maybe_compressed) {
    return Status::NotSupported("Compression is disabled for GDS reads");
  }

  std::unique_ptr<TBlocklike> block;

  {
    Histograms histogram =
        for_compaction ? READ_BLOCK_COMPACTION_MICROS : READ_BLOCK_GET_MICROS;
    StopWatch sw(rep_->ioptions.clock, rep_->ioptions.stats, histogram);
    s = GDSReadAndParseBlockFromFile(
        gds_file_.get(), prefetch_buffer, rep_->footer, ro, handle, &block,
        rep_->ioptions, rep_->create_context, maybe_compressed, decomp,
        rep_->persistent_cache_options, GetMemoryAllocator(rep_->table_options),
        for_compaction, async_read, stream);

    if (get_context) {
      switch (TBlocklike::kBlockType) {
        case BlockType::kIndex:
          ++(get_context->get_context_stats_.num_index_read);
          break;
        case BlockType::kFilter:
        case BlockType::kFilterPartitionIndex:
          ++(get_context->get_context_stats_.num_filter_read);
          break;
        default:
          break;
      }
    }
  }

  if (!s.ok()) {
    return s;
  }

  out_parsed_block->SetOwnedValue(std::move(block));

  assert(s.ok());
  return s;
}




// ------------------------------------------------------------------------------------------------
// GDSGetContext implementation
// ------------------------------------------------------------------------------------------------

GDSGetContext::GDSGetContext(const Comparator* ucmp, const MergeOperator* merge_operator,
    Logger* logger, Statistics* statistics, GetState init_state,
    const Slice& user_key, DeviceSlice* value,
    PinnableWideColumns* columns, std::string* timestamp,
    bool* value_found, MergeContext* merge_context, bool do_merge,
    SequenceNumber* max_covering_tombstone_seq, SystemClock* clock,
    SequenceNumber* seq,
    PinnedIteratorsManager* _pinned_iters_mgr,
    ReadCallback* callback, bool* is_blob_index,
    uint64_t tracing_get_id, BlobFetcher* blob_fetcher) : GetContext(ucmp, merge_operator, logger, statistics, init_state, user_key, nullptr, columns, timestamp, value_found, merge_context, do_merge, max_covering_tombstone_seq, clock, seq, _pinned_iters_mgr, callback, is_blob_index, tracing_get_id, blob_fetcher),
     device_value_(value) {}


bool GDSGetContext::SaveValue(const ParsedInternalKey& parsed_key, const DeviceSlice& value, bool* matched, Status* read_status, Cleanable* value_pinner, cudaStream_t stream) {
  assert(matched);
  assert((state_ != kMerge && parsed_key.type != kTypeMerge) ||
         merge_context_ != nullptr);
  
  // ============================================================================
  // 1. Key Matching Check (CPU-side)
  // ============================================================================
  if (ucmp_->EqualWithoutTimestamp(parsed_key.user_key, user_key_)) {
    *matched = true;
    
    // If the value is not in the snapshot, skip it
    if (!CheckCallback(parsed_key.sequence)) {
      return true;  // to continue to the next seq
    }

    // ============================================================================
    // 2. Sequence Number Handling (CPU-side)
    // ============================================================================
    if (seq_ != nullptr) {
      // Set the sequence number if it is uninitialized
      if (*seq_ == kMaxSequenceNumber) {
        *seq_ = parsed_key.sequence;
      }
      if (max_covering_tombstone_seq_) {
        *seq_ = std::max(*seq_, *max_covering_tombstone_seq_);
      }
    }

    // ============================================================================
    // 3. Timestamp Handling (CPU-side)
    // ============================================================================
    size_t ts_sz = ucmp_->timestamp_size();
    Slice ts;

    if (ts_sz > 0) {
      // ensure always have ts if cf enables ts.
      ts = ExtractTimestampFromUserKey(parsed_key.user_key, ts_sz);
      if (timestamp_ != nullptr) {
        if (!timestamp_->empty()) {
          assert(ts_sz == timestamp_->size());
          // `timestamp` can be set before `SaveValue` is ever called
          // when max_covering_tombstone_seq_ was set.
          // If this key has a higher sequence number than range tombstone,
          // then timestamp should be updated. `ts_from_rangetombstone_` is
          // set to false afterwards so that only the key with highest seqno
          // updates the timestamp.
          if (ts_from_rangetombstone_) {
            assert(max_covering_tombstone_seq_);
            if (parsed_key.sequence > *max_covering_tombstone_seq_) {
              timestamp_->assign(ts.data(), ts.size());
              ts_from_rangetombstone_ = false;
            }
          }
        }
        // TODO optimize for small size ts
        const std::string kMaxTs(ts_sz, '\xff');
        if (timestamp_->empty() ||
            ucmp_->CompareTimestamp(*timestamp_, kMaxTs) == 0) {
          timestamp_->assign(ts.data(), ts.size());
        }
      }
    }
    
    // For replay log, we need to copy value from GPU to CPU temporarily
    // This is unavoidable for the replay log feature
    if (replay_log_) {
      std::vector<char> cpu_value_buf(value.size());
      cudaError_t err = cudaMemcpyAsync(cpu_value_buf.data(), value.data(), 
                                    value.size(), cudaMemcpyDeviceToHost, stream);
      cudaStreamSynchronize(stream);
      std::cerr << "Warning: cudaMemcpyDeviceToHost called in " << std::string(__FILE__) << ":" << std::to_string(__LINE__) << std::endl;
      if (err != cudaSuccess) {
        *read_status = Status::IOError("Failed to copy value for replay log: " + 
                                       std::string(cudaGetErrorString(err)));
        return false;
      }
      Slice cpu_value(cpu_value_buf.data(), cpu_value_buf.size());
      appendToReplayLog(parsed_key.type, cpu_value, ts);
    }

    // ============================================================================
    // 4. Type-based Processing
    // ============================================================================
    auto type = parsed_key.type;
    DeviceSlice unpacked_value = value;
    // Check if covered by range deletion
    if ((type == kTypeValue || type == kTypeValuePreferredSeqno ||
         type == kTypeMerge || type == kTypeBlobIndex ||
         type == kTypeWideColumnEntity || type == kTypeDeletion ||
         type == kTypeDeletionWithTimestamp || type == kTypeSingleDeletion) &&
        max_covering_tombstone_seq_ != nullptr &&
        *max_covering_tombstone_seq_ > parsed_key.sequence) {
      // Note that deletion types are also considered, this is for the case
      // when we need to return timestamp to user. If a range tombstone has a
      // higher seqno than point tombstone, its timestamp should be returned.
      type = kTypeRangeDeletion;
    }
    
    switch (type) {
      // ========================================================================
      // 4A. Value Types (kTypeValue, kTypeBlobIndex, etc.)
      // ========================================================================
      case kTypeValue:
      case kTypeValuePreferredSeqno:
      case kTypeBlobIndex:
      case kTypeWideColumnEntity:
        assert(state_ == kNotFound || state_ == kMerge);
        
        // Handle kTypeValuePreferredSeqno: need to unpack value
        // This requires GPU-side unpacking or D2H copy
        if (type == kTypeValuePreferredSeqno) {
          // TODO: For now, we need to copy to CPU to parse packed value
          // Future optimization: implement GPU-side ParsePackedValueForValue
          std::vector<char> cpu_buf(value.size());
          cudaError_t err = cudaMemcpyAsync(cpu_buf.data(), value.data(),
                                        value.size(), cudaMemcpyDeviceToHost, stream);
          cudaStreamSynchronize(stream);
          std::cerr << "Warning: cudaMemcpyDeviceToHost called in " << std::string(__FILE__) << ":" << std::to_string(__LINE__) << std::endl;
          if (err != cudaSuccess) {
            state_ = kCorrupt;
            *read_status = Status::IOError("Failed to copy packed value: " +
                                           std::string(cudaGetErrorString(err)));
            return false;
          }
          Slice cpu_value(cpu_buf.data(), cpu_buf.size());
          Slice unpacked_cpu = ParsePackedValueForValue(cpu_value);
          
          // Adjust unpacked_value to point to the unpacked portion on GPU
          // The unpacked value is at the same offset in GPU memory
          size_t offset = unpacked_cpu.data() - cpu_value.data();
          unpacked_value = DeviceSlice(value.data() + offset, unpacked_cpu.size());
        }
        
        if (type == kTypeBlobIndex) {
          // GDS does not support blob index operations
          state_ = kCorrupt;
          *read_status = Status::NotSupported(
              "GDS SaveValue does not support kTypeBlobIndex - "
              "blob values are not supported in GDS mode");
          return false;
        }

        if (is_blob_index_ != nullptr) {
          *read_status = Status::NotSupported("GDS does not support blob index operations");
          *is_blob_index_ = false;  // GDS never returns blob index
          return false;
        }

        // ----------------------------------------------------------------------
        // State: kNotFound -> kFound
        // ----------------------------------------------------------------------
        if (kNotFound == state_) {
          state_ = kFound;
          if (do_merge_) {
            // Note: kTypeBlobIndex is not supported and already rejected above
            
            // For GDS: store the GPU value pointer directly
            if (device_value_ != nullptr) {
              DeviceSlice value_to_use = unpacked_value;
              
              // Handle kTypeWideColumnEntity
              if (type == kTypeWideColumnEntity) {
                // TODO: This requires parsing wide column format on GPU
                // For now, we need to copy to CPU to parse
                std::vector<char> cpu_buf(unpacked_value.size());
                MLKV_CUDA_CHECK(cudaMemcpyAsync(cpu_buf.data(), unpacked_value.data(),
                                              unpacked_value.size(), 
                                              cudaMemcpyDeviceToHost, stream));
                cudaStreamSynchronize(stream);
                std::cerr << "Warning: cudaMemcpyDeviceToHost called in " << std::string(__FILE__) << ":" << std::to_string(__LINE__) << std::endl;

                Slice cpu_value(cpu_buf.data(), cpu_buf.size());
                Slice value_of_default;
                
                if (!WideColumnSerialization::GetValueOfDefaultColumn(
                         cpu_value, value_of_default).ok()) {
                  state_ = kCorrupt;
                  return false;
                }
                
                // Adjust pointer to default column in GPU memory
                size_t offset = value_of_default.data() - cpu_value.data();
                value_to_use = DeviceSlice(unpacked_value.data() + offset, 
                                           value_of_default.size());
              }
              
              // Store the GPU value directly - no D2H copy needed!
              *device_value_ = value_to_use;
              
              // TODO: If value_pinner is provided, we should pin the GPU memory
              // to prevent it from being freed. This might require a custom
              // cleanup mechanism for GPU memory.
              
            } else if (columns_ != nullptr) {
              // Wide columns case: need to copy to CPU for now
              // TODO: Future optimization - keep columns on GPU
              std::vector<char> cpu_buf(unpacked_value.size());
              MLKV_CUDA_CHECK(cudaMemcpyAsync(cpu_buf.data(), unpacked_value.data(),
                                            unpacked_value.size(),
                                            cudaMemcpyDeviceToHost, stream));
              cudaStreamSynchronize(stream);
              std::cerr << "Warning: cudaMemcpyDeviceToHost called in " << std::string(__FILE__) << ":" << std::to_string(__LINE__) << std::endl;
              Slice cpu_value(cpu_buf.data(), cpu_buf.size());
              
              if (type == kTypeWideColumnEntity) {
                if (!columns_->SetWideColumnValue(cpu_value, value_pinner).ok()) {
                  state_ = kCorrupt;
                  return false;
                }
              } else {
                columns_->SetPlainValue(cpu_value, value_pinner);
              }
            }
          } else {
            // GetMergeOperands API: need to handle merge operands
            // Note: Blob operations are not supported in GDS mode (already handled above)
            if (type == kTypeWideColumnEntity) {
              std::vector<char> cpu_buf(unpacked_value.size());
              MLKV_CUDA_CHECK(cudaMemcpyAsync(cpu_buf.data(), unpacked_value.data(),
                                            unpacked_value.size(),
                                            cudaMemcpyDeviceToHost, stream));
              cudaStreamSynchronize(stream);
              std::cerr << "Warning: cudaMemcpyDeviceToHost called in " << std::string(__FILE__) << ":" << std::to_string(__LINE__) << std::endl;

              Slice cpu_value(cpu_buf.data(), cpu_buf.size());
              Slice value_of_default;

              if (!WideColumnSerialization::GetValueOfDefaultColumn(
                       cpu_value, value_of_default).ok()) {
                state_ = kCorrupt;
                return false;
              }

              push_operand(value_of_default, value_pinner);
              
            } else {
              assert(type == kTypeValue || type == kTypeValuePreferredSeqno);
              // For plain value, need to copy to CPU for merge operands
              std::vector<char> cpu_buf(unpacked_value.size());
              MLKV_CUDA_CHECK(cudaMemcpyAsync(cpu_buf.data(), unpacked_value.data(),
                                            unpacked_value.size(),
                                            cudaMemcpyDeviceToHost, stream));
              cudaStreamSynchronize(stream);
              std::cerr << "Warning: cudaMemcpyDeviceToHost called in " << std::string(__FILE__) << ":" << std::to_string(__LINE__) << std::endl;

              Slice cpu_value(cpu_buf.data(), cpu_buf.size());
              push_operand(cpu_value, value_pinner);
            }
          }
          
        // ----------------------------------------------------------------------
        // State: kMerge -> kFound (with merge)
        // ----------------------------------------------------------------------
        } else if (kMerge == state_) {
          assert(merge_operator_ != nullptr);
          
          // Note: kTypeBlobIndex already handled above with error
          if (type == kTypeWideColumnEntity) {
            state_ = kFound;

            if (do_merge_) {
              // Need CPU copy for wide column merge
              std::vector<char> cpu_buf(unpacked_value.size());
              MLKV_CUDA_CHECK(cudaMemcpyAsync(cpu_buf.data(), unpacked_value.data(),
                                            unpacked_value.size(),
                                            cudaMemcpyDeviceToHost, stream));
              cudaStreamSynchronize(stream);
              std::cerr << "Warning: cudaMemcpyDeviceToHost called in " << std::string(__FILE__) << ":" << std::to_string(__LINE__) << std::endl;

              Slice cpu_value(cpu_buf.data(), cpu_buf.size());
              MergeWithWideColumnBaseValue(cpu_value);
            } else {
              // GetMergeOperands path
              std::vector<char> cpu_buf(unpacked_value.size());
              MLKV_CUDA_CHECK(cudaMemcpyAsync(cpu_buf.data(), unpacked_value.data(),
                                            unpacked_value.size(),
                                            cudaMemcpyDeviceToHost, stream));
              cudaStreamSynchronize(stream);
              std::cerr << "Warning: cudaMemcpyDeviceToHost called in " << std::string(__FILE__) << ":" << std::to_string(__LINE__) << std::endl;
              Slice cpu_value(cpu_buf.data(), cpu_buf.size());
              Slice value_of_default;

              if (!WideColumnSerialization::GetValueOfDefaultColumn(
                       cpu_value, value_of_default).ok()) {
                state_ = kCorrupt;
                return false;
              }

              push_operand(value_of_default, value_pinner);
            }
            
          } else {
            assert(type == kTypeValue || type == kTypeValuePreferredSeqno);

            state_ = kFound;
            if (do_merge_) {
              // Need CPU copy for merge operation
              std::vector<char> cpu_buf(unpacked_value.size());
              MLKV_CUDA_CHECK(cudaMemcpyAsync(cpu_buf.data(), unpacked_value.data(),
                                            unpacked_value.size(),
                                            cudaMemcpyDeviceToHost, stream));
              cudaStreamSynchronize(stream);
              std::cerr << "Warning: cudaMemcpyDeviceToHost called in " << std::string(__FILE__) << ":" << std::to_string(__LINE__) << std::endl;
              Slice cpu_value(cpu_buf.data(), cpu_buf.size());
              MergeWithPlainBaseValue(cpu_value);
            } else {
              // GetMergeOperands API
              std::vector<char> cpu_buf(unpacked_value.size());
              MLKV_CUDA_CHECK(cudaMemcpyAsync(cpu_buf.data(), unpacked_value.data(),
                                            unpacked_value.size(),
                                            cudaMemcpyDeviceToHost, stream));
              cudaStreamSynchronize(stream);
              std::cerr << "Warning: cudaMemcpyDeviceToHost called in " << std::string(__FILE__) << ":" << std::to_string(__LINE__) << std::endl;
              Slice cpu_value(cpu_buf.data(), cpu_buf.size());
              push_operand(cpu_value, value_pinner);
            }
          }
        }
        return false;

      // ========================================================================
      // 4B. Deletion Types
      // ========================================================================
      case kTypeDeletion:
      case kTypeDeletionWithTimestamp:
      case kTypeSingleDeletion:
      case kTypeRangeDeletion:
        // TODO(noetzli): Verify correctness once merge of single-deletes
        // is supported
        assert(state_ == kNotFound || state_ == kMerge);
        if (kNotFound == state_) {
          state_ = kDeleted;
        } else if (kMerge == state_) {
          state_ = kFound;
          if (do_merge_) {
            MergeWithNoBaseValue();
          }
          // If do_merge_ = false then the current value shouldn't be part of
          // merge_context_->operand_list
        }
        return false;

      // ========================================================================
      // 4C. Merge Type
      // ========================================================================
      case kTypeMerge:
        assert(state_ == kNotFound || state_ == kMerge);
        state_ = kMerge;
        
        // Merge operands need to be stored - requires CPU copy for now
        // value_pinner is not set from plain_table_reader.cc for example.
        {
          std::vector<char> cpu_buf(value.size());
          MLKV_CUDA_CHECK(cudaMemcpyAsync(cpu_buf.data(), value.data(),
                                        value.size(), cudaMemcpyDeviceToHost, stream));
          cudaStreamSynchronize(stream);
          std::cerr << "Warning: cudaMemcpyDeviceToHost called in " << std::string(__FILE__) << ":" << std::to_string(__LINE__) << std::endl;
          Slice cpu_value(cpu_buf.data(), cpu_buf.size());
          push_operand(cpu_value, value_pinner);
          
          PERF_COUNTER_ADD(internal_merge_point_lookup_count, 1);

          if (do_merge_ && merge_operator_ != nullptr &&
              merge_operator_->ShouldMerge(
                  merge_context_->GetOperandsDirectionBackward())) {
            state_ = kFound;
            MergeWithNoBaseValue();
            return false;
          }
          if (merge_context_->get_merge_operands_options != nullptr &&
              merge_context_->get_merge_operands_options->continue_cb !=
                  nullptr &&
              !merge_context_->get_merge_operands_options->continue_cb(cpu_value)) {
            state_ = kFound;
            return false;
          }
        }
        return true;

      default:
        assert(false);
        break;
    }
  }

  // state_ could be Corrupt, merge or notfound
  return false;
}




// Read and parse block from file using GDS
// This function:
// 1. Uses GDS to read block data to GPU memory for high-speed I/O
// 2. Copies data from GPU to CPU (since TBlocklike is CPU-side)
// 3. Creates TBlocklike object using BlockCreateContext
//
// Note: Block cache and compression are disabled as per requirements
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
    cudaStream_t stream) {


  // std::cout << "[GDS DEBUG] GDSReadAndParseBlockFromFile Invoked" << std::endl;
  
  assert(result);
  
  // Compression is disabled per requirements, so we expect uncompressed blocks
  if (maybe_compressed) {
    return Status::NotSupported("Compression is disabled for GDS reads");
  }
  
  // Use GDS to read block to GPU memory
  DeviceBlockContents device_contents;
  Status s = GDSReadBlock(options, *file, handle, &device_contents, footer, stream);
  
  if (!s.ok()) {
    return s;
  }
  
  // Create DeviceBlock_kData and parse block footer
  // This may throw if block uses unsupported features (e.g. hash index)
  try {
    result->reset(new DeviceBlock_kData(std::move(device_contents), stream));
  } catch (const std::exception& e) {
    // Catch any exceptions during block initialization and convert to Status
    return Status::NotSupported(e.what());
  }

  // std::cout << "[GDS DEBUG] deviceblock restart offset: " << (*result)->GetRestartOffset() << std::endl;
  // std::cout << "[GDS DEBUG] deviceblock num restarts: " << (*result)->NumRestarts() << std::endl;
  // std::cout << "[GDS DEBUG] deviceblock index type: " 
  //           << ((*result)->IndexType() == BlockBasedTableOptions::kDataBlockBinarySearch ? "BinarySearch" : "BinaryAndHash") 
  //           << std::endl;
  
  // // Copy data from GPU to CPU
  // cudaError_t cuda_err = cudaMemcpy(heap_buf.get(), 
  //                                   device_contents.data.data(),
  //                                   block_size,
  //                                   cudaMemcpyDeviceToHost);
  
  // // Free GPU memory
  // if (device_contents.own_bytes()) {
  //   cudaFree(device_contents.allocation);
  // }
  
  // if (cuda_err != cudaSuccess) {
  //   return Status::IOError("Failed to copy block data from GPU to CPU: " + 
  //                         std::string(cudaGetErrorString(cuda_err)));
  // }
  
  // // Create BlockContents from CPU memory
  // BlockContents contents(std::move(heap_buf), block_size);

  
  // // Create TBlocklike object
  // create_context.Create(result, std::move(contents));
  
  return Status::OK();
}


// Read block from storage via GDS to GPU memory
// This function:
// 1. Reads block + trailer from file to GPU memory using GDS
// 2. Copies trailer to CPU for checksum verification
// 3. Verifies checksum if requested
// 4. Returns block data (without trailer) in GPU memory
Status GDSReadBlock(
    const ReadOptions& ro, 
    GDSRandomAccessFile& file,  // Changed to non-const for Read operation
    const BlockHandle& handle,  // Use CPU-side BlockHandle for compatibility
    DeviceBlockContents* contents,
    const Footer& footer,
    cudaStream_t stream) {
  
  assert(contents);
  
  // Calculate sizes
  const uint64_t block_size = handle.size();
  const size_t trailer_size = footer.GetBlockTrailerSize();
  const size_t block_size_with_trailer = block_size + trailer_size;
  

  
  // Read block + trailer via GDS
  IOOptions io_opts;
  IODebugContext dbg;
  DeviceSlice result;
  IOStatus io_status = file.Read(handle.offset(), block_size_with_trailer,
                                          io_opts, &result, nullptr /* scratch */, &dbg, stream);
  
  if (!io_status.ok()) {
    return Status::IOError("GDS read failed: " + io_status.message());
  }
  
  if (result.size() != block_size_with_trailer) {
    return Status::Corruption(
        "Truncated block read from " + file.GetFileName() + 
        " offset " + std::to_string(handle.offset()) + 
        ", expected " + std::to_string(block_size_with_trailer) + 
        " bytes, got " + std::to_string(result.size()));
  }
  

  // TODO: Implement GPU-side checksum verification
  // // Verify checksum if requested
  // if (trailer_size > 0 && ro.verify_checksums) {
  //   // Copy trailer to CPU for verification
  //   std::unique_ptr<char[]> trailer_buf(new char[trailer_size]);
  //   cuda_err = cudaMemcpy(trailer_buf.get(), 
  //                        gpu_buffer + block_size,
  //                        trailer_size,
  //                        cudaMemcpyDeviceToHost);
  //   if (cuda_err != cudaSuccess) {
  //     cudaFree(gpu_buffer);
  //     return Status::IOError("Failed to copy trailer to CPU: " + 
  //                           std::string(cudaGetErrorString(cuda_err)));
  //   }
    

    
  //   // Create a temporary slice for checksum verification
  //   // Note: We need to copy the data block to CPU for verification
  //   std::unique_ptr<char[]> data_buf(new char[block_size]);
  //   cuda_err = cudaMemcpy(data_buf.get(), gpu_buffer, block_size,
  //                        cudaMemcpyDeviceToHost);
  //   if (cuda_err != cudaSuccess) {
  //     cudaFree(gpu_buffer);
  //     return Status::IOError("Failed to copy data to CPU for verification: " + 
  //                           std::string(cudaGetErrorString(cuda_err)));
  //   }
  //   // Reconstruct full block with trailer for verification
  //   std::unique_ptr<char[]> full_buf(new char[block_size_with_trailer]);
  //   memcpy(full_buf.get(), data_buf.get(), block_size);
  //   memcpy(full_buf.get() + block_size, trailer_buf.get(), trailer_size);
    
  //   Status s = VerifyBlockChecksum(footer, full_buf.get(), block_size,
  //                                  file.GetFileName(), handle.offset());
  //   if (!s.ok()) {
  //     cudaFree(gpu_buffer);
  //     return s;
  //   }
  // }
  
  // Fill DeviceBlockContents with block data (without trailer)
  *contents = DeviceBlockContents(DeviceSlice(result.data(), block_size));
  
  return Status::OK();
}





} // namespace gds

} // namespace mlkv_plus
