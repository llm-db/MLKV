#include "gds_db_accessor.cuh"
#include "gds_block_based_table_reader.cuh"

#include "rocksdb/db.h"
#include "rocksdb/snapshot.h"

#include "utils.cuh"
#include "db/db_impl/db_impl.h"
#include "db/column_family.h"
#include "db/version_set.h"
#include "db/lookup_key.h"
#include "table/table_reader.h"
#include "util/autovector.h"
#include "db/merge_context.h"
#include <cuda_runtime.h>


namespace mlkv_plus {
namespace gds {

OperationResult GDSDBAccessor::GetFromSST(rocksdb::DB* db, const rocksdb::ReadOptions& read_options, const DeviceSlice& key, const Slice& host_key, DeviceSlice* value, GDSDataBlockIterDevicePinnedBuffer* buffer, cudaStream_t stream) {

    
    // Cast to DBImpl to access internal methods
    auto* db_impl = static_cast_with_check<DBImpl>(db);
    
    // Get default column family
    auto cfh = static_cast_with_check<ColumnFamilyHandleImpl>(db->DefaultColumnFamily());
    auto cfd = cfh->cfd();
    
    // Determine snapshot sequence number
    SequenceNumber snapshot;
    if (read_options.snapshot != nullptr) {
        snapshot = static_cast<const SnapshotImpl*>(read_options.snapshot)->number_;
    } else {
        snapshot = db_impl->GetLastPublishedSequence();
    }
    
    // Create host-side Slice for the key (copy from device for LookupKey)
    size_t key_size = key.size();
    if (key_size == 0 || key.data() == nullptr) {
        std::cerr << "Invalid key: null or empty at " << __FILE__ << ":" << __LINE__ << std::endl;
        return OperationResult::INVALID_PARAMETER;
    }
    
    // Create LookupKey (internal key format with snapshot)
    LookupKey lkey(host_key, snapshot, nullptr);  // no timestamp
    Slice ikey = lkey.internal_key();
    char* d_ikey;
    MLKV_CUDA_CHECK(cudaMallocAsync(&d_ikey, ikey.size(), stream));
    MLKV_CUDA_CHECK(cudaMemcpyAsync(d_ikey, ikey.data(), ikey.size(), cudaMemcpyHostToDevice, stream));
    DeviceSlice d_ikey_slice(d_ikey, ikey.size());
    
    // Acquire SuperVersion
    SuperVersion* sv = db_impl->GetAndRefSuperVersion(cfd);
    if (sv == nullptr) {
        std::cerr << "Failed to get SuperVersion at " << __FILE__ << ":" << __LINE__ << std::endl;
        return OperationResult::ROCKSDB_ERROR;
    }
    
    // Get the current version
    Version* current = sv->current;
    VersionStorageInfo* vstorage = current->storage_info();
    
    // Build level_files_brief vector for FilePicker
    autovector<LevelFilesBrief> level_files_brief;
    unsigned int num_non_empty = static_cast<unsigned int>(vstorage->num_non_empty_levels());
    for (unsigned int level = 0; level < num_non_empty; ++level) {
        level_files_brief.push_back(vstorage->LevelFilesBrief(level));
    }
    
    // Create FilePicker to traverse SST files
    FilePicker fp(
        host_key,                                  // user_key
        ikey,                                                // internal_key
        &level_files_brief,                      // file_levels
        num_non_empty,                           // num_levels
        &(vstorage->file_indexer()),          // file_indexer
        cfd->user_comparator(),              // user_comparator
        &(cfd->internal_comparator())             // internal_comparator (pointer)
    );
    
    // Status tracking
    Status status = Status::OK();
    IOStatus io_status = IOStatus::OK();
    bool value_found = false;
    
    // Get table cache and mutable options
    TableCache* table_cache = cfd->table_cache();
    const MutableCFOptions& mutable_cf_options = sv->mutable_cf_options;
    VersionSet* vset = current->version_set();
    const FileOptions& file_opts = vset->file_options();
    const SliceTransform* prefix_extractor = mutable_cf_options.prefix_extractor.get();
    
    // Iterate through files using FilePicker
    FdWithKeyRange* f = fp.GetNextFile();
    
    while (f != nullptr) {
        // Get file metadata and level
        const FileMetaData* file_meta = f->file_metadata;
        int level = fp.GetHitFileLevel();
        bool is_last_file_in_level = fp.IsHitFileLastInLevel();
        
        // Determine if filters should be skipped
        bool skip_filters = false;
        if (cfd->ioptions().optimize_filters_for_hits && 
            (level > 0 || is_last_file_in_level) &&
            level == num_non_empty - 1) {
            skip_filters = true;
        }
        
        // Try to get the table reader from cache
        TableCache::TypedHandle* handle = nullptr;
        status = table_cache->FindTable(
            read_options,
            file_opts,
            cfd->internal_comparator(),
            *file_meta,
            &handle,
            mutable_cf_options,
            /*no_io=*/false,
            cfd->internal_stats()->GetFileReadHist(level),
            skip_filters,
            level,
            /*prefetch_index_and_filter_in_cache=*/false,
            MaxFileSizeForL0MetaPin(mutable_cf_options),
            file_meta->temperature
        );
        
        if (!status.ok()) {
            if (handle) {
                table_cache->get_cache().Release(handle);
            }
            io_status = IOStatus::IOError(status.ToString());
            break;
        }
        
        // Get the table reader
        TableReader* table_reader = table_cache->get_cache().Value(handle);
        
        // Check if it's a GDS BlockBasedTableReader
        auto* gds_reader = dynamic_cast<GDSBlockBasedTableReader*>(table_reader);
        if (gds_reader != nullptr) {
            // Prepare merge operator and merge context
            auto* merge_op = cfd->ioptions().merge_operator.get();
            ROCKSDB_NAMESPACE::MergeContext merge_ctx;
            const bool do_merge = (merge_op != nullptr);
            // Create GDSGetContext for this lookup
            GDSGetContext get_context(
                cfd->user_comparator(),           // ucmp
                merge_op,                         // merge_operator
                cfd->ioptions().logger,           // logger
                cfd->ioptions().stats,            // statistics
                GetContext::kNotFound,            // init_state
                host_key,                         // user_key
                value,                            // device value output
                nullptr,                          // columns (not used)
                nullptr,                          // timestamp (not used)
                &value_found,                     // value_found
                &merge_ctx,                       // merge_context
                value,                         // do_merge
                nullptr,                          // max_covering_tombstone_seq
                cfd->ioptions().clock,            // clock
                nullptr,                          // seq
                nullptr,                          // pinned_iters_mgr
                nullptr,                          // callback
                nullptr,                          // is_blob_index
                0,                                // tracing_get_id
                nullptr                          // blob_fetcher
            );
            
            // Use GDS reader
            status = gds_reader->Get(
                read_options,
                ikey,                             // internal key (host)
                d_ikey_slice,                     // internal key (device)
                &get_context,                     // GDS get context
                prefix_extractor,                 // prefix extractor
                skip_filters,                     // skip_filters
                buffer,                           // buffer
                stream                             // stream
            );
            
            if (!status.ok()) {
                table_cache->get_cache().Release(handle);
                io_status = IOStatus::IOError(status.ToString());
                break;
            }
            
            // Check if value was found
            if (get_context.State() == GetContext::kFound) {
                value_found = true;
                table_cache->get_cache().Release(handle);
                break;
            } else if (get_context.State() == GetContext::kCorrupt) {
                table_cache->get_cache().Release(handle);
                io_status = IOStatus::IOError("Corrupted key");
                break;
            }
            // If kNotFound, continue to next file
        } else {
            // Fallback: not a GDS reader
            io_status = IOStatus::UnsupportedError("Table reader is not GDS-enabled");
            table_cache->get_cache().Release(handle);
            break;
        }
        
        // Release the table handle
        table_cache->get_cache().Release(handle);
        
        // Get next file
        f = fp.GetNextFile();
    }
    
    // Release SuperVersion
    db_impl->ReturnAndCleanupSuperVersion(cfd, sv);
    
    // Return appropriate status
    if (value_found) {
        return OperationResult::SUCCESS;
    } else if (!io_status.ok()) {
        std::cerr << "Failed to get from SST files: " << io_status.message() << std::endl;
        return OperationResult::ROCKSDB_ERROR;
    } else {
        return OperationResult::KEY_NOT_FOUND;
    }
}


} // namespace gds
} // namespace mlkv_plus
