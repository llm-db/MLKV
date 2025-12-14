#include "gds_block_based_table_factory.cuh"
#include "gds_block_based_table_reader.cuh"
#include "rocksdb/table.h"

#include "rocksdb/file/random_access_file_reader.h"

#include "rocksdb/options/cf_options.h"
#include "rocksdb/table/table_builder.h"

#include "gds_file_system.cuh"

using namespace ROCKSDB_NAMESPACE;

namespace mlkv_plus {

namespace gds {

const BlockBasedTableOptions& GDSBlockBasedTableFactory::GetTableOptions() const {
    return *GetOptions<BlockBasedTableOptions>();
}

Status GDSBlockBasedTableFactory::NewTableReader(
    const ReadOptions& ro, const TableReaderOptions& table_reader_options,
    std::unique_ptr<RandomAccessFileReader>&& file, uint64_t file_size,
    std::unique_ptr<TableReader>* table_reader,
    bool prefetch_index_and_filter_in_cache) const {


    (void) file;



    // Create GDS-backed file reader
    std::unique_ptr<GDSRandomAccessFile> gds_file =
        std::make_unique<GDSRandomAccessFile>(file->file_name(), *gds_options_);


    ImmutableOptions ioptions = table_reader_options.ioptions;
        
    // Wrap it in RocksDB's RandomAccessFileReader
    // std::unique_ptr<RandomAccessFileReader> file_reader(
    //     new RandomAccessFileReader(std::move(gds_file), file->file_name(), 
    //                             ioptions.clock, nullptr, ioptions.stats,
    //                             Histograms::SST_READ_MICROS, nullptr,
    //                             ioptions.rate_limiter.get(),
    //                             ioptions.listeners,
    //                             Temperature::kUnknown,
    //                             table_reader_options.level == ioptions.num_levels - 1));


    return GDSBlockBasedTableReader::Open(
        ro, table_reader_options.ioptions, table_reader_options.env_options,
        table_options_, table_reader_options.internal_comparator, std::move(file),
        std::move(gds_file), file_size, table_reader_options.block_protection_bytes_per_key,
        table_reader, table_reader_options.tail_size,
        shared_state_->table_reader_cache_res_mgr,
        table_reader_options.prefix_extractor,
        table_reader_options.compression_manager,
        prefetch_index_and_filter_in_cache, table_reader_options.skip_filters,
        table_reader_options.level, table_reader_options.immortal,
        table_reader_options.largest_seqno,
        table_reader_options.force_direct_prefetch,
        &shared_state_->tail_prefetch_stats,
        table_reader_options.block_cache_tracer,
        table_reader_options.max_file_size_for_l0_meta_pin,
        table_reader_options.cur_db_session_id, table_reader_options.cur_file_num,
        table_reader_options.unique_id,
        table_reader_options.user_defined_timestamps_persisted, 
        *gds_options_);



}



} // namespace gds

} // namespace mlkv_plus