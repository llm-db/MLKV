#pragma once

#include "rocksdb/table/block_based/block_based_table_factory.h"
#include "utils.cuh"
#include <memory>

using namespace ROCKSDB_NAMESPACE;


namespace mlkv_plus {

namespace gds {

class GDSBlockBasedTableFactory : public BlockBasedTableFactory {
public:
    GDSBlockBasedTableFactory(const BlockBasedTableOptions& options, const GDSOptions& gds_options)
        : BlockBasedTableFactory(options), gds_options_(std::make_unique<GDSOptions>(gds_options)) {}

    // Override NewTableReader to create GDS version
    using TableFactory::NewTableReader;
    Status NewTableReader(
        const ReadOptions& ro, const TableReaderOptions& table_reader_options,
        std::unique_ptr<RandomAccessFileReader>&& file, uint64_t file_size,
        std::unique_ptr<TableReader>* table_reader,
        bool prefetch_index_and_filter_in_cache = true) const override;

    // Method to allow CheckedCast to work for this class
    static const char* kClassName() { return "GDSBlockBasedTableFactory"; }

    const char* Name() const override { return kClassName(); }

    // Helper method to access table options
    const BlockBasedTableOptions& GetTableOptions() const;

private:
    std::unique_ptr<GDSOptions> gds_options_;
};

} // namespace gds

} // namespace mlkv_plus