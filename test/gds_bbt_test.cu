// Example usage of GDS BlockBasedTable implementation
// This shows how to use the GDS version of BlockBasedTableFactory and Reader

#include "gds/gds_block_based_table_factory.cuh"
#include "gds/gds_block_based_table_reader.cuh"
#include "rocksdb/table.h"
#include "rocksdb/options.h"

#include "rocksdb/db.h"

#include "gds/gds_db_accessor.cuh"

#include <iostream>


using namespace mlkv_plus::gds;


int main() {
    // 1. Set up GDS options
    GDSOptions gds_options;
    gds_options.GDSAlignment = 4096;  // 4KB alignment for optimal GDS performance
    gds_options.DefaultGPUBufferSize = 64 * 1024 * 1024;  // 64MB default buffer
    
    // 2. Set up BlockBasedTable options
    BlockBasedTableOptions table_options;
    table_options.no_block_cache = true;
    table_options.persistent_cache = nullptr;
    table_options.data_block_index_type = BlockBasedTableOptions::kDataBlockBinarySearch;


    // 3. Create GDS-enabled table factory
    std::shared_ptr<TableFactory> gds_factory(
        new GDSBlockBasedTableFactory(table_options, gds_options));


    
    // 4. Use the factory in RocksDB options
    Options db_options;

    db_options.create_if_missing = true;
    db_options.compression = kNoCompression;
    db_options.comparator = rocksdb::BytewiseComparator();
    // db_options.table_factory.reset(NewBlockBasedTableFactory(table_options));
    db_options.table_factory = gds_factory;

    std::cout << "GDS BlockBasedTable factory created successfully!" << std::endl;
    std::cout << "Factory name: " << gds_factory->Name() << std::endl;


    // 5. Create a database and open it
    DB* db;
    Status s = DB::Open(db_options, "gds_bbt_test", &db);
    if (!s.ok()) {
        std::cerr << "Error opening database: " << s.ToString() << std::endl;
        return 1;
    }

    std::cout << "Database opened successfully!" << std::endl;

    // // 6. Put some data into the database
    // db->Put(WriteOptions(), "key1", "value1");
    // db->Put(WriteOptions(), "key2", "value2");
    // db->Put(WriteOptions(), "key3", "value3");

    DeviceSlice value;
    std::string value_str;
    ReadOptions read_options;
    read_options.async_io = false;
    read_options.read_tier = kPersistedTier;
    s = db->Get(read_options, "key1", &value_str);
    std::cout << "key1 -> " << value_str << " " << s.ToString() << std::endl;

    char* key1;


    cudaMalloc(&key1, 4);
    cudaMemcpy(key1, "key1", 4, cudaMemcpyHostToDevice);

    GDSDataBlockIterDevicePinnedBuffer* gds_buffer = mlkv_plus::gds::GDSDataBlockIterDevicePinnedBufferManager::Allocate();
    mlkv_plus::OperationResult status = GDSDBAccessor::GetFromSST(db, read_options, DeviceSlice(key1, 4), rocksdb::Slice("key1"), &value, gds_buffer);
    std::cout << "value.size() -> " << value.size() << std::endl;

    char* h_value = new char[value.size()];
    cudaMemcpy(h_value, value.data(), value.size(), cudaMemcpyDeviceToHost);
    std::cout << "key1 -> " << std::string(h_value, value.size()) << " " << (int) status << std::endl;


    // std::cout << "--------------------------------" << std::endl;
    // // 8. from sst
    // ReadOptions read_options_sst;
    // read_options_sst.read_tier = kPersistedTier;
    // s = db->Get(read_options_sst, "key1", &value);
    // std::cout << "key1 -> " << value << " " << s.ToString() << std::endl;
    // s = db->Get(read_options_sst, "key2", &value);
    // std::cout << "key2 -> " << value << " " << s.ToString() << std::endl;
    // s = db->Get(read_options_sst, "key3", &value);
    // std::cout << "key3 -> " << value << " " << s.ToString() << std::endl;

    mlkv_plus::gds::GDSDataBlockIterDevicePinnedBufferManager::Free(gds_buffer);


    delete db;
    

    return 0;
}
