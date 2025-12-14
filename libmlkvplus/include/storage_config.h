#pragma once

#include <cstdint>
#include <string>
#include <memory>

namespace mlkv_plus {

// Storage level enumeration
enum class StorageLevel {
    HBM = 0,    // GPU HBM
    DRAM = 1,   // CPU DRAM  
    DISK = 2    // Disk (RocksDB)
};


template<typename Key, typename Value>
struct EvictedData {
    size_t count;
    Key* keys;
    Value* values;    
};

// Storage configuration structure
struct StorageConfig {
    // HKV (HBM+DRAM) configuration
    size_t hkv_init_capacity = 64 * 1024 * 1024UL;  // 64M entries
    size_t hkv_max_capacity = 64 * 1024 * 1024UL;   // 64M entries
    size_t dim = 5;
    size_t max_hbm_for_vectors_gb = 0;              // HBM size in GB
    bool hkv_io_by_cpu = false;                      // Use CPU for I/O
    int gpu_id = 0;                                   // GPU ID
    size_t max_batch_size = 1048576;

    //gds
    bool enable_gds_log = false;
    bool enable_gds_get_from_sst = false;
    
    // RocksDB configuration
    std::string rocksdb_path = "/tmp/mlkv_plus_rocksdb";
    bool create_if_missing = true;
    bool disableWAL = false;
    bool force_skip_memtable = false;
    bool rocksdb_use_direct_reads = false;
};



} // namespace mlkv_plus 