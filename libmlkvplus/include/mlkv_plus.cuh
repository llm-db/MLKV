#pragma once

#include "storage_config.h"
#include <memory>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <atomic>

#include "gpu_tree.cuh"
#include "memdisk_tree.cuh"
#include "utils.cuh"

namespace mlkv_plus {

template<typename Key, typename Value, typename Score = uint64_t>
class DB {
public:
    // Constructor and destructor
    explicit DB(const StorageConfig& config = StorageConfig{});
    ~DB();
    
    // Initialize storage system
    OperationResult initialize(cudaStream_t stream = nullptr);
    
    // Clean up resources
    void cleanup();
    
    // Single operations
    OperationResult put(const Key& key, const Value* values);
    OperationResult get(const Key* d_key, Value* d_values);
    
    // Batch operations
    OperationResult multiset(const Key* d_keys, const Value* d_values, size_t batch_size);
    OperationResult multiget(const Key* d_keys, Value* d_values_out, bool* d_found, size_t batch_size);

    OperationResult multiget_gds(const Key* d_keys, Value* d_values_out, bool* d_found, size_t batch_size);

    OperationResult old_multiget(const Key* d_keys, Value* d_values_out, bool* h_found, size_t batch_size);
    
    // Delete operations
    OperationResult remove(const Key& key);
    
    // Check if key exists
    bool contains(const Key& key);
    
    
    // Get storage level for a key
    StorageLevel get_key_location(const Key& key) const;
    
    // Update configuration (some configurations may require reinitialization)
    OperationResult update_config(const StorageConfig& new_config);
    
    // Get current configuration
    const StorageConfig& get_config() const { return config_; }

private:
    // Configuration
    StorageConfig config_;
    
    // Storage instances
    std::unique_ptr<IGPUTree<Key, Value, Score>> gpu_tree_;
    std::unique_ptr<IMemDiskTree<Key, Value, Score>> memdisk_tree_;
    
    // CUDA related
    cudaStream_t cuda_stream_;
    bool owns_cuda_stream_;  // Track if we created the stream and should destroy it
    
    // Internal state
    bool initialized_;
    mutable std::mutex operation_mutex_;
    
    // Key location tracking
    mutable std::mutex location_mutex_;
    std::unordered_map<Key, StorageLevel> key_locations_;

    
    // Helper methods
    void update_key_location(const Key& key, StorageLevel level);
    StorageLevel find_key_location(const Key& key) const;
    

    // Serialization and deserialization
    std::string serialize_value_array(const Value* values);
    Value* deserialize_value_array(const std::string& raw);

};

} // namespace mlkv_plus 