#include "mlkv_plus.cuh"
#include <iostream>
#include <sstream>
#include <algorithm>
#include <memory>
#include <cuda_runtime.h>
#include <chrono>
#include <thread>
#include <condition_variable>
#include <unordered_map>
#include "utils.cuh"

namespace mlkv_plus {


// CUDA kernel to extract found keys and values and write to buffer arrays
template<typename Key, typename Value>
__global__ void extract_found_keys_values_kernel(
    const Key* d_keys_input,           // Input keys array
    const Value* d_values_input,       // Input values array  
    const bool* d_found,               // Boolean array indicating which items were found
    Key* d_keys_found_output,          // Output keys array for found items
    Value* d_values_found_output,      // Output values array for found items
    size_t* d_found_count,             // Output count of found items
    Key* d_keys_unfound_output,        // Output keys array for unfound items
    size_t* d_unfound_count,           // Output count of unfound items
    size_t batch_size,                 // Total batch size
    size_t dim                         // Dimension of each value vector
)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < batch_size && d_found[idx]) {
        // Use atomic add to get unique output position
        size_t output_pos = atomicAdd(d_found_count, 1);
        
        // Copy the key
        d_keys_found_output[output_pos] = d_keys_input[idx];
        
        // Copy the value vector
        for (size_t d = 0; d < dim; ++d) {
            d_values_found_output[output_pos * dim + d] = d_values_input[idx * dim + d];
        }
    }
    else if (idx < batch_size && !d_found[idx] && d_keys_unfound_output != nullptr && d_unfound_count != nullptr) {
        size_t output_pos = atomicAdd(d_unfound_count, 1);
        d_keys_unfound_output[output_pos] = d_keys_input[idx];
    }
}

template<typename Key, typename Value, typename Score>
DB<Key, Value, Score>::DB(const StorageConfig& config)
    : config_(config)
    , gpu_tree_(nullptr)
    , memdisk_tree_(nullptr)
    , cuda_stream_(nullptr)
    , owns_cuda_stream_(false)
    , initialized_(false) {
    
}

template<typename Key, typename Value, typename Score>
DB<Key, Value, Score>::~DB() {
    try {
        cleanup();
    } catch (const std::exception& e) {
        std::cerr << "Exception during DB destruction: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "Unknown exception during DB destruction" << std::endl;
    }
}

template<typename Key, typename Value, typename Score>
OperationResult DB<Key, Value, Score>::initialize(cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(operation_mutex_);
    
    if (initialized_) {
        return OperationResult::SUCCESS;
    }

    
    MLKV_CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));
    MLKV_CUDA_CHECK(cudaSetDevice(config_.gpu_id));
    cudaDeviceProp p{};
    MLKV_CUDA_CHECK(cudaGetDeviceProperties(&p, config_.gpu_id));
    assert(p.canMapHostMemory);

    
    // Set the CUDA stream for DB operations
    if (stream != nullptr) {
        cuda_stream_ = stream;
        owns_cuda_stream_ = false;  // External stream, don't destroy it
    } else {
        // Create a new stream if none provided
        cudaError_t error = cudaStreamCreate(&cuda_stream_);
        if (error != cudaSuccess) {
            std::cerr << "Failed to create CUDA stream for DB" << std::endl;
            return OperationResult::CUDA_ERROR;
        }
        owns_cuda_stream_ = true;  // We created it, we should destroy it
    }
    
    // initialize memdisk tree
    memdisk_tree_ = std::make_unique<MemDiskTreeRocksDB<Key, Value, Score>>(config_);
    auto memdisk_tree_result = memdisk_tree_->initialize();
    if (memdisk_tree_result != OperationResult::SUCCESS) {
        return memdisk_tree_result;
    }
    
    // initialize gpu tree
    gpu_tree_ = std::make_unique<GPUTreeHkv<Key, Value, Score>>(config_);
    auto gpu_tree_result = gpu_tree_->initialize(cuda_stream_);
    if (gpu_tree_result != OperationResult::SUCCESS) {
        return gpu_tree_result;
    }
    
    initialized_ = true;


    // print the config
    std::cout << "DB config: " << std::endl;
    std::cout << "  enable_gds_log: " << config_.enable_gds_log << std::endl;
    std::cout << "  disableWAL: " << config_.disableWAL << std::endl;
    std::cout << "  rocksdb_path: " << config_.rocksdb_path << std::endl;
    std::cout << "  create_if_missing: " << config_.create_if_missing << std::endl;
    
    std::cout << "DB initialized successfully" << std::endl;
    return OperationResult::SUCCESS;
}

template<typename Key, typename Value, typename Score>
void DB<Key, Value, Score>::cleanup() {
    std::lock_guard<std::mutex> lock(operation_mutex_);
    
    if (!initialized_) {
        return;
    }
    
    // First synchronize any pending CUDA operations
    cudaError_t error = cudaDeviceSynchronize();
    if (error != cudaSuccess && error != cudaErrorCudartUnloading) {
        std::cerr << "Warning: CUDA synchronization failed during cleanup: " 
                  << cudaGetErrorString(error) << std::endl;
    }

    // Clean up GPU tree first (it may have CUDA resources)
    if (gpu_tree_) {
        gpu_tree_.reset();
    }
    
    // Clean up CPU/disk tree
    if (memdisk_tree_) {
        memdisk_tree_.reset();
    }
    
    // Clean up CUDA stream last, with error checking
    // Only destroy if we created it
    if (cuda_stream_ && owns_cuda_stream_) {
        error = cudaStreamDestroy(cuda_stream_);
        if (error != cudaSuccess && error != cudaErrorCudartUnloading) {
            std::cerr << "Warning: CUDA stream destruction failed: " 
                      << cudaGetErrorString(error) << std::endl;
        }
        cuda_stream_ = nullptr;
        owns_cuda_stream_ = false;
    }
    
    initialized_ = false;
    std::cout << "DB cleaned up" << std::endl;
}


template<typename Key, typename Value, typename Score>
OperationResult DB<Key, Value, Score>::put(const Key& key, const Value* values) {
    if (!initialized_) {
        throw std::runtime_error("DB not initialized");
    }
    
    std::lock_guard<std::mutex> lock(operation_mutex_);
    
    // First try writing to GPU tree
    EvictedData<Key, Value> evicted_data;
    OperationResult gpu_tree_result = gpu_tree_->put(key, values, evicted_data);
    
    // Handle evicted data if any
    if (gpu_tree_result == OperationResult::GPU_EVICTED && evicted_data.count > 0) {
        for (size_t i = 0; i < evicted_data.count; ++i) {
            memdisk_tree_->put(&evicted_data.keys[i], &evicted_data.values[i*config_.dim]);
        }
        MLKV_CUDA_CHECK(cudaFree(evicted_data.keys));
        MLKV_CUDA_CHECK(cudaFree(evicted_data.values));
    }
    
    return OperationResult::SUCCESS;
}

template<typename Key, typename Value, typename Score>
OperationResult DB<Key, Value, Score>::get(const Key* d_key, Value* d_values) {
    if (!initialized_) {
        throw std::runtime_error("DB not initialized");
    }
    
    // Search in order of storage hierarchy: HBM -> DRAM -> Disk
    

    OperationResult gpu_tree_result = gpu_tree_->get(d_key, d_values);
    if (gpu_tree_result == OperationResult::SUCCESS) {
        return OperationResult::SUCCESS;
    }
    
    OperationResult memdisk_tree_result = memdisk_tree_->get(d_key, d_values);
    if (memdisk_tree_result == OperationResult::SUCCESS) {
        return OperationResult::SUCCESS;
    }
    
    return OperationResult::KEY_NOT_FOUND;
}

template<typename Key, typename Value, typename Score>
OperationResult DB<Key, Value, Score>::multiset(const Key* d_keys, 
                                             const Value* d_values,
                                             size_t batch_size) {
    if (!initialized_) {
        throw std::runtime_error("DB not initialized");
    }

    if (batch_size == 0) {
        return OperationResult::SUCCESS;
    }
    
    std::lock_guard<std::mutex> lock(operation_mutex_);

    // First try batch writing to GPU tree
    EvictedData<Key, Value> evicted_data;
    OperationResult gpu_tree_result = gpu_tree_->multiset(d_keys, d_values, batch_size, evicted_data);
    
    if (gpu_tree_result == OperationResult::CUDA_ERROR) {
        return gpu_tree_result;
    }
    
    
    // Handle evicted data if any
    if (gpu_tree_result == OperationResult::GPU_EVICTED && evicted_data.count > 0) {
        // std::cout << "evicted_data.count: " << evicted_data.count << std::endl;
        memdisk_tree_->multiset(evicted_data.keys, evicted_data.values, evicted_data.count);
        MLKV_CUDA_CHECK(cudaFree(evicted_data.keys));
        MLKV_CUDA_CHECK(cudaFree(evicted_data.values));
    }
    
    return OperationResult::SUCCESS;
}


template<typename Key, typename Value, typename Score>
OperationResult DB<Key, Value, Score>::multiget_gds(const Key* d_keys, 
                                             Value* d_values_out,
                                             bool* d_found,
                                             size_t batch_size) {
    if (!initialized_ || batch_size == 0) {
        throw std::runtime_error("DB not initialized or invalid batch size");
    }
    
    // Try getting from GPU tree first
    OperationResult gpu_tree_result = gpu_tree_->multiget_gpu_only(d_keys, d_values_out, d_found, batch_size);
    
    if (gpu_tree_result == OperationResult::GPU_ALL_FOUND) {
        return OperationResult::SUCCESS;
    }

    // Extract found keys and values and write to buffer
    
    // Allocate device memory for output arrays
    Key* d_found_keys = nullptr;
    Value* d_found_values = nullptr; 
    size_t* d_found_count = nullptr;
    Key* d_unfound_keys = nullptr;
    size_t* d_unfound_count = nullptr;
    
    MLKV_CUDA_CHECK(cudaMalloc(&d_found_keys, batch_size * sizeof(Key)));
    MLKV_CUDA_CHECK(cudaMalloc(&d_found_values, batch_size * config_.dim * sizeof(Value)));
    MLKV_CUDA_CHECK(cudaMalloc(&d_found_count, sizeof(size_t)));
    MLKV_CUDA_CHECK(cudaMemset(d_found_count, 0, sizeof(size_t)));
    MLKV_CUDA_CHECK(cudaMalloc(&d_unfound_keys, batch_size * sizeof(Key)));
    MLKV_CUDA_CHECK(cudaMalloc(&d_unfound_count, sizeof(size_t)));
    MLKV_CUDA_CHECK(cudaMemset(d_unfound_count, 0, sizeof(size_t)));
    
    // Launch kernel to extract found keys and values
    int block_size = 256;
    int grid_size = (batch_size + block_size - 1) / block_size;
    
    extract_found_keys_values_kernel<Key, Value><<<grid_size, block_size, 0, cuda_stream_>>>(
        d_keys, d_values_out, d_found, 
        d_found_keys, d_found_values, d_found_count,
        d_unfound_keys, d_unfound_count,
        batch_size, config_.dim
    );
    
    // Synchronize and get the count of found items
    MLKV_CUDA_CHECK(cudaDeviceSynchronize());
    
    size_t h_found_count = 0;
    MLKV_CUDA_CHECK(cudaMemcpy(&h_found_count, d_found_count, sizeof(size_t), cudaMemcpyDeviceToHost));
    
    // Write found keys and values to buffer if there are any
    if (h_found_count > 0) {
        std::cout << "Writing " << h_found_count << " found items to buffer" << std::endl;
        gpu_tree_->multiset_buffer(d_found_keys, d_found_values, h_found_count);
    }
    
    // Clean up temporary device memory
    MLKV_CUDA_CHECK(cudaFree(d_found_keys));
    MLKV_CUDA_CHECK(cudaFree(d_found_values));
    MLKV_CUDA_CHECK(cudaFree(d_found_count));

    size_t h_unfound_count = 0;
    MLKV_CUDA_CHECK(cudaMemcpy(&h_unfound_count, d_unfound_count, sizeof(size_t), cudaMemcpyDeviceToHost));
    Value* d_unfound_values = nullptr;
    MLKV_CUDA_CHECK(cudaMalloc(&d_unfound_values, h_unfound_count * config_.dim * sizeof(Value)));

    bool* h_found_in_memdisk_tree = new bool[h_unfound_count];
   
    OperationResult memdisk_tree_result = memdisk_tree_->multiget_gds(d_unfound_keys, d_unfound_values, h_found_in_memdisk_tree, h_unfound_count);
    
    // store the unfound keys and values to GPU tree
    OperationResult result = multiset(d_unfound_keys, d_unfound_values, h_unfound_count);


    // store he unfound keys and values to buffer
    gpu_tree_->multiset_buffer(d_unfound_keys, d_unfound_values, h_unfound_count);
    
    // use buffer to get the final result
    OperationResult gpu_tree_result_final = gpu_tree_->multiget_buffer(d_keys, d_values_out, d_found, batch_size);
    if (gpu_tree_result_final == OperationResult::GPU_ALL_FOUND) {
        return OperationResult::SUCCESS;
    }


    // Clean up unfound keys and values memory
    MLKV_CUDA_CHECK(cudaFree(d_unfound_keys));
    MLKV_CUDA_CHECK(cudaFree(d_unfound_values));
    
    return OperationResult::SUCCESS;
}




template<typename Key, typename Value, typename Score>
OperationResult DB<Key, Value, Score>::multiget(const Key* d_keys, 
                                             Value* d_values_out,
                                             bool* d_found,
                                             size_t batch_size) {
    if (!initialized_ || batch_size == 0) {
        throw std::runtime_error("DB not initialized or invalid batch size");
    }

    if (config_.enable_gds_get_from_sst) {
        return multiget_gds(d_keys, d_values_out, d_found, batch_size);
    }
    
    // Try getting from GPU tree first
    OperationResult gpu_tree_result = gpu_tree_->multiget_gpu_only(d_keys, d_values_out, d_found, batch_size);
    
    if (gpu_tree_result == OperationResult::GPU_ALL_FOUND) {
        return OperationResult::SUCCESS;
    }



    // Extract found keys and values and write to buffer
    
    // Allocate device memory for output arrays
    Key* d_found_keys = nullptr;
    Value* d_found_values = nullptr; 
    size_t* d_found_count = nullptr;
    
    MLKV_CUDA_CHECK(cudaMalloc(&d_found_keys, batch_size * sizeof(Key)));
    MLKV_CUDA_CHECK(cudaMalloc(&d_found_values, batch_size * config_.dim * sizeof(Value)));
    MLKV_CUDA_CHECK(cudaMalloc(&d_found_count, sizeof(size_t)));
    MLKV_CUDA_CHECK(cudaMemset(d_found_count, 0, sizeof(size_t)));
    
    // Launch kernel to extract found keys and values
    int block_size = 256;
    int grid_size = (batch_size + block_size - 1) / block_size;
    
    extract_found_keys_values_kernel<Key, Value><<<grid_size, block_size, 0, cuda_stream_>>>(
        d_keys, d_values_out, d_found, 
        d_found_keys, d_found_values, d_found_count,
        nullptr, nullptr,
        batch_size, config_.dim
    );
    
    // Synchronize and get the count of found items
    MLKV_CUDA_CHECK(cudaDeviceSynchronize());
    
    size_t h_found_count = 0;
    MLKV_CUDA_CHECK(cudaMemcpy(&h_found_count, d_found_count, sizeof(size_t), cudaMemcpyDeviceToHost));
    
    // Write found keys and values to buffer if there are any
    if (h_found_count > 0) {
        std::cout << "Writing " << h_found_count << " found items to buffer" << std::endl;
        gpu_tree_->multiset_buffer(d_found_keys, d_found_values, h_found_count);
    }
    
    // Clean up temporary device memory
    MLKV_CUDA_CHECK(cudaFree(d_found_keys));
    MLKV_CUDA_CHECK(cudaFree(d_found_values));
    MLKV_CUDA_CHECK(cudaFree(d_found_count));


    // move keys and found status from gpu to cpu
    std::vector<Key> h_keys(batch_size);
    std::vector<char> h_found(batch_size);
    cudaMemcpy(h_keys.data(), d_keys, batch_size * sizeof(Key), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_found.data(), d_found, batch_size * sizeof(char), cudaMemcpyDeviceToHost);

    // Collect all unfound keys with deduplication
    std::unordered_map<Key, std::vector<size_t>> unfound_key_to_indices;
    for (size_t i = 0; i < batch_size; ++i) {
        if (h_found[i] == 0) {
            unfound_key_to_indices[h_keys[i]].push_back(i);
        }
    }
    
    std::vector<Key> unfound_keys;
    std::vector<char> unfound_found_in_memdisk_tree;
    unfound_keys.reserve(unfound_key_to_indices.size());
    unfound_found_in_memdisk_tree.reserve(unfound_key_to_indices.size());
    
    for (const auto& pair : unfound_key_to_indices) {
        unfound_keys.push_back(pair.first);
        unfound_found_in_memdisk_tree.push_back(0);
    }

    // Batch get from memdisk tree
    std::vector<Value> unfound_values;
    unfound_values.reserve(unfound_keys.size()*config_.dim);

    // print the number of unfound keys
    std::cout << "number of unfound keys in gpu tree: " << unfound_keys.size() << std::endl;

    OperationResult memdisk_tree_result = memdisk_tree_->multiget(unfound_keys, unfound_values, unfound_found_in_memdisk_tree);


    // send keys and values to gpu mem
    Key* d_unfound_keys = nullptr;
    Value* d_unfound_values = nullptr;
    MLKV_CUDA_CHECK(cudaMalloc(&d_unfound_keys, unfound_keys.size() * sizeof(Key)));
    MLKV_CUDA_CHECK(cudaMalloc(&d_unfound_values, unfound_keys.size() * config_.dim * sizeof(Value)));
    MLKV_CUDA_CHECK(cudaMemcpy(d_unfound_keys, unfound_keys.data(), unfound_keys.size() * sizeof(Key), cudaMemcpyHostToDevice));
    MLKV_CUDA_CHECK(cudaMemcpy(d_unfound_values, unfound_values.data(), unfound_keys.size() * config_.dim * sizeof(Value), cudaMemcpyHostToDevice));

    // store the unfound keys and values to GPU tree
    OperationResult result = multiset(d_unfound_keys, d_unfound_values, unfound_keys.size());


    // store he unfound keys and values to buffer
    gpu_tree_->multiset_buffer(d_unfound_keys, d_unfound_values, unfound_keys.size());
    
    // use buffer to get the final result
    OperationResult gpu_tree_result_final = gpu_tree_->multiget_buffer(d_keys, d_values_out, d_found, batch_size);
    if (gpu_tree_result_final == OperationResult::GPU_ALL_FOUND) {
        return OperationResult::SUCCESS;
    }


    // Clean up unfound keys and values memory
    MLKV_CUDA_CHECK(cudaFree(d_unfound_keys));
    MLKV_CUDA_CHECK(cudaFree(d_unfound_values));
    
    return OperationResult::SUCCESS;
}


template<typename Key, typename Value, typename Score>
std::string DB<Key, Value, Score>::serialize_value_array(const Value* values) {
    std::string buffer;
    size_t size = config_.dim;
    buffer.append(reinterpret_cast<const char*>(&size), sizeof(size));
    buffer.append(reinterpret_cast<const char*>(values), sizeof(Value) * size);
    return buffer;
}

template<typename Key, typename Value, typename Score>
Value* DB<Key, Value, Score>::deserialize_value_array(const std::string& raw) {
    size_t size;
    memcpy(&size, raw.data(), sizeof(size));
    Value* result = new Value[size];
    memcpy(result, raw.data() + sizeof(size), sizeof(Value) * size);
    return result;
}




template<typename Key, typename Value, typename Score>
void DB<Key, Value, Score>::update_key_location(const Key& key, StorageLevel level) {
    std::lock_guard<std::mutex> lock(location_mutex_);
    key_locations_[key] = level;
}

template<typename Key, typename Value, typename Score>
StorageLevel DB<Key, Value, Score>::find_key_location(const Key& key) const {
    std::lock_guard<std::mutex> lock(location_mutex_);
    auto it = key_locations_.find(key);
    return (it != key_locations_.end()) ? it->second : StorageLevel::DISK;
}


template<typename Key, typename Value, typename Score>
StorageLevel DB<Key, Value, Score>::get_key_location(const Key& key) const {
    return find_key_location(key);
}

template<typename Key, typename Value, typename Score>
OperationResult DB<Key, Value, Score>::update_config(const StorageConfig& new_config) {
    config_ = new_config;
    return OperationResult::SUCCESS;
}

// --------------------------------------------------------------
// --------------------- old implementation ---------------------
// --------------------------------------------------------------

template<typename Key, typename Value, typename Score>
OperationResult DB<Key, Value, Score>::old_multiget(const Key* d_keys, 
                                             Value* d_values_out,
                                             bool* h_found,
                                             size_t batch_size) {
    if (!initialized_ || batch_size == 0) {
        throw std::runtime_error("DB not initialized or invalid batch size");
    }
    
    // Try getting from GPU tree first
    OperationResult gpu_tree_result = gpu_tree_->multiget(d_keys, d_values_out, h_found, batch_size);
    
    if (gpu_tree_result == OperationResult::GPU_ALL_FOUND) {
        return OperationResult::SUCCESS;
    }


    // move keys from gpu to cpu
    std::vector<Key> h_keys(batch_size);
    cudaMemcpy(h_keys.data(), d_keys, batch_size * sizeof(d_keys), cudaMemcpyDeviceToHost);

    // Collect all unfound keys with deduplication
    std::unordered_map<Key, std::vector<size_t>> unfound_key_to_indices;
    for (size_t i = 0; i < batch_size; ++i) {
        if (!h_found[i]) {
            unfound_key_to_indices[h_keys[i]].push_back(i);
        }
    }
    
    std::vector<Key> unfound_keys;
    std::vector<char> unfound_found_in_memdisk_tree;
    unfound_keys.reserve(unfound_key_to_indices.size());
    unfound_found_in_memdisk_tree.reserve(unfound_key_to_indices.size());
    
    for (const auto& pair : unfound_key_to_indices) {
        unfound_keys.push_back(pair.first);
        unfound_found_in_memdisk_tree.push_back(0);
    }

    // Batch get from memdisk tree
    std::vector<Value> unfound_values;
    unfound_values.reserve(unfound_keys.size()*config_.dim);

    // print the number of unfound keys
    std::cout << "number of unfound keys in gpu tree: " << unfound_keys.size() << std::endl;

    OperationResult memdisk_tree_result = memdisk_tree_->multiget(unfound_keys, unfound_values, unfound_found_in_memdisk_tree);


    // send keys and values to gpu mem
    Key* d_unfound_keys = nullptr;
    Value* d_unfound_values = nullptr;
    MLKV_CUDA_CHECK(cudaMalloc(&d_unfound_keys, unfound_keys.size() * sizeof(Key)));
    MLKV_CUDA_CHECK(cudaMalloc(&d_unfound_values, unfound_keys.size() * config_.dim * sizeof(Value)));
    MLKV_CUDA_CHECK(cudaMemcpy(d_unfound_keys, unfound_keys.data(), unfound_keys.size() * sizeof(Key), cudaMemcpyHostToDevice));
    MLKV_CUDA_CHECK(cudaMemcpy(d_unfound_values, unfound_values.data(), unfound_keys.size() * config_.dim * sizeof(Value), cudaMemcpyHostToDevice));

    // store the unfound keys and values to GPU tree
    OperationResult result = multiset(d_unfound_keys, d_unfound_values, unfound_keys.size());


    // final round of getting from gpu tree
    OperationResult gpu_tree_result_final = gpu_tree_->multiget(d_keys, d_values_out, h_found, batch_size);
    

    if (gpu_tree_result_final == OperationResult::GPU_ALL_FOUND) {
        return OperationResult::SUCCESS;
    }

    // For remaining unfound keys, batch get from memdisk tree with multi-stream optimization
    std::vector<Key> remaining_unfound_keys;
    std::vector<size_t> remaining_key_indices;
    
    // Collect all remaining unfound keys and their indices
    for (size_t i = 0; i < batch_size; ++i) {
        if (!h_found[i]) {
            remaining_unfound_keys.push_back(h_keys[i]);
            remaining_key_indices.push_back(i);
        }
    }
    
    if (!remaining_unfound_keys.empty()) {
        // Batch get from memdisk tree
        std::vector<Value> batch_values(remaining_unfound_keys.size() * config_.dim);
        std::vector<char> batch_found(remaining_unfound_keys.size(), 0);
        
        OperationResult batch_result = memdisk_tree_->multiget(remaining_unfound_keys, batch_values, batch_found);
        
        if (batch_result == OperationResult::SUCCESS) {
            // Create multiple CUDA streams for parallel memory copy
            const int num_streams = std::min(5, static_cast<int>(remaining_unfound_keys.size()));
            std::vector<cudaStream_t> streams(num_streams);
            
            // Create streams
            for (int i = 0; i < num_streams; ++i) {
                cudaStreamCreate(&streams[i]);
            }
            
            // Parallel copy using multiple streams
            for (size_t i = 0; i < remaining_unfound_keys.size(); ++i) {
                if (batch_found[i]) {
                    size_t original_idx = remaining_key_indices[i];
                    int stream_idx = i % num_streams;
                    
                    // Asynchronous copy to GPU memory
                    cudaMemcpyAsync(&d_values_out[original_idx * config_.dim], 
                                   &batch_values[i * config_.dim],
                                   sizeof(Value) * config_.dim, 
                                   cudaMemcpyHostToDevice, 
                                   streams[stream_idx]);
                    
                    h_found[original_idx] = true;
                }
            }
            
            // Synchronize all streams
            for (int i = 0; i < num_streams; ++i) {
                cudaStreamSynchronize(streams[i]);
                cudaStreamDestroy(streams[i]);
            }
        }
    }
    
    return OperationResult::SUCCESS;
}

// Explicit template instantiation
template class DB<uint64_t, double, uint64_t>;
template class DB<int64_t, double, uint64_t>;
template class DB<int64_t, float, uint64_t>;

} // namespace mlkv_plus 