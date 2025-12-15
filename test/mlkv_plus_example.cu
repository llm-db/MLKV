/**
 * MLKV Plus - Simple Insert & Read Example
 */

#include <iostream>
#include <cuda_runtime.h>
#include "mlkv_plus.cuh"
#include "utils.cuh"

using namespace mlkv_plus;

int main() {
    // Configure
    StorageConfig config;
    config.dim = 4;
    config.hkv_init_capacity = 1024;
    config.hkv_max_capacity = 4096;
    config.rocksdb_path = "/tmp/mlkv_example";
    config.enable_gds_log = false;
    config.enable_gds_get_from_sst = false;
    config.disableWAL = false;
    config.force_skip_memtable = false;
    config.rocksdb_use_direct_reads = false;
    
    // Initialize DB
    DB<int64_t, float, uint64_t> db(config);
    db.initialize();
    
    // Prepare data: 1 key with 4 floats
    const size_t n = 1;
    int64_t h_key = 42;
    float h_values[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    
    // Allocate GPU memory
    int64_t* d_keys;
    float* d_values;
    float* d_values_out;
    bool* d_found;
    
    cudaMalloc(&d_keys, sizeof(int64_t));
    cudaMalloc(&d_values, 4 * sizeof(float));
    cudaMalloc(&d_values_out, 4 * sizeof(float));
    cudaMalloc(&d_found, sizeof(bool));
    
    // Copy to GPU
    cudaMemcpy(d_keys, &h_key, sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, h_values, 4 * sizeof(float), cudaMemcpyHostToDevice);
    
    // Insert
    db.multiset(d_keys, d_values, n);
    std::cout << "Inserted key " << h_key << " -> [1, 2, 3, 4]\n";
    
    // Read
    cudaMemset(d_found, 0, sizeof(bool));
    db.multiget(d_keys, d_values_out, d_found, n);
    
    // Copy result back
    float h_result[4];
    bool h_found;
    cudaMemcpy(h_result, d_values_out, 4 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_found, d_found, sizeof(bool), cudaMemcpyDeviceToHost);
    
    std::cout << "Read key " << h_key << " -> [" 
              << h_result[0] << ", " << h_result[1] << ", " 
              << h_result[2] << ", " << h_result[3] << "] "
              << (h_found ? "(found)" : "(not found)") << "\n";
    
    // Cleanup
    cudaFree(d_keys);
    cudaFree(d_values);
    cudaFree(d_values_out);
    cudaFree(d_found);
    
    return 0;
}
