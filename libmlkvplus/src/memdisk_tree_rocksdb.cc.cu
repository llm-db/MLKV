#include <driver_types.h>
#include <iostream>
#include <string>

#include "memdisk_tree.cuh"
#include "storage_config.h"

#include "gds/gds_block_based_table_factory.cuh"
#include "gds/gds_db_accessor.cuh"
#include "gds/format.cuh"
#include "utils.cuh"
#include <omp.h>


namespace mlkv_plus {

using namespace gds;

template<typename Key, typename Value, typename Score>
MemDiskTreeRocksDB<Key, Value, Score>::MemDiskTreeRocksDB(const StorageConfig& config) : config_(config) {}


template<typename Key, typename Value, typename Score>
OperationResult MemDiskTreeRocksDB<Key, Value, Score>::initialize() {
    rocksdb::Options options;
    
    options.create_if_missing = config_.create_if_missing;
    options.use_direct_reads = false;

    
    write_options_.write_by_gds = config_.enable_gds_log;
    write_options_.disableWAL = config_.disableWAL;
    

    rocksdb::BlockBasedTableOptions table_options;
    table_options.persistent_cache = nullptr;
    table_options.data_block_index_type = rocksdb::BlockBasedTableOptions::kDataBlockBinarySearch;
    
    options.compression = kNoCompression;
    options.comparator = rocksdb::BytewiseComparator();


    if (config_.enable_gds_get_from_sst) {
        omp_set_dynamic(0);
        omp_set_num_threads(omp_get_num_procs()-5);
        gds::GDSOptions gds_options;
        std::shared_ptr<rocksdb::TableFactory> gds_factory(
            new GDSBlockBasedTableFactory(table_options, gds_options));
        options.table_factory = gds_factory;
        gds_buffers_.resize(omp_get_num_procs()-5);
        std::cout << "gds_buffers_.size(): " << gds_buffers_.size() << std::endl;
        for (size_t i = 0; i < gds_buffers_.size(); ++i) {
            gds_buffers_[i] = gds::GDSDataBlockIterDevicePinnedBufferManager::Allocate();
        }

        gds_streams_.resize(omp_get_num_procs()-5);
        for (size_t i = 0; i < gds_streams_.size(); ++i) {
            cudaStreamCreate(&gds_streams_[i]);
        }



        


    } else {
        options.table_factory.reset(new rocksdb::BlockBasedTableFactory(table_options));
    }
    
    rocksdb::Status status = rocksdb::DB::Open(options, config_.rocksdb_path, &rocksdb_);
    if (!status.ok()) {
        std::cerr << "Failed to open RocksDB: " << status.ToString() << std::endl;
        return OperationResult::ROCKSDB_ERROR;
    }


    if (config_.enable_gds_log) {
        uint64_t latest_sequence_number = rocksdb_->GetLatestSequenceNumber();
        std::cout << "Latest sequence number: " << latest_sequence_number << std::endl;

        std::unique_ptr<rocksdb::WalFile> wal_files;
        rocksdb::Status status = rocksdb_->GetCurrentWalFile(&wal_files);
        if (!status.ok()) {
            std::cerr << "Failed to get current wal file: " << status.ToString() << std::endl;
            throw std::runtime_error("Failed to get current wal file");
        }
        std::string current_wal_relative_path = wal_files->PathName();
        std::cout << "current_wal_relative_path: " << current_wal_relative_path << std::endl;
        gwal_writer_ = std::make_unique<GWALWriter>(current_wal_relative_path, config_.rocksdb_path);
        write_batch_generator_ = std::make_unique<GPUWriteBatchGenerator>(config_.max_batch_size);

    }


    
    return OperationResult::SUCCESS;
}


template<typename Key, typename Value, typename Score>
OperationResult MemDiskTreeRocksDB<Key, Value, Score>::put(const Key* d_key, const Value* d_values) {
    if (!rocksdb_) {
        return OperationResult::ROCKSDB_ERROR;
    }
    
    // Convert key and value to RocksDB Slice
    Key* key = new Key();
    MLKV_CUDA_CHECK(cudaMemcpy(key, d_key, sizeof(Key), cudaMemcpyDeviceToHost));
    Value* values = new Value[config_.dim];
    MLKV_CUDA_CHECK(cudaMemcpy(values, d_values, sizeof(Value) * config_.dim, cudaMemcpyDeviceToHost));
    
    std::string key_str;
    key_str.append(reinterpret_cast<const char*>(&key), sizeof(Key));
    rocksdb::Slice key_slice(reinterpret_cast<const char*>(&key), sizeof(Key));
    std::string value_str = serialize_value_array(values);
    rocksdb::Slice value_slice(value_str.data(), value_str.size());
    
    rocksdb::Status status = rocksdb_->Put(write_options_, key_slice, value_slice);


    delete key;
    delete[] values;

    if (config_.enable_gds_log) {
        IOStatus s = IOStatus::OK();
        if (!s.ok()) {
            std::cerr << "Failed to add key-value pair to batch generator: " << s.message() << std::endl;
            throw std::runtime_error("Failed to add key-value pair to batch generator");
        }
        std::unique_ptr<rocksdb::WalFile> wal_files;
        rocksdb::Status status = rocksdb_->GetCurrentWalFile(&wal_files);
        if (!status.ok()) {
            std::cerr << "Failed to get current wal file: " << status.ToString() << std::endl;
            throw std::runtime_error("Failed to get current wal file");
        }
        std::string current_wal_relative_path = wal_files->PathName();
        if (!gwal_writer_->IsWALRelativePathEqual(current_wal_relative_path)) {
            gwal_writer_ = std::make_unique<GWALWriter>(current_wal_relative_path, config_.rocksdb_path);
        }
        uint64_t latest_sequence_number = rocksdb_->GetLatestSequenceNumber();
        s = write_batch_generator_->GenerateTypedWriteBatch<Key, Value>(d_key, d_values, 1, config_.dim, latest_sequence_number);

        // Write the generated batch to WAL using GDS
        const char* batch_data = write_batch_generator_->GetWriteBatchData();
        size_t batch_size_bytes = write_batch_generator_->GetWriteBatchSize();
        
        if (batch_data != nullptr && batch_size_bytes > 0) {
            s = gwal_writer_->AddRecord(batch_data, batch_size_bytes);
        } else {
            s = IOStatus::IOError("Generated WriteBatch is empty");
        }
        
        if (!s.ok()) {
            std::cerr << "Failed to write generated batch to WAL: " << s.message() << std::endl;
            throw std::runtime_error("Failed to write generated batch to WAL");
        }

        write_batch_generator_->Clear();
    }
    
    if (!status.ok()) {
        std::cerr << "Failed to put to RocksDB: " << status.ToString() << std::endl;
        return OperationResult::ROCKSDB_ERROR;
    }
    
    return OperationResult::SUCCESS;
}



template<typename Key, typename Value, typename Score>
OperationResult MemDiskTreeRocksDB<Key, Value, Score>::get(const Key* d_key, Value* d_values) {
    if (!rocksdb_) {
        return OperationResult::ROCKSDB_ERROR;
    }


    if (config_.enable_gds_get_from_sst) {

        Key* h_key = new Key();
        MLKV_CUDA_CHECK(cudaMemcpy(h_key, d_key, sizeof(Key), cudaMemcpyDeviceToHost));
        rocksdb::Slice host_key_slice(reinterpret_cast<const char*>(h_key), sizeof(Key));

        // first check if the key is in the memtable
        if (!config_.force_skip_memtable) {
            rocksdb::ReadOptions read_options_memtable;
            read_options_memtable.read_tier = rocksdb::kMemtableTier;
            std::string value_str;
            rocksdb::Status status_rocksdb = rocksdb_->Get(read_options_memtable, host_key_slice, &value_str);

            if (status_rocksdb.ok()) {
                Value* temp_values = deserialize_value_array(value_str);
                MLKV_CUDA_CHECK(cudaMemcpy(d_values, temp_values, sizeof(Value) * config_.dim, cudaMemcpyHostToDevice));
                delete[] temp_values;
                delete h_key;
                return OperationResult::SUCCESS;
            }
        }
        gds::DeviceSlice key_slice(reinterpret_cast<const char*>(d_key), sizeof(Key));
        gds::DeviceSlice value_slice;
        OperationResult status = GDSDBAccessor::GetFromSST(rocksdb_.get(), rocksdb::ReadOptions(), key_slice, host_key_slice, &value_slice, gds_buffers_[omp_get_thread_num()]);
        if (status != OperationResult::SUCCESS) {
            delete h_key;
            return status;
        }
        
        // Deserialize value array directly to output buffer on GPU using parallel kernel
        if (!device_deserialize_value_array(value_slice, d_values)) {
            std::cerr << "Failed to deserialize value array on GPU" << std::endl;
            delete h_key;
            return OperationResult::ROCKSDB_ERROR;
        }
        delete h_key;
        return OperationResult::SUCCESS;
    } else {
        rocksdb::ReadOptions read_options;
        read_options.force_skip_memtable = config_.force_skip_memtable;

        Key* key = new Key();
        MLKV_CUDA_CHECK(cudaMemcpy(key, d_key, sizeof(Key), cudaMemcpyDeviceToHost));

        rocksdb::Slice key_slice(reinterpret_cast<const char*>(key), sizeof(Key));
        std::string value_str;  
    
        rocksdb::Status status = rocksdb_->Get(read_options, key_slice, &value_str);
        
        if (status.IsNotFound()) {
            return OperationResult::KEY_NOT_FOUND;
        } else if (!status.ok()) {
            std::cerr << "Failed to get from RocksDB: " << status.ToString() << std::endl;
            return OperationResult::ROCKSDB_ERROR;
        }
        
        Value* temp_values = deserialize_value_array(value_str);
        MLKV_CUDA_CHECK(cudaMemcpy(d_values, temp_values, sizeof(Value) * config_.dim, cudaMemcpyHostToDevice));

        delete key;
        delete[] temp_values;

            
        return OperationResult::SUCCESS;
    }
}


template<typename Key, typename Value, typename Score>
OperationResult MemDiskTreeRocksDB<Key, Value, Score>::multiget_gds(const Key* d_keys, 
                                                                    Value* d_values_out,
                                                                    bool* h_found_out,
                                                                    size_t batch_size) {
    if (!rocksdb_) {
        return OperationResult::ROCKSDB_ERROR;
    }
    std::vector<Key> h_keys(batch_size);

    MLKV_CUDA_CHECK(cudaMemcpy(h_keys.data(), d_keys, sizeof(Key) * batch_size, cudaMemcpyDeviceToHost));

    std::vector<rocksdb::Slice> key_slices;
    key_slices.reserve(batch_size);
    for (const auto& key : h_keys) {
        key_slices.emplace_back(reinterpret_cast<const char*>(&key), sizeof(Key));
    }

    std::vector<std::pair<rocksdb::Slice, size_t>> need_to_get_from_sst_key_slices;

    // first check if the key is in the memtable
    if (!config_.force_skip_memtable) {
        rocksdb::ReadOptions read_options_memtable;
        read_options_memtable.read_tier = rocksdb::kMemtableTier;
        std::vector<std::string> value_strings;
        std::vector<rocksdb::Status>  status_rocksdb = rocksdb_->MultiGet(read_options_memtable, key_slices, &value_strings);

        bool all_found = true;
        for (size_t i = 0; i < batch_size; ++i) {
            if (status_rocksdb[i].ok()) {
                // Convert string value back to Value type
                Value* temp_values = deserialize_value_array(value_strings[i]);
                // Copy to GPU directly
                MLKV_CUDA_CHECK(cudaMemcpy(&d_values_out[i * config_.dim], temp_values, config_.dim * sizeof(Value), cudaMemcpyHostToDevice));
                delete[] temp_values;  // Clean up temporary memory
                h_found_out[i] = 1;
            } else {
                all_found = false;
                h_found_out[i] = 0;
                need_to_get_from_sst_key_slices.push_back(std::make_pair(key_slices[i], i));
            }
        }

        if (all_found) {
            return OperationResult::SUCCESS;
        }
    }

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < need_to_get_from_sst_key_slices.size(); ++i) {
        const auto& pair = need_to_get_from_sst_key_slices[i];
        gds::DeviceSlice key_slice(reinterpret_cast<const char*>(d_keys[pair.second]), sizeof(Key));
        gds::DeviceSlice value_slice;
        OperationResult status = GDSDBAccessor::GetFromSST(rocksdb_.get(), rocksdb::ReadOptions(), key_slice, pair.first, &value_slice, gds_buffers_[omp_get_thread_num()], gds_streams_[omp_get_thread_num()]);
        if (status == OperationResult::KEY_NOT_FOUND) {
            h_found_out[pair.second] = 0;
        } else if (status == OperationResult::SUCCESS) {
            h_found_out[pair.second] = 1;
            if (!device_deserialize_value_array(value_slice, &d_values_out[pair.second * config_.dim], gds_streams_[omp_get_thread_num()])) {
                std::cerr << "Failed to deserialize value array on GPU" << std::endl;
                return OperationResult::ROCKSDB_ERROR;
            }
        } else {
            return status;
        }
    }

    return OperationResult::SUCCESS;
}


template<typename Key, typename Value, typename Score>
OperationResult MemDiskTreeRocksDB<Key, Value, Score>::multiget(const std::vector<Key>& keys, 
                                                    std::vector<Value>& values,
                                                    std::vector<char>& found) {
    if (!rocksdb_) {
        return OperationResult::ROCKSDB_ERROR;
    }
    
    try {
        const size_t n = keys.size();
        
        // Convert keys to RocksDB Slices
        std::vector<rocksdb::Slice> key_slices;
        key_slices.reserve(n);
        for (const auto& key : keys) {
            key_slices.emplace_back(reinterpret_cast<const char*>(&key), sizeof(Key));
        }
        
        // Prepare vector for values
        std::vector<std::string> value_strings;
        
        // Perform batch get
        std::vector<rocksdb::Status> statuses = rocksdb_->MultiGet(
            rocksdb::ReadOptions(),
            key_slices,
            &value_strings
        );

        
        // Process results
        for (size_t i = 0; i < n; ++i) {
            if (statuses[i].ok()) {
                // Convert string value back to Value type
                Value* temp_values = deserialize_value_array(value_strings[i]);
                memcpy(&values[i*config_.dim], temp_values, config_.dim * sizeof(Value));
                delete[] temp_values;  // Clean up temporary memory
                found[i] = 1;
            } else {
                found[i] = 0;
            }
        }
        
        return OperationResult::SUCCESS;
    } catch (const std::exception& e) {
        std::cerr << "RocksDB MultiGet error: " << e.what() << std::endl;
        return OperationResult::ROCKSDB_ERROR;
    }
}


template<typename Key, typename Value, typename Score>
OperationResult MemDiskTreeRocksDB<Key, Value, Score>::multiset(const Key* d_keys, const Value* d_values, size_t batch_size) {
    if (!rocksdb_ || batch_size == 0) {
        throw std::runtime_error("RocksDB multiset error");
    }


    // For RocksDB write, we still need host memory (RocksDB doesn't support GPU memory directly)
    Key* keys = new Key[batch_size];
    Value* values = new Value[batch_size * config_.dim];
    MLKV_CUDA_CHECK(cudaMemcpy(keys, d_keys, batch_size * sizeof(Key), cudaMemcpyDeviceToHost));
    MLKV_CUDA_CHECK(cudaMemcpy(values, d_values, batch_size * config_.dim * sizeof(Value), cudaMemcpyDeviceToHost));
    
    rocksdb::WriteBatch batch;
    
    for (size_t i = 0; i < batch_size; ++i) {
        std::string key_str;
        key_str.append(reinterpret_cast<const char*>(&keys[i]), sizeof(Key));
        rocksdb::Slice key_slice(key_str.data(), key_str.size());
        std::string value_str = serialize_value_array(&values[i * config_.dim]);
        rocksdb::Slice value_slice(value_str.data(), value_str.size());
        
        batch.Put(key_slice, value_slice);
    }

    uint64_t latest_sequence_number = rocksdb_->GetLatestSequenceNumber();
    // std::cout << "latest sequence number: " << latest_sequence_number << std::endl;


    if (config_.enable_gds_log) {
        // get the latest wal path name
        std::unique_ptr<rocksdb::WalFile> wal_files;
        rocksdb::Status status = rocksdb_->GetCurrentWalFile(&wal_files);
        if (!status.ok()) {
            std::cerr << "Failed to get current wal file: " << status.ToString() << std::endl;
            throw std::runtime_error("Failed to get current wal file");
        }
        std::string current_wal_relative_path = wal_files->PathName();
        if (!gwal_writer_->IsWALRelativePathEqual(current_wal_relative_path)) {
            gwal_writer_ = std::make_unique<GWALWriter>(current_wal_relative_path, config_.rocksdb_path);
            std::cout << "new gwal_writer: " << current_wal_relative_path << std::endl;
        }
        
        // Use template-based GPU write batch generation directly from device memory
        IOStatus s = write_batch_generator_->GenerateTypedWriteBatch<Key, Value>(
            d_keys, d_values, batch_size, config_.dim, latest_sequence_number+1);
        
        if (!s.ok()) {
            std::cerr << "Failed to generate typed write batch: " << s.message() << std::endl;
            throw std::runtime_error("Failed to generate typed write batch");
        }
        
        // Write the generated batch to WAL using GDS
        const char* batch_data = write_batch_generator_->GetWriteBatchData();
        size_t batch_size_bytes = write_batch_generator_->GetWriteBatchSize();
        
        if (batch_data != nullptr && batch_size_bytes > 0) {
            s = gwal_writer_->AddRecord(batch_data, batch_size_bytes);
        } else {
            s = IOStatus::IOError("Generated WriteBatch is empty");
        }

        write_batch_generator_->Clear();
        
        // std::cout << "latest sequence number: " << latest_sequence_number << std::endl;
        if (!s.ok()) {
            std::cerr << "Failed to write generated batch to WAL: " << s.message() << std::endl;
            throw std::runtime_error("Failed to write generated batch to WAL");
        }
    }
    
    rocksdb::Status status = rocksdb_->Write(write_options_, &batch);

    // Clean up host memory
    delete[] keys;
    delete[] values;


    if (!status.ok()) {
        std::cerr << "Failed to write batch to RocksDB: " << status.ToString() << std::endl;
        throw std::runtime_error("Failed to write batch to RocksDB");
    }
    
    return OperationResult::SUCCESS;

}


template<typename Key, typename Value, typename Score>
std::string MemDiskTreeRocksDB<Key, Value, Score>::serialize_value_array(const Value* values) {
    std::string buffer;
    size_t size = config_.dim;
    buffer.append(reinterpret_cast<const char*>(&size), sizeof(size));
    buffer.append(reinterpret_cast<const char*>(values), sizeof(Value) * size);
    return buffer;
}

template<typename Key, typename Value, typename Score>
Value* MemDiskTreeRocksDB<Key, Value, Score>::deserialize_value_array(const std::string& raw) {
    size_t size;
    memcpy(&size, raw.data(), sizeof(size));
    Value* result = new Value[size];
    memcpy(result, raw.data() + sizeof(size), sizeof(Value) * size);
    return result;
}

// CUDA kernel for parallel deserialization on GPU
// Each thread copies one element from the serialized data using memcpy to avoid alignment issues
template<typename Value>
__global__ void kernel_deserialize_value_array(
    const char* __restrict__ d_raw_data,
    size_t offset,           // Offset to skip the size field
    Value* __restrict__ d_output,
    size_t num_elements
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_elements) {
        // Use memcpy to avoid alignment issues
        // Copy from unaligned source to aligned destination
        const char* src = d_raw_data + offset + idx * sizeof(Value);
        memcpy(&d_output[idx], src, sizeof(Value));
    }
}

// Optimized version using shared memory for better coalescing (for small arrays)
template<typename Value>
__global__ void kernel_deserialize_value_array_shared(
    const char* __restrict__ d_raw_data,
    size_t offset,
    Value* __restrict__ d_output,
    size_t num_elements
) {
    extern __shared__ char shared_mem[];
    Value* s_buffer = reinterpret_cast<Value*>(shared_mem);
    
    size_t tid = threadIdx.x;
    size_t global_idx = blockIdx.x * blockDim.x + tid;
    
    // Cooperative loading into shared memory using memcpy to avoid alignment issues
    if (global_idx < num_elements) {
        const char* src = d_raw_data + offset + global_idx * sizeof(Value);
        memcpy(&s_buffer[tid], src, sizeof(Value));
    }
    __syncthreads();
    
    // Write from shared memory to global output
    if (global_idx < num_elements) {
        d_output[global_idx] = s_buffer[tid];
    }
}

// Vectorized version for better memory throughput (when Value type allows)
template<typename Value>
__global__ void kernel_deserialize_value_array_vectorized(
    const char* __restrict__ d_raw_data,
    size_t offset,
    Value* __restrict__ d_output,
    size_t num_elements
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // For float/double, use vectorized copy with memcpy to avoid alignment issues
    constexpr size_t vec_size = sizeof(Value) >= 8 ? 2 : 4;
    size_t vec_idx = idx * vec_size;
    
    if (vec_idx + vec_size <= num_elements) {
        // Vectorized copy using memcpy
        #pragma unroll
        for (size_t i = 0; i < vec_size; ++i) {
            const char* src = d_raw_data + offset + (vec_idx + i) * sizeof(Value);
            memcpy(&d_output[vec_idx + i], src, sizeof(Value));
        }
    } else if (vec_idx < num_elements) {
        // Handle remaining elements
        for (size_t i = 0; vec_idx + i < num_elements; ++i) {
            const char* src = d_raw_data + offset + (vec_idx + i) * sizeof(Value);
            memcpy(&d_output[vec_idx + i], src, sizeof(Value));
        }
    }
}

template<typename Key, typename Value, typename Score>
bool MemDiskTreeRocksDB<Key, Value, Score>::device_deserialize_value_array(const gds::DeviceSlice& raw, Value* d_output, cudaStream_t stream) {
    if (raw.empty() || raw.data() == nullptr) {
        return false;
    }
    
    if (d_output == nullptr) {
        std::cerr << "Output buffer is null" << std::endl;
        return false;
    }
    
    // Step 1: Extract size from the first sizeof(size_t) bytes on GPU
    size_t h_size;
    MLKV_CUDA_CHECK(cudaMemcpyAsync(&h_size, raw.data(), sizeof(size_t), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    if (h_size == 0) {
        std::cerr << "Got zero size when deserializing value array of RocksDB" << std::endl;
        return false;
    }
    
    // Validate size
    size_t expected_raw_size = sizeof(size_t) + sizeof(Value) * h_size;
    if (raw.size() < expected_raw_size) {
        std::cerr << "Invalid raw data size. Expected at least " << expected_raw_size 
                  << " but got " << raw.size() << std::endl;
        return false;
    }
    
    // Step 2: Launch parallel deserialization kernel (write directly to caller's buffer)
    const size_t offset = sizeof(size_t);  // Skip the size field
    
    // Choose kernel based on array size and value type
    constexpr size_t THREADS_PER_BLOCK = 256;
    constexpr size_t SHARED_MEM_THRESHOLD = 4096;  // Use shared memory for small arrays
    
    if (h_size <= SHARED_MEM_THRESHOLD) {
        // Small array: use shared memory for better coalescing
        size_t num_blocks = (h_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        size_t shared_mem_size = THREADS_PER_BLOCK * sizeof(Value);
        
        kernel_deserialize_value_array_shared<Value><<<num_blocks, THREADS_PER_BLOCK, shared_mem_size, stream>>>(
            raw.data(), offset, d_output, h_size
        );
    } else if (sizeof(Value) == sizeof(float) || sizeof(Value) == sizeof(double)) {
        // Large array with float/double: use vectorized loads
        constexpr size_t vec_size = sizeof(Value) >= 8 ? 2 : 4;
        size_t num_vec_elements = (h_size + vec_size - 1) / vec_size;
        size_t num_blocks = (num_vec_elements + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        
        kernel_deserialize_value_array_vectorized<Value><<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
            raw.data(), offset, d_output, h_size
        );
    } else {
        // Default: simple parallel copy
        size_t num_blocks = (h_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        
        kernel_deserialize_value_array<Value><<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
            raw.data(), offset, d_output, h_size
        );
    }
    
    // Step 3: Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    cudaStreamSynchronize(stream);
    
    return true;
}



// Explicit template instantiation
template class MemDiskTreeRocksDB<uint64_t, double, uint64_t>;
template class MemDiskTreeRocksDB<int64_t, double, uint64_t>;
template class MemDiskTreeRocksDB<int64_t, float, uint64_t>;

}

