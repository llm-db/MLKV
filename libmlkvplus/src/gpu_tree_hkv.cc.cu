#include <cuda_runtime_api.h>
#include <iostream>
#include <exception>
#include "gpu_tree.cuh"
#include "storage_config.h"
#include <cuda_runtime.h>
#include <memory>

namespace mlkv_plus
{

    // CUDA kernel to check if all boolean values in an array are true
    __global__ void check_all_true_kernel(const bool *d_array, size_t size, int *d_result)
    {
        // Initialize result to true (only first thread)
        if (blockIdx.x == 0 && threadIdx.x == 0)
        {
            *d_result = 1; // 1 represents true
        }
        __syncthreads();

        // Each thread checks one element
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size)
        {
            if (!d_array[idx])
            {
                // If any element is false, set result to false atomically
                atomicExch(d_result, 0); // 0 represents false
            }
        }
    }

    template <typename Key, typename Value, typename Score>
    GPUTreeHkv<Key, Value, Score>::GPUTreeHkv(const StorageConfig &config) : config_(config) {}

    template <typename Key, typename Value, typename Score>
    OperationResult GPUTreeHkv<Key, Value, Score>::initialize(cudaStream_t stream)
    {
        cuda_stream_ = stream;
        nv::merlin::HashTableOptions options;
        options.init_capacity = config_.hkv_init_capacity;
        options.max_capacity = config_.hkv_max_capacity;
        options.dim = config_.dim;
        options.max_hbm_for_vectors = nv::merlin::GB(config_.max_hbm_for_vectors_gb);
        options.io_by_cpu = config_.hkv_io_by_cpu;
        options.device_id = config_.gpu_id;
        hkv_table_ = std::make_unique<HKVTable>();
        hkv_table_->init(options);
        MLKV_CUDA_CHECK(cudaGetLastError());
        // print the config
        std::cout << "hkv_table_ config: " << std::endl;
        std::cout << "  init_capacity: " << options.init_capacity << std::endl;
        std::cout << "  max_capacity: " << options.max_capacity << std::endl;
        std::cout << "  dim: " << options.dim << std::endl;
        std::cout << "  hbm_gb: " << config_.max_hbm_for_vectors_gb << std::endl;
        std::cout << "  io_by_cpu: " << options.io_by_cpu << std::endl;
        std::cout << "  device_id: " << options.device_id << std::endl;
        std::cout << "  max_batch_size: " << config_.max_batch_size << std::endl;


        init_device_buffers(config_.max_batch_size, 4);

        // initialize gpu tree buffer
        nv::merlin::HashTableOptions buffer_options;
        // Then `num_of_buckets_per_alloc` must be equal or less than initial required buckets number.
        // for more details refer to merlin_hashtable.cuh#L950
        buffer_options.init_capacity = config_.max_batch_size >= options.num_of_buckets_per_alloc * options.max_bucket_size ? config_.max_batch_size : options.num_of_buckets_per_alloc * options.max_bucket_size;
        buffer_options.max_capacity = config_.max_batch_size >= options.num_of_buckets_per_alloc * options.max_bucket_size ? config_.max_batch_size : options.num_of_buckets_per_alloc * options.max_bucket_size;
        buffer_options.dim = config_.dim;
        buffer_options.io_by_cpu = config_.hkv_io_by_cpu;
        buffer_options.device_id = config_.gpu_id;
        hkv_table_buffer_ = std::make_unique<HKVTable>();
        hkv_table_buffer_->init(buffer_options);
            

        return OperationResult::SUCCESS;
    }

    template <typename Key, typename Value, typename Score>
    OperationResult GPUTreeHkv<Key, Value, Score>::put(const Key &key, const Value *values, EvictedData<Key, Value> &evicted_data)
    {
        if (!hkv_table_)
        {
            return OperationResult::CUDA_ERROR;
        }

        // Allocate memory for input data
        Key *d_keys;
        Value *d_values;
        CUDA_CHECK(cudaMalloc(&d_keys, sizeof(Key)));
        CUDA_CHECK(cudaMalloc(&d_values, sizeof(Value) * config_.dim));

        // Allocate memory for evicted data
        Key *d_evicted_keys;
        Value *d_evicted_values;
        size_t *d_evicted_counter;
        CUDA_CHECK(cudaMalloc(&d_evicted_keys, sizeof(Key)));
        CUDA_CHECK(cudaMalloc(&d_evicted_values, sizeof(Value) * config_.dim));
        CUDA_CHECK(cudaMalloc(&d_evicted_counter, sizeof(size_t)));

        // Initialize evicted counter to 0
        CUDA_CHECK(cudaMemset(d_evicted_counter, 0, sizeof(size_t)));

        // Copy input data to device
        CUDA_CHECK(cudaMemcpy(d_keys, &key, sizeof(Key), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_values, values, sizeof(Value) * config_.dim, cudaMemcpyHostToDevice));

        // Perform insert and evict
        hkv_table_->insert_and_evict(1, d_keys, d_values, nullptr,
                                     d_evicted_keys, d_evicted_values, nullptr,
                                     d_evicted_counter, cuda_stream_);

        // Synchronize and check if any evictions occurred
        cudaStreamSynchronize(cuda_stream_);
        size_t evicted_count;
        CUDA_CHECK(cudaMemcpy(&evicted_count, d_evicted_counter, sizeof(size_t), cudaMemcpyDeviceToHost));

        // If evictions occurred, copy evicted data back to host
        if (evicted_count > 0)
        {
            evicted_data.count = evicted_count;
            evicted_data.keys = d_evicted_keys;
            evicted_data.values = d_evicted_values;

            // CUDA_CHECK(cudaMemcpy(evicted_data.keys, d_evicted_keys,
            //                       evicted_count * sizeof(Key), cudaMemcpyDeviceToHost));
            // CUDA_CHECK(cudaMemcpy(evicted_data.values, d_evicted_values,
            //                       evicted_count * sizeof(Value) * config_.dim, cudaMemcpyDeviceToHost));
        }

        // Clean up allocated memory
        CUDA_CHECK(cudaFree(d_keys));
        CUDA_CHECK(cudaFree(d_values));
        CUDA_CHECK(cudaFree(d_evicted_counter));

        return (evicted_count > 0) ? OperationResult::GPU_EVICTED : OperationResult::SUCCESS;
    }

    template <typename Key, typename Value, typename Score>
    OperationResult GPUTreeHkv<Key, Value, Score>::get(const Key *d_key, Value *d_values)
    {
        if (!hkv_table_)
        {
            return OperationResult::CUDA_ERROR;
        }

        try
        {
            bool *d_found;
            bool found;
            CUDA_CHECK(cudaMalloc(&d_found, sizeof(bool)));
            hkv_table_->find(1, d_key, d_values, d_found, nullptr, cuda_stream_);
            cudaStreamSynchronize(cuda_stream_);

            // copy d_found to host
            CUDA_CHECK(cudaMemcpy(&found, d_found, sizeof(bool), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaFree(d_found));
            if (found) { return OperationResult::SUCCESS;}
            else{ return OperationResult::KEY_NOT_FOUND; }
        }
        catch (const std::exception &e)
        {
            return OperationResult::CUDA_ERROR;
        }
    }

    template <typename Key, typename Value, typename Score>
    OperationResult GPUTreeHkv<Key, Value, Score>::multiset(const Key *d_keys,
                                                            const Value *d_values,
                                                            size_t batch_size,
                                                            EvictedData<Key, Value> &evicted_data,
                                                            cudaStream_t stream)
    {
        if (!hkv_table_)
        {
            return OperationResult::CUDA_ERROR;
        }

        if (batch_size == 0)
        {
            return OperationResult::SUCCESS;
        }

        try {
            
            // Allocate memory for evicted data
            Key *d_evicted_keys;
            Value *d_evicted_values;
            size_t *d_evicted_counter;
            cudaMalloc(&d_evicted_keys, batch_size * sizeof(Key));
            cudaMalloc(&d_evicted_values, batch_size * sizeof(Value) * config_.dim);
            cudaMalloc(&d_evicted_counter, sizeof(size_t));

            // Initialize evicted counter to 0
            cudaMemset(d_evicted_counter, 0, sizeof(size_t));


            // Use provided stream or default stream
            cudaStream_t active_stream = (stream != nullptr) ? stream : cuda_stream_;

            // Perform batch insert and evict
            hkv_table_->insert_and_evict(batch_size, d_keys, d_values, nullptr,
                                        d_evicted_keys, d_evicted_values, nullptr,
                                        d_evicted_counter, active_stream);


            // Synchronize and check if any evictions occurred
            CUDA_CHECK(cudaDeviceSynchronize());
            size_t evicted_count;
            CUDA_CHECK(cudaMemcpy(&evicted_count, d_evicted_counter, sizeof(size_t), cudaMemcpyDeviceToHost));

            // std::cout << "evicted_count: " << evicted_count << std::endl;
            // If evictions occurred, copy evicted data back to host
            if (evicted_count > 0)
            {
                evicted_data.count = evicted_count;
                evicted_data.keys = d_evicted_keys;
                evicted_data.values = d_evicted_values;

                

                // CUDA_CHECK(cudaMemcpy(evicted_data.keys, d_evicted_keys,
                //                     evicted_count * sizeof(Key), cudaMemcpyDeviceToHost));
                // CUDA_CHECK(cudaMemcpy(evicted_data.values, d_evicted_values,
                //                     evicted_count * sizeof(Value) * config_.dim, cudaMemcpyDeviceToHost));
            }

            // Clean up allocated memory
            // CUDA_CHECK(cudaFree(d_evicted_keys));
            // CUDA_CHECK(cudaFree(d_evicted_values));
            CUDA_CHECK(cudaFree(d_evicted_counter));

            return (evicted_count > 0) ? OperationResult::GPU_EVICTED : OperationResult::SUCCESS;

        } catch (const std::exception& e) {
            std::cerr << "Multiset failed: " << e.what() << std::endl;
            return OperationResult::CUDA_ERROR;
        }
    }

    template <typename Key, typename Value, typename Score>
    OperationResult GPUTreeHkv<Key, Value, Score>::multiget(const Key *d_keys,
                                                            Value *d_values_out,
                                                            bool *h_found,
                                                            size_t batch_size)
    {

        // try
        // {
        // Allocate device memory
        DeviceBuffer *device_buffer = get_device_buffer();
        // Key *d_keys = device_buffer->d_keys;
        // d_values_out = device_buffer->d_values_out;
        bool *d_found = device_buffer->d_found;
        int *d_all_true = device_buffer->d_all_true;
        int all_true_int;

        // // Copy keys to device
        // CUDA_CHECK(cudaMemcpy(d_keys, keys, batch_size * sizeof(Key), cudaMemcpyHostToDevice));

        hkv_table_->find(batch_size, d_keys, d_values_out, d_found, nullptr, cuda_stream_);

        CUDA_CHECK(cudaStreamSynchronize(cuda_stream_));

        // Launch kernel to check if all values are true
        int block_size = 256;
        int grid_size = (batch_size + block_size - 1) / block_size;
        check_all_true_kernel<<<grid_size, block_size, 0, cuda_stream_>>>(d_found, batch_size, d_all_true);

        CUDA_CHECK(cudaStreamSynchronize(cuda_stream_));

        // Copy result back to host
        CUDA_CHECK(cudaMemcpy(&all_true_int, d_all_true, sizeof(int), cudaMemcpyDeviceToHost));

        if (all_true_int == 1)
        {
            // Clean up device memory
            release_device_buffer(device_buffer);
            // set all h_found to true (in host)
            memset(h_found, true, batch_size * sizeof(bool));

            return OperationResult::GPU_ALL_FOUND;
        }

        CUDA_CHECK(cudaMemcpy(h_found, d_found, batch_size * sizeof(bool), cudaMemcpyDeviceToHost));

        // Clean up device memory
        release_device_buffer(device_buffer);

        return OperationResult::SUCCESS;
        // }
        // catch (const std::exception &e)
        // {
        //     std::cerr << "Multiget failed: " << e.what() << std::endl;
        //     return OperationResult::CUDA_ERROR;
        // }
    }

    template <typename Key, typename Value, typename Score>
    OperationResult GPUTreeHkv<Key, Value, Score>::multiget_gpu_only(const Key *d_keys,
                                                            Value *d_values_out,
                                                            bool *d_found,
                                                            size_t batch_size)
    {

        DeviceBuffer *device_buffer = get_device_buffer();
        int *d_all_true = device_buffer->d_all_true;
        int all_true_int;

        hkv_table_->find(batch_size, d_keys, d_values_out, d_found, nullptr, cuda_stream_);
        CUDA_CHECK(cudaStreamSynchronize(cuda_stream_));


        // Launch kernel to check if all values are true
        int block_size = 256;
        int grid_size = (batch_size + block_size - 1) / block_size;
        check_all_true_kernel<<<grid_size, block_size, 0, cuda_stream_>>>(d_found, batch_size, d_all_true);

        CUDA_CHECK(cudaStreamSynchronize(cuda_stream_));

        // Copy result back to host
        CUDA_CHECK(cudaMemcpy(&all_true_int, d_all_true, sizeof(int), cudaMemcpyDeviceToHost));

        if (all_true_int == 1)
        {
            // Clean up device memory
            release_device_buffer(device_buffer);

            return OperationResult::GPU_ALL_FOUND;
        }


        release_device_buffer(device_buffer);

        return OperationResult::SUCCESS;
    }

    // ------------------------------------
    // HKV buffer related functions
    // ------------------------------------


    template <typename Key, typename Value, typename Score>
    OperationResult GPUTreeHkv<Key, Value, Score>::multiset_buffer(const Key *d_keys,
                                                        const Value *d_values,
                                                        size_t batch_size)
    {
        
        hkv_table_buffer_->insert_or_assign(batch_size, d_keys, d_values, nullptr, cuda_stream_, true, false);
        CUDA_CHECK(cudaStreamSynchronize(cuda_stream_));
        return OperationResult::SUCCESS;
    }


    template <typename Key, typename Value, typename Score>
    OperationResult GPUTreeHkv<Key, Value, Score>::multiget_buffer(const Key *d_keys,
                                                        Value *d_values_out,
                                                        bool *d_found,
                                                        size_t batch_size)
    {
        hkv_table_buffer_->find(batch_size, d_keys, d_values_out, d_found, nullptr, cuda_stream_);
        CUDA_CHECK(cudaStreamSynchronize(cuda_stream_));
        return OperationResult::SUCCESS;
    }



    // ------------------------------------
    // device buffer related functions
    // ------------------------------------

    template <typename Key, typename Value, typename Score>
    void GPUTreeHkv<Key, Value, Score>::init_device_buffers(uint64_t max_batch_size, uint32_t num_buffers)
    {
        // Ensure we have at least 2 buffers for concurrent operations
        uint32_t actual_num_buffers = std::max(num_buffers, 2u);

        std::lock_guard<std::mutex> lock(buffer_mutex_);

        // Make sure we're on the correct GPU
        cudaSetDevice(config_.gpu_id);

        device_buffers_pool_.resize(actual_num_buffers);

        for (uint32_t i = 0; i < actual_num_buffers; ++i)
        {
            DeviceBuffer &device_buffer = device_buffers_pool_[i];

            device_buffer.buffer_id = i;
            device_buffer.in_use = false;

            // Allocate device memory with error checking
            cudaError_t err = cudaMalloc(&device_buffer.d_keys, max_batch_size * sizeof(Key));
            if (err != cudaSuccess)
            {
                std::cerr << "Failed to allocate d_keys for buffer " << i << ": " << cudaGetErrorString(err) << std::endl;
                continue;
            }

            err = cudaMalloc(&device_buffer.d_values, max_batch_size * sizeof(Value) * config_.dim);
            if (err != cudaSuccess)
            {
                std::cerr << "Failed to allocate d_values for buffer " << i << ": " << cudaGetErrorString(err) << std::endl;
                if (device_buffer.d_keys)
                    cudaFree(device_buffer.d_keys);
                continue;
            }

            err = cudaMalloc(&device_buffer.d_values_out, max_batch_size * sizeof(Value) * config_.dim);
            if (err != cudaSuccess)
            {
                std::cerr << "Failed to allocate d_values_out for buffer " << i << ": " << cudaGetErrorString(err) << std::endl;
                if (device_buffer.d_keys)
                    cudaFree(device_buffer.d_keys);
                if (device_buffer.d_values)
                    cudaFree(device_buffer.d_values);
                continue;
            }

            err = cudaMalloc(&device_buffer.d_found, max_batch_size * sizeof(bool));
            if (err != cudaSuccess)
            {
                std::cerr << "Failed to allocate d_found for buffer " << i << ": " << cudaGetErrorString(err) << std::endl;
                if (device_buffer.d_keys)
                    cudaFree(device_buffer.d_keys);
                if (device_buffer.d_values)
                    cudaFree(device_buffer.d_values);
                if (device_buffer.d_values_out)
                    cudaFree(device_buffer.d_values_out);
                continue;
            }

            err = cudaMalloc(&device_buffer.d_all_true, sizeof(int));
            if (err != cudaSuccess)
            {
                std::cerr << "Failed to allocate d_all_true for buffer " << i << ": " << cudaGetErrorString(err) << std::endl;
                if (device_buffer.d_keys)
                    cudaFree(device_buffer.d_keys);
                if (device_buffer.d_values)
                    cudaFree(device_buffer.d_values);
                if (device_buffer.d_values_out)
                    cudaFree(device_buffer.d_values_out);
                if (device_buffer.d_found)
                    cudaFree(device_buffer.d_found);
                continue;
            }
        }

        // Count successfully initialized buffers
        size_t successful_buffers = 0;
        for (const auto &buffer : device_buffers_pool_)
        {
            if (buffer.d_keys && buffer.d_values && buffer.d_values_out && buffer.d_found && buffer.d_all_true)
            {
                successful_buffers++;
            }
        }

        std::cout << "Initialized " << successful_buffers << " device buffers on GPU " << config_.gpu_id << std::endl;
    }

    template <typename Key, typename Value, typename Score>
    void GPUTreeHkv<Key, Value, Score>::free_device_buffers()
    {
        std::lock_guard<std::mutex> lock(buffer_mutex_);

        cudaSetDevice(config_.gpu_id);

        for (auto &device_buffer : device_buffers_pool_)
        {
            if (device_buffer.d_keys)
                cudaFree(device_buffer.d_keys);
            if (device_buffer.d_values)
                cudaFree(device_buffer.d_values);
            if (device_buffer.d_values_out)
                cudaFree(device_buffer.d_values_out);
            if (device_buffer.d_found)
                cudaFree(device_buffer.d_found);
            if (device_buffer.d_all_true)
                cudaFree(device_buffer.d_all_true);
        }

        device_buffers_pool_.clear();
    }

    template <typename Key, typename Value, typename Score>
    typename GPUTreeHkv<Key, Value, Score>::DeviceBuffer *
    GPUTreeHkv<Key, Value, Score>::get_device_buffer()
    {
        std::lock_guard<std::mutex> lock(buffer_mutex_);

        // Find an available buffer
        for (auto &buffer : device_buffers_pool_)
        {
            if (!buffer.in_use && buffer.d_keys && buffer.d_values && buffer.d_values_out && buffer.d_found && buffer.d_all_true)
            {
                buffer.in_use = true;
                return &buffer;
            }
        }

        std::cerr << "No available device buffers on GPU " << config_.gpu_id << std::endl;
        return nullptr;
    }

    template <typename Key, typename Value, typename Score>
    void GPUTreeHkv<Key, Value, Score>::release_device_buffer(DeviceBuffer *device_buffer)
    {
        if (!device_buffer)
        {
            return;
        }

        std::lock_guard<std::mutex> lock(buffer_mutex_);
        device_buffer->in_use = false;
    }

    template <typename Key, typename Value, typename Score>
    GPUTreeHkv<Key, Value, Score>::~GPUTreeHkv()
    {
        free_device_buffers();
    }

    // Explicit template instantiation
    template class GPUTreeHkv<uint64_t, double, uint64_t>;
    template class GPUTreeHkv<int64_t, double, uint64_t>;
    template class GPUTreeHkv<int64_t, float, uint64_t>;

}