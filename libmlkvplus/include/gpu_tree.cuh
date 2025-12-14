#pragma once

#include <iostream>
#include <queue>
#include <mutex>
#include "storage_config.h"
#include "merlin_hashtable.cuh"
#include "utils.cuh"


namespace mlkv_plus {

template<typename Key, typename Value, typename Score = uint64_t>
class IGPUTree {
    public:

        virtual ~IGPUTree() = default;
        
        virtual OperationResult initialize(cudaStream_t stream) = 0;

        virtual OperationResult put(const Key& key, const Value* values, EvictedData<Key, Value>& evicted_data) = 0;
        virtual OperationResult get(const Key* d_key, Value* d_values) = 0;


        virtual OperationResult multiset(const Key* d_keys, 
                                const Value* d_values,
                                size_t batch_size,
                                EvictedData<Key, Value>& evicted_data,
                                cudaStream_t stream = nullptr) = 0;
        virtual OperationResult multiget(const Key* d_keys, 
                                Value* d_values_out,
                                bool* h_found,
                                size_t batch_size) = 0;
        virtual OperationResult multiget_gpu_only(const Key* d_keys, 
                                Value* d_values_out,
                                bool* d_found,
                                size_t batch_size) = 0;                                

        virtual OperationResult multiset_buffer(const Key* d_keys,
                                    const Value* d_values,
                                    size_t batch_size) = 0;
        virtual OperationResult multiget_buffer(const Key* d_keys,
                                    Value* d_values_out,
                                    bool* d_found,
                                    size_t batch_size) = 0;
};





template<typename Key, typename Value, typename Score>
class GPUTreeHkv : public IGPUTree<Key, Value, Score> {

    using HKVTable = nv::merlin::HashTable<Key, Value, Score, nv::merlin::EvictStrategy::kLru>;
    using TableOptions = nv::merlin::HashTableOptions;

    public:
        GPUTreeHkv(const StorageConfig& config);
        ~GPUTreeHkv();

        OperationResult initialize(cudaStream_t stream) override;

        OperationResult put(const Key& key, const Value* values, EvictedData<Key, Value>& evicted_data) override;
        OperationResult get(const Key* d_key, Value* d_values) override;
        OperationResult multiset(const Key* d_keys, 
                                const Value* d_values,
                                size_t batch_size,
                                EvictedData<Key, Value>& evicted_data,
                                cudaStream_t stream = nullptr) override;

        OperationResult multiget(const Key* keys, 
                                Value* d_values_out,
                                bool* h_found,
                                size_t batch_size) override;
        OperationResult multiget_gpu_only(const Key* keys, 
                                Value* d_values_out,
                                bool* d_found,
                                size_t batch_size) override;

        OperationResult multiset_buffer(const Key* d_keys,
                                        const Value* d_values,
                                        size_t batch_size) override;

        OperationResult multiget_buffer(const Key* d_keys,
                                        Value* d_values_out,
                                        bool* d_found,
                                        size_t batch_size) override;

    private:
        StorageConfig config_;
        std::unique_ptr<HKVTable> hkv_table_;
        std::unique_ptr<HKVTable> hkv_table_buffer_;

        cudaStream_t cuda_stream_;


        struct DeviceBuffer {
            int buffer_id;
            Key* d_keys;
            Value* d_values;
            Value* d_values_out;
            bool* d_found;
            int* d_all_true;
            bool in_use;  // Track if buffer is in use
        };

        std::vector<DeviceBuffer> device_buffers_pool_;
        std::mutex buffer_mutex_;  // For thread-safe buffer management

        void init_device_buffers(uint64_t max_batch_size, uint32_t num_buffers);
        void free_device_buffers();


        DeviceBuffer* get_device_buffer();
        void release_device_buffer(DeviceBuffer* device_buffer);


};






};