#pragma once


#include "storage_config.h"
#include "rocksdb/db.h"
#include "gds/gwal_writer.cuh"
#include "gds/write_batch_generator.cuh"
#include "gds/format.cuh"
#include "utils.cuh"


namespace mlkv_plus {

template<typename Key, typename Value, typename Score = uint64_t>
class IMemDiskTree {

    public:
        virtual ~IMemDiskTree() = default;

        virtual OperationResult initialize() = 0;

        virtual OperationResult put(const Key* d_key, const Value* d_values) = 0;
        virtual OperationResult get(const Key* d_key, Value* d_values) = 0;

        virtual OperationResult multiget(const std::vector<Key>& keys, 
                                    std::vector<Value>& values,
                                    std::vector<char>& found) = 0;
        virtual OperationResult multiget_gds(const Key* d_keys, 
                                    Value* d_values_out,
                                    bool* h_found_out,
                                    size_t batch_size) = 0;
        
        virtual OperationResult multiset(const Key* d_keys, const Value* d_values, size_t batch_size) = 0;

};


template<typename Key, typename Value, typename Score>
class MemDiskTreeRocksDB : public IMemDiskTree<Key, Value, Score> {
    public:
        MemDiskTreeRocksDB(const StorageConfig& config);
        ~MemDiskTreeRocksDB() {
            if (config_.enable_gds_get_from_sst) {
                for (auto& gds_buffer : gds_buffers_) {
                    gds::GDSDataBlockIterDevicePinnedBufferManager::Free(gds_buffer);
                }
                
                for (auto& gds_stream : gds_streams_) {
                    cudaStreamDestroy(gds_stream);
                }
            }
        };

        OperationResult initialize() override;
        OperationResult put(const Key* d_key, const Value* d_values) override;
        OperationResult get(const Key* d_key, Value* d_values) override;
        OperationResult multiget(const std::vector<Key>& keys, 
                                    std::vector<Value>& values,
                                    std::vector<char>& found) override;
        OperationResult multiget_gds(const Key* d_keys, 
                                    Value* d_values_out,
                                    bool* h_found_out,
                                    size_t batch_size) override;

        
        OperationResult multiset(const Key* d_keys, const Value* d_values, size_t batch_size) override;

    private:
        StorageConfig config_;
        std::unique_ptr<rocksdb::DB> rocksdb_;
        rocksdb::WriteOptions write_options_;
        std::unique_ptr<gds::GWALWriter> gwal_writer_;
        std::unique_ptr<gds::GPUWriteBatchGenerator> write_batch_generator_;
        std::vector<gds::GDSDataBlockIterDevicePinnedBuffer*> gds_buffers_;
        std::vector<cudaStream_t> gds_streams_;

        std::string serialize_value_array(const Value* values);
        Value* deserialize_value_array(const std::string& raw);
        bool device_deserialize_value_array(const gds::DeviceSlice& raw, Value* d_output, cudaStream_t stream = nullptr);
        
};


}