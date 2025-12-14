#pragma once

#include "rocksdb/db.h"
#include "utils.cuh"
#include "format.cuh"


namespace mlkv_plus {

namespace gds {
    


class GDSDBAccessor {

public:


    

    static OperationResult GetFromSST(rocksdb::DB* db, const rocksdb::ReadOptions& read_options, 
                               const DeviceSlice& key, const rocksdb::Slice& host_key, DeviceSlice* value, 
                               gds::GDSDataBlockIterDevicePinnedBuffer* buffer, cudaStream_t stream = nullptr);



};
} // namespace gds
} // namespace mlkv_plus