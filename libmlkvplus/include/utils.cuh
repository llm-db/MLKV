#pragma once

#include <cstdint>
#include <string>
#include <sstream>
#include <cuda_runtime.h>

#define MLKV_CUDA_CHECK(call)                                                     \
  do {                                                                            \
    cudaError_t _err = (call);                                                    \
    if (_err != cudaSuccess) {                                                    \
      std::ostringstream _oss;                                                    \
      _oss << "CUDA error (" << static_cast<int>(_err) << " - "                   \
           << cudaGetErrorName(_err) << "): " << cudaGetErrorString(_err)         \
           << " | call: " << #call                                                \
           << " | at " << __FILE__ << ":" << __LINE__;                            \
      throw std::runtime_error(_oss.str());                                       \
    }                                                                             \
  } while (0)





inline bool is_gpu_pointer(const void* ptr) {
  if (ptr == nullptr) {
    return false;
  }
  
  cudaPointerAttributes attributes;
  cudaError_t err = cudaPointerGetAttributes(&attributes, ptr);
  
  if (err != cudaSuccess) {
    // If we can't get attributes, assume it's a CPU pointer
    cudaGetLastError(); // Clear the error
    return false;
  }
  
  // Check if the pointer is on device memory
  return (attributes.type == cudaMemoryTypeDevice);
}

inline bool is_cpu_pointer(const void* ptr) {
  return !is_gpu_pointer(ptr);
}


namespace mlkv_plus {


// Operation result enumeration
enum class OperationResult {
  SUCCESS = 0,
  KEY_NOT_FOUND = 1,
  STORAGE_FULL = 2,
  GPU_ALL_FOUND = 3,
  GPU_EVICTED = 4,
  CUDA_ERROR = 5,
  ROCKSDB_ERROR = 6,
  INVALID_PARAMETER = 7
};

namespace gds {

struct GDSOptions {
    uint64_t DefaultGPUBufferSize = 64 * 1024 * 1024;
    uint64_t GDSAlignment = 4096;
    bool EnableGPUCache = true;  
    size_t GPUCacheBlockSize = 8 * 4096;  
    size_t GPUCacheMaxBlocks = 500000;   
    bool EnableCachePrefetch = true;       // Read full block for small requests
    size_t MinReadSizeForCache = 0;        // 0 = cache all reads
};


class IOStatus {
    public:
        enum IOStatusCode {
            SUCCESS = 0,
            FILE_NOT_FOUND = 1,
            FILE_ALREADY_EXISTS = 2,
            INVALID_PARAMETER = 3,
            CUDA_ERROR = 4,
            IO_ERROR = 5,
            USAGE_ERROR = 6,
            UNSUPPORTED_ERROR = 7,

        };

        static IOStatus OK(std::string msg="") { return IOStatus(IOStatusCode::SUCCESS, msg); }
        static IOStatus NotFound(std::string msg) { return IOStatus(IOStatusCode::FILE_NOT_FOUND, msg); }
        static IOStatus AlreadyExists(std::string msg) { return IOStatus(IOStatusCode::FILE_ALREADY_EXISTS, msg); }
        static IOStatus InvalidParam(std::string msg) { return IOStatus(IOStatusCode::INVALID_PARAMETER, msg); }
        static IOStatus CudaError(std::string msg) { return IOStatus(IOStatusCode::CUDA_ERROR, msg); }
        static IOStatus IOError(std::string msg) { return IOStatus(IOStatusCode::IO_ERROR, msg); }
        static IOStatus UsageError(std::string msg) { return IOStatus(IOStatusCode::USAGE_ERROR, msg); }
        static IOStatus UnsupportedError(std::string msg) { return IOStatus(IOStatusCode::UNSUPPORTED_ERROR, msg); }
    
        bool ok() const { return code_ == IOStatusCode::SUCCESS; }
        IOStatusCode code() const { return code_; }
        const std::string& message() const { return message_; }
    
    private:
        IOStatus(IOStatusCode code, std::string msg)
            : code_(code), message_(std::move(msg)) {}
    
        IOStatusCode code_;
        std::string message_;
    };

} // namespace gds



  
} // namespace mlkv_plus

// end of file