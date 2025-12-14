#include <thrust/reduce.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <iostream>
#include <vector>
#include <cstdlib>

// CUDA runtime
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                                                               \
  do {                                                                                                                 \
    cudaError_t err = call;                                                                                            \
    if (err != cudaSuccess) {                                                                                          \
      fprintf(stderr, "CUDA Error: %s (code: %d) at %s:%d\n", cudaGetErrorString(err), err, __FILE__, __LINE__);        \
      exit(EXIT_FAILURE);                                                                                              \
    }                                                                                                                  \
  } while (0)

int main() {

    // Get number of CUDA devices
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        std::cerr << "No CUDA devices found." << std::endl;
        return 1;
    }
    std::cout << "Found " << device_count << " CUDA device(s)." << std::endl;

    // --- Part 1: A correct execution on device 0 ---
    std::cout << "\n--- Running on device 0 (Correct execution) ---" << std::endl;
    int device_to_use = 0;
    std::cout << "Setting device to: " << device_to_use << std::endl;
    CUDA_CHECK(cudaSetDevice(device_to_use));

    // Create data on the host
    std::vector<int> h_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    // Transfer data to the device
    thrust::device_vector<int> d_data = h_data;

    // Perform reduction on the device
    int sum = thrust::reduce(thrust::device, d_data.begin(), d_data.end(), 0, thrust::plus<int>());

    std::cout << "Sum calculated on device " << device_to_use << ": " << sum << std::endl;
    CUDA_CHECK(cudaDeviceSynchronize()); // Wait for all kernels to complete
    std::cout << "Correct execution finished successfully." << std::endl;


    // --- Part 2: Attempt to trigger the error ---
    // We will try to use an invalid device ID. Device IDs are 0-indexed.
    // So, 'device_count' is always an invalid ID.
    int invalid_device = -1;
    std::cout << "\n--- Trying to trigger error with invalid device " << invalid_device << " ---" << std::endl;
    
    cudaError_t err = cudaSetDevice(invalid_device);
    if (err != cudaSuccess) {
        std::cerr << "As expected, failed to set device to " << invalid_device << "." << std::endl;
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " (code: " << err << ")" << std::endl;
    } else {
        std::cerr << "Warning: Setting to an invalid device did not fail. This is unexpected." << std::endl;
        // If it somehow succeeds, the next CUDA operation will fail.
        thrust::device_vector<int> d_data_invalid = h_data;
        thrust::reduce(thrust::device, d_data_invalid.begin(), d_data_invalid.end());
        cudaDeviceSynchronize();
    }

    return 0;
}
