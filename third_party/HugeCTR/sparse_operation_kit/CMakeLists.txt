cmake_minimum_required(VERSION 3.20)

project(test_thrust_reduce LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

add_executable(test_thrust_reduce test_thrust_reduce.cu)

set_target_properties(test_thrust_reduce PROPERTIES CUDA_ARCHITECTURES OFF POSITION_INDEPENDENT_CODE ON)

message(STATUS "Configured to build test_thrust_reduce")
