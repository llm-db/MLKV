cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
cmake_policy(SET CMP0077 NEW)
project(mlkv_plus LANGUAGES CUDA CXX)
# ------------------------------------------------------------
# Export compile commands for clangd
set(CMAKE_CUDA_USE_RESPONSE_FILE OFF CACHE BOOL "" FORCE)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF CACHE BOOL "" FORCE)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS  OFF CACHE BOOL "" FORCE)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# ------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

if(DEFINED ENV{CONDA_PREFIX})
    list(PREPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
endif()



find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)
find_package(nlohmann_json 3.12.0 REQUIRED)

foreach(cuda_arch ${sm})
  list(APPEND cuda_arch_list ${cuda_arch})
  message(STATUS "Assign GPU architecture (sm=${cuda_arch})")
endforeach()

list(LENGTH cuda_arch_list cuda_arch_list_length)
if(cuda_arch_list_length EQUAL 0)
  list(APPEND cuda_arch_list "86")
  message(STATUS "Assign default GPU architecture sm=86")
endif()

foreach(cuda_arch ${cuda_arch_list})
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_${cuda_arch},code=sm_${cuda_arch}")
endforeach()

# set(CUDA_SEPARABLE_COMPILATION ON)  # Temporarily disabled due to HierarchicalKV symbol conflicts


# third party
add_subdirectory(third_party)


include_directories(
  ${PROJECT_SOURCE_DIR}/libmlkvplus 
  ${PROJECT_SOURCE_DIR}/libmlkvplus/include
  # third party
  ${PROJECT_SOURCE_DIR}/third_party
  ${PROJECT_SOURCE_DIR}/third_party/HierarchicalKV/include
  ${PROJECT_SOURCE_DIR}/third_party/rocksdb
  ${PROJECT_SOURCE_DIR}/third_party/rocksdb/include
)

include_directories(BEFORE ${PROJECT_SOURCE_DIR}/libmlkvplus/rocksdb)
include_directories(BEFORE ${PROJECT_SOURCE_DIR}/libmlkvplus/rocksdb/include)
include_directories(BEFORE ${PROJECT_SOURCE_DIR}/libmlkvplus/HierarchicalKV/include)


include_directories($ENV{CONDA_PREFIX}/include)
link_directories($ENV{CONDA_PREFIX}/lib)

set(MLKV_PLUS_SOURCES
    libmlkvplus/src/mlkv_plus.cc.cu
    libmlkvplus/src/gpu_tree_hkv.cc.cu
    libmlkvplus/src/memdisk_tree_rocksdb.cc.cu
    libmlkvplus/gds/gwal_writer.cu
    libmlkvplus/gds/gds_block_based_table_reader.cu
    libmlkvplus/gds/gds_block_based_table_factory.cu
    libmlkvplus/gds/gds_block.cu
    libmlkvplus/gds/gds_db_accessor.cu
)




# Function to inherit RocksDB compile definitions
# This gets the directory-level compile definitions from the rocksdb directory
function(inherit_rocksdb_definitions target_name)
    # Get directory properties from the rocksdb source directory
    get_directory_property(ROCKSDB_COMPILE_DEFS 
        DIRECTORY ${CMAKE_SOURCE_DIR}/third_party/rocksdb 
        COMPILE_DEFINITIONS)
    
    if(ROCKSDB_COMPILE_DEFS)
        message(STATUS "Inheriting directory compile definitions from rocksdb: ${ROCKSDB_COMPILE_DEFS}")
        target_compile_definitions(${target_name} PRIVATE ${ROCKSDB_COMPILE_DEFS})
    else()
        message(STATUS "No directory compile definitions found, using essential RocksDB definitions")
        # Essential definitions that RocksDB needs
        target_compile_definitions(${target_name} PRIVATE
            OS_LINUX
            ROCKSDB_PLATFORM_POSIX
            ROCKSDB_LIB_IO_POSIX
        )
    endif()
    
    # Also try to get target-specific definitions
    get_target_property(ROCKSDB_TARGET_COMPILE_DEFS rocksdb-shared COMPILE_DEFINITIONS)
    if(ROCKSDB_TARGET_COMPILE_DEFS)
        message(STATUS "Inheriting target compile definitions from rocksdb-shared: ${ROCKSDB_TARGET_COMPILE_DEFS}")
        target_compile_definitions(${target_name} PRIVATE ${ROCKSDB_TARGET_COMPILE_DEFS})
    endif()
    
    # Inherit interface compile definitions
    get_target_property(ROCKSDB_INTERFACE_COMPILE_DEFS rocksdb-shared INTERFACE_COMPILE_DEFINITIONS)
    if(ROCKSDB_INTERFACE_COMPILE_DEFS)
        message(STATUS "Inheriting interface compile definitions from rocksdb-shared: ${ROCKSDB_INTERFACE_COMPILE_DEFS}")
        target_compile_definitions(${target_name} PRIVATE ${ROCKSDB_INTERFACE_COMPILE_DEFS})
    endif()
    
    # Inherit include directories
    get_target_property(ROCKSDB_INCLUDE_DIRS rocksdb-shared INTERFACE_INCLUDE_DIRECTORIES)
    if(ROCKSDB_INCLUDE_DIRS)
        message(STATUS "Inheriting include directories from rocksdb-shared: ${ROCKSDB_INCLUDE_DIRS}")
        target_include_directories(${target_name} PRIVATE ${ROCKSDB_INCLUDE_DIRS})
    endif()
endfunction()


add_library(mlkv_plus SHARED
    ${MLKV_PLUS_SOURCES}
)
target_compile_features(mlkv_plus PUBLIC cxx_std_20 cuda_std_20)
set_target_properties(mlkv_plus PROPERTIES 
                          CUDA_ARCHITECTURES OFF
                          POSITION_INDEPENDENT_CODE ON)
                          # CUDA_SEPARABLE_COMPILATION ON)  # Disabled due to HierarchicalKV conflicts
target_link_libraries(mlkv_plus PUBLIC rocksdb-shared pthread cufile OpenMP::OpenMP_CXX)
inherit_rocksdb_definitions(mlkv_plus)
target_compile_definitions(mlkv_plus PRIVATE CUDA_ERROR_CHECK)



# test
add_subdirectory(test)

# gYCSB Benchmark
add_subdirectory(gYCSB)