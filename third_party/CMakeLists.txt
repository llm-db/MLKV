function(build_sok)
  message("=== Dependency: Merlin HugeCTR Sparse Operation Kit ===================================================")
  set(BUILD_GMOCK ON)
  add_subdirectory(HugeCTR/sparse_operation_kit)
  message("=== End of Dependency: Merlin HugeCTR Sparse Operation Kit ===============================================")
endfunction()
# build_sok()


function(build_rocksdb)
  message("=== Dependency: RocksDB ===============================================")
  set(FAIL_ON_WARNINGS OFF)
  set(WITH_ALL_TESTS OFF)

  # copy the mutated cmake files to the rocksdb directory
  file(COPY ${PROJECT_SOURCE_DIR}/libmlkvplus/rocksdb/cmake/CMakeLists.txt DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/rocksdb)
  file(COPY ${PROJECT_SOURCE_DIR}/libmlkvplus/rocksdb/cmake/Finduring.cmake DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/rocksdb/cmake/modules)
  message(STATUS "Copied the mutated cmake files to the rocksdb directory")
  
  add_subdirectory(rocksdb)
  message("=== End of Dependency: RocksDB ===============================================")
endfunction()
build_rocksdb()


