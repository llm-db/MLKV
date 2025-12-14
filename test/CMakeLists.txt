

# add_executable(integrate_gwal_to_rocksdb_test integrate_gwal_to_rocksdb_test.cu 
#                ${PROJECT_SOURCE_DIR}/libmlkvplus/gds/gwal_writer.cu)
# target_compile_features(integrate_gwal_to_rocksdb_test PUBLIC cxx_std_20 cuda_std_20)
# set_target_properties(integrate_gwal_to_rocksdb_test PROPERTIES CUDA_ARCHITECTURES OFF)
# target_link_libraries(integrate_gwal_to_rocksdb_test PUBLIC rocksdb pthread cufile)



add_executable(run_rocksdb run_rocksdb.cc)
target_compile_features(run_rocksdb PUBLIC cxx_std_20 cuda_std_20)
set_target_properties(run_rocksdb PROPERTIES CUDA_ARCHITECTURES OFF)
target_link_libraries(run_rocksdb PUBLIC rocksdb-shared)


add_executable(gds_bbt_test 
    gds_bbt_test.cu
    ${PROJECT_SOURCE_DIR}/libmlkvplus/gds/gds_block_based_table_factory.cu
    ${PROJECT_SOURCE_DIR}/libmlkvplus/gds/gds_block_based_table_reader.cu
    ${PROJECT_SOURCE_DIR}/libmlkvplus/gds/gds_block.cu
    ${PROJECT_SOURCE_DIR}/libmlkvplus/gds/gds_db_accessor.cu)
target_compile_features(gds_bbt_test PUBLIC cxx_std_20 cuda_std_20)
set_target_properties(gds_bbt_test PROPERTIES CUDA_ARCHITECTURES OFF)
inherit_rocksdb_definitions(gds_bbt_test)
target_link_libraries(gds_bbt_test PUBLIC rocksdb-shared pthread cufile)