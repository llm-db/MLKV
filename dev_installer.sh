#! /bin/bash


export EXTRA_CXXFLAGS="-I$CONDA_PREFIX/include" 

(cd third_party/rocksdb && make shared_lib -j$(nproc))


(cd build && cmake .. && make -j$(nproc) && make install)