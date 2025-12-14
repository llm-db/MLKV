#!/usr/bin/env python3
"""
Setup script for building SOK (Sparse Operation Kit) PyTorch Extension

This script builds the PyTorch custom operators for DummyVar functionality,
which provides hash table operations for sparse embedding tables.
"""

import os
import sys
import glob
from pathlib import Path
from setuptools import setup, Extension, find_packages
import torch
from torch.utils.cpp_extension import (
    CppExtension,
    CUDAExtension,
    BuildExtension,
    CUDA_HOME,
)
from distutils.sysconfig import get_python_inc

if torch.__version__ >= "2.6.0":
    py_limited_api = True
else:
    py_limited_api = False


# Get the directory containing this script
SOK_DIR = Path(__file__).parent.absolute()
ROOT_DIR = SOK_DIR.parent.parent.parent

def setup_third_party_dependencies():
    """Setup third-party dependencies by copying them to local third_party directory"""
    print("Setting up third-party dependencies...")
    
    # Create third_party directory if it doesn't exist
    third_party_dir = SOK_DIR / "third_party"
    if not third_party_dir.exists():
        os.system(f"mkdir -p {third_party_dir}")
    
    # Copy json library
    json_src = SOK_DIR.parent.parent / "json"
    json_dst = third_party_dir / "json"
    if json_src.exists() and not json_dst.exists():
        print(f"Copying json library from {json_src} to {json_dst}")
        os.system(f"cp -r {json_src} {json_dst}")
    
    # Copy HierarchicalKV library  
    hkv_src = SOK_DIR.parent.parent / "HierarchicalKV"
    hkv_dst = third_party_dir / "HierarchicalKV"
    print(f"Copying HierarchicalKV library from {hkv_src} to {hkv_dst}")
    if hkv_dst.exists():
        # remove the hkv_dst directory
        os.system(f"rm -rf {hkv_dst}")
    
    
    os.system(f"cp -r {hkv_src} {hkv_dst}")
    
    print("Third-party dependencies setup complete")

def get_include_dirs():
    """Get all necessary include directories"""
    include_dirs = [
        str(SOK_DIR / "kit_src"),
        str(SOK_DIR / "kit_src" / "variable"),
        str(SOK_DIR / "kit_src" / "variable" / "binding"),
        str(SOK_DIR / "kit_src" / "variable" / "impl"),
        str(SOK_DIR / "kit_src" / "lookup"),
        str(SOK_DIR / "kit_src" / "lookup" / "binding"),
        str(SOK_DIR / "kit_src" / "lookup" / "impl"),
        str(SOK_DIR / "kit_src" / "common"),
        # Add cuCollections and cuco include paths
        str(SOK_DIR / "kit_src" / "variable" / "impl" / "dynamic_embedding_table"),
        str(SOK_DIR / "kit_src" / "variable" / "impl" / "dynamic_embedding_table" / "cuCollections" / "include"),
        str(SOK_DIR / "kit_src" / "variable" / "impl" / "dynamic_embedding_table" / "cudf"),
        # Add third-party include paths
        str(SOK_DIR / "third_party" / "json" / "include"),
        str(SOK_DIR / "third_party" / "HierarchicalKV" / "include"),
    ]
    
    return include_dirs

def get_source_files():
    """Get all source files to compile"""
    source_files = [
        str(SOK_DIR / "kit_src" / "py_init.cc"),
        str(SOK_DIR / "kit_src" / "register.cc"),
        # Core kernel implementations
        str(SOK_DIR / "kit_src" / "variable" / "binding" / "dummy_var_handle_torch.cu"),
        str(SOK_DIR / "kit_src" / "variable" / "binding" / "dummy_var.cc"),
        str(SOK_DIR / "kit_src" / "variable" / "binding" / "dummy_var_ops_torch.cu"),
        
        
        str(SOK_DIR / "kit_src" / "lookup" / "binding" / "select.cu"),
        str(SOK_DIR / "kit_src" / "lookup" / "binding" / "reorder.cu"),
    ]
    
    # Find additional required source files
    impl_dirs = [
        SOK_DIR / "kit_src" / "variable" / "impl",
        SOK_DIR / "kit_src" / "lookup" / "impl",
    ]
    
    for impl_dir in impl_dirs:
        if impl_dir.exists():
            impl_files = list(impl_dir.glob("*.cc")) + list(impl_dir.glob("*.cu"))
        # Avoid duplicates
        for f in impl_files:
            f_str = str(f)
            if f_str not in source_files:
                source_files.append(f_str)
    
    
    return source_files


def get_compile_args():
    """Get compilation arguments"""
    cxx_args = [
        "-std=c++17",
        "-g",
        "-fdiagnostics-color=always",
        "-DPy_LIMITED_API=0x03090000",  # min CPython version 3.9
    ]
    
    arch = ["86"]
    
    nvcc_args = [
        "-std=c++17",
        "-g",
        "-DTORCH_USE_CUDA_DSA=1"
    ]
    
    for a in arch:
        nvcc_args.append(f"-gencode=arch=compute_{a},code=sm_{a}")
    
    return cxx_args, nvcc_args



def main():
    """Main setup function"""
    
    # Setup third-party dependencies first
    setup_third_party_dependencies()
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("Warning: CUDA not available, but SOK requires CUDA")
        print("Please install CUDA and PyTorch with CUDA support")
        sys.exit(1)
    
    # Get compilation parameters
    include_dirs = get_include_dirs()
    source_files = get_source_files()
    # libraries = get_libraries()
    cxx_args, nvcc_args = get_compile_args()
    
    print(f"Building SOK PyTorch Extension...")
    print(f"Include dirs: {include_dirs}")
    print(f"Source files: {source_files}")
    
    # Create the extension
    ext_modules = [
        CUDAExtension(
            name="kit_src.sok",
            sources=source_files,
            include_dirs=include_dirs,
            # libraries=libraries,
            extra_compile_args={
                "cxx": cxx_args,
                "nvcc": nvcc_args,
            },
            py_limited_api=py_limited_api,
        ),
        
    ]
    
    setup(
        name="kit_src",
        version="0.1.0",
        author="Haiqiang Zhang",
        author_email="",
        description="Sparse Operation Kit (SOK) PyTorch Extension",
        long_description="PyTorch custom operators for sparse embedding operations using hash tables",
        long_description_content_type="text/plain",
        packages=find_packages(),
        ext_modules=ext_modules,
        cmdclass={"build_ext": BuildExtension}, 
        python_requires=">=3.9",
        install_requires=[
            "torch>=2.5.0",
            "numpy",
        ]
    )

if __name__ == "__main__":
    main() 