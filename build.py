#!/usr/bin/env python3
"""
Setup script for building MLKV Plus PyTorch Extension

"""

import os
import sys
import subprocess
from pathlib import Path
from setuptools import setup
import torch
from torch.utils.cpp_extension import (
    CppExtension,
    CUDAExtension,
    BuildExtension,
    CUDA_HOME,
)

if torch.__version__ >= "2.6.0":
    py_limited_api = True
else:
    py_limited_api = False





# Get the directory containing this script
PROJECT_ROOT_DIR = Path(__file__).parent.absolute()
BUILD_DIR = PROJECT_ROOT_DIR / "build"
CONDA_PREFIX = Path(os.getenv("CONDA_PREFIX"))
PROJECT_NAME = "mlkv_plus"
CPP_STANDARD = "c++20"

def rel(p: Path) -> str:
    return os.path.relpath(str(p), start=str(PROJECT_ROOT_DIR)).replace(os.sep, "/")


def get_cuda_architectures():
    """Get CUDA architectures from environment variable or default to 86"""
    sm_env = os.environ.get("CUDA_SM", "86")
    return [arch.strip() for arch in sm_env.split(",") if arch.strip()]


def run_cmake_build():
    """Run CMake configure and build before Python extension build"""
    print("Running CMake build...")
    
    # Get CUDA architectures
    cuda_archs = get_cuda_architectures()
    print(f"Using CUDA architectures: {cuda_archs}")
    
    # Ensure build directory exists
    BUILD_DIR.mkdir(exist_ok=True)
    
    # Configure CMake with CUDA architectures
    cmake_configure_cmd = [
        "cmake",
        "-S", str(PROJECT_ROOT_DIR),
        "-B", str(BUILD_DIR),
        f"-Dsm={';'.join(cuda_archs)}",
        "-DCMAKE_BUILD_TYPE=Debug"
    ]
    
    print(f"CMake configure command: {' '.join(cmake_configure_cmd)}")
    
    try:
        subprocess.run(cmake_configure_cmd, check=True, cwd=str(PROJECT_ROOT_DIR))
        print("CMake configuration completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"CMake configuration failed: {e}")
        sys.exit(1)
    
    # Build with CMake
    cmake_build_cmd = [
        "cmake",
        "--build", str(BUILD_DIR),
        "--parallel", os.environ.get("MAX_JOBS", 8)
    ]
    
    print(f"CMake build command: {' '.join(cmake_build_cmd)}")
    
    try:
        subprocess.run(cmake_build_cmd, check=True, cwd=str(PROJECT_ROOT_DIR))
        print("CMake build completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"CMake build failed: {e}")
        sys.exit(1)
    
    # Install CMake targets (especially python_binding component)
    cmake_install_cmd = [
        "cmake",
        "--install", str(BUILD_DIR),
        "--component", "ycsb_binding"
    ]
    
    print(f"CMake install command: {' '.join(cmake_install_cmd)}")
    
    try:
        subprocess.run(cmake_install_cmd, check=True, cwd=str(PROJECT_ROOT_DIR))
        print("CMake install completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"CMake install failed: {e}")
        # Note: Install failure is not always critical, continue with Python extension build
        print("Warning: CMake install failed, but continuing with Python extension build")

def mlkv_plus_torch_binding():
    cxx_args = [
        "-std=c++20",
        "-g",
        "-fdiagnostics-color=always",
        "-DPy_LIMITED_API=0x03120000",  # min CPython version 3.12
    ]
    
    # Get CUDA architectures from environment
    arch = get_cuda_architectures()
    
    nvcc_args = [
        f"-std={CPP_STANDARD}",
        "-g",
    ]
    
    for a in arch:
        nvcc_args.append(f"-gencode=arch=compute_{a},code=sm_{a}")
    
    
    source_files = [
        rel(PROJECT_ROOT_DIR / "libmlkvplus" / "torch_binding" / "dummy_var_handle.cu"),
        rel(PROJECT_ROOT_DIR / "libmlkvplus" / "torch_binding" / "dummy_var_ops.cu"),
        rel(PROJECT_ROOT_DIR / "libmlkvplus" / "torch_binding" / "dummy_var.cu"),
        rel(PROJECT_ROOT_DIR / "libmlkvplus" / "torch_binding" / "register.cc"),
    ]
    
    include_dirs = [
        str(PROJECT_ROOT_DIR / "libmlkvplus"),
        str(PROJECT_ROOT_DIR / "libmlkvplus" / "include"),
        str(PROJECT_ROOT_DIR / "libmlkvplus" / "torch_binding"),
        str(PROJECT_ROOT_DIR / "third_party" / "HierarchicalKV" / "include"),
        str(PROJECT_ROOT_DIR / "libmlkvplus" / "rocksdb" / "include"), # match priority with original rocksdb
        str(PROJECT_ROOT_DIR / "third_party" / "rocksdb" / "include"),
    ]
    
    library_dirs = [
        str(BUILD_DIR),
    ]
    
    print(f"Building MLKV Plus PyTorch Extension...")
    print(f"Include dirs: {include_dirs}")
    print(f"Source files: {source_files}")
    
    
    rpaths = []
    rpaths.append(str(BUILD_DIR))
    
    extra_link_args = []
    if rpaths:
        extra_link_args.append(f"-Wl,-rpath,{','.join(rpaths)}")
        
    
    # Create the extension
    ext_module = CUDAExtension(
            name=f"{PROJECT_NAME}.libmlkvplus_torch",
            sources=source_files,
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            libraries=["mlkv_plus"],
            extra_compile_args={
                "cxx": cxx_args,
                "nvcc": nvcc_args,
            },
            extra_link_args=extra_link_args,
            py_limited_api=py_limited_api,
        )
    
    return ext_module

def sok():
    base_dir = PROJECT_ROOT_DIR / "third_party" / "HugeCTR" / "sparse_operation_kit"
    
    source_files = [
        rel(base_dir / "kit_src" / "py_init.cc"),
        rel(base_dir / "kit_src" / "lookup" / "binding" / "select.cu"),
        rel(base_dir / "kit_src" / "lookup" / "binding" / "reorder.cu"),
        rel(base_dir / "kit_src" / "lookup" / "impl" / "reorder_kernel.cu"),
        rel(base_dir / "kit_src" / "lookup" / "impl" / "select_kernel.cu"),
    ]
    
    
    include_dirs = [
        str(base_dir / "kit_src"),
    ]
    
    cxx_args = [
        f"-std={CPP_STANDARD}",
        "-g",
        "-fdiagnostics-color=always",
        "-DPy_LIMITED_API=0x03120000",  # min CPython version 3.12
    ]
    
    # Get CUDA architectures from environment
    arch = get_cuda_architectures()
    
    nvcc_args = [
        "-std=c++20",
        "-g",
    ]
    for a in arch:
        nvcc_args.append(f"-gencode=arch=compute_{a},code=sm_{a}")
        
    # Create the extension
    ext_module = CUDAExtension(
            name=f"{PROJECT_NAME}.sok",
            sources=source_files,
            include_dirs=include_dirs,
            extra_compile_args={
                "cxx": cxx_args,
                "nvcc": nvcc_args,
            },
            py_limited_api=py_limited_api,
        )
    
    return ext_module
    
    

def build(setup_kwargs):
    import os
    
    # Check if we're installing benchmark-only mode
    benchmark_only = os.environ.get("gYCSB_ONLY", "false").lower() == "true"

    if benchmark_only:
        print("Installing gYCSB only...")
        # Still need to call setup() for metadata generation, but without extensions
        setup(
            ext_modules=[],
            cmdclass={}
        )
        return
    else:
        print("Installing MLKV+ (PyTorch + libmlkvplus) and gYCSB...")
        
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("Warning: CUDA not available, but SOK requires CUDA")
        print("Please install CUDA and PyTorch with CUDA support")
        sys.exit(1)
        
    # Run CMake build first
    print("Step 1: Running CMake build...")
    run_cmake_build()
    
    # Then build Python extensions
    print("Step 2: Building Python extensions...")
    ext_modules = [sok(), mlkv_plus_torch_binding()]
    
    
    setup_kwargs.update({
        "ext_modules": ext_modules,
        "cmdclass": {"build_ext": BuildExtension}
    })
