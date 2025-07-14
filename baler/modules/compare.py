# Copyright 2022 Baler Contributors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time
import numpy as np
from dataclasses import dataclass

@dataclass
class BenchmarkResult:
    """A container for the results of a single compression benchmark."""
    name: str
    size_mb: float
    compress_time_sec: float
    decompress_time_sec: float
    rmse: float
    max_err: float
    psnr: float

def benchmark_and_analyze(
    name: str, 
    output_dir: str, 
    compress_func, 
    decompress_func, 
    data_original: np.ndarray, 
    names: np.ndarray
) -> BenchmarkResult:
    """
    Performs a single benchmark: compress, decompress, analyze, and save artifacts.
    """
    print(f"\nBenchmarking: {name}")
    
    # 1. Compression
    start_compress = time.perf_counter()
    compressed_data = compress_func()
    end_compress = time.perf_counter()
    compress_time = end_compress - start_compress
    print(f"  Compression time: {compress_time:.3f} seconds")
    
    # safe_name = name.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '').lower()
    compressed_path = os.path.join(output_dir, "compressed.bin")
    with open(compressed_path, 'wb') as f:
        f.write(compressed_data)
    compressed_file_size_bytes = os.path.getsize(compressed_path)
    print(f"  Compressed size: {compressed_file_size_bytes / (1024 * 1024):.3f} MB")

    # 2. Decompression
    start_decompress = time.perf_counter()
    decompressed_data = decompress_func(compressed_data)
    end_decompress = time.perf_counter()
    decompress_time = end_decompress - start_decompress
    print(f"  Decompression time: {decompress_time:.3f} seconds")

    # 3. Save Decompressed Data (with both data and names)
    decompressed_path = os.path.join(output_dir, "decompressed.npz")
    
    np.savez(
        decompressed_path,
        data=decompressed_data,
        names=names,
    )
        
    print(f"  Decompressed file saved to: {decompressed_path}")
    
    # 4. Error Analysis
    if decompressed_data.shape != data_original.shape:
        raise ValueError(f"Shape mismatch after decompression for {name}!")
    
    diff = data_original.astype(np.float64) - decompressed_data.astype(np.float64)
    mse = np.mean(diff ** 2)
    rmse = np.sqrt(mse)
    max_abs_err = np.max(np.abs(diff))
    
    data_range = np.max(data_original) - np.min(data_original)
    psnr = 20 * np.log10(data_range) - 10 * np.log10(mse) if mse > 0 else float('inf')

    print(f"  -> Done. RMSE: {rmse:.2e}, Max Error: {max_abs_err:.2e}")

    # 5. Create and return a single result object
    return BenchmarkResult(
        name=name,
        size_mb=compressed_file_size_bytes / (1024 * 1024),
        compress_time_sec=compress_time,
        decompress_time_sec=decompress_time,
        rmse=rmse,
        max_err=max_abs_err,
        psnr=psnr
    )

def baler_benchmark_analyze(
    name: str, 
    output_path: str,
    compress_func, 
    decompress_func, 
    data_original: np.ndarray, 
    names: np.ndarray
) -> BenchmarkResult:
    """
    Performs a single benchmark: compress, decompress, analyze, and save artifacts.
    """
    print(f"\nBenchmarking: {name}")
    
    # 1. Compression
    start_compress = time.perf_counter()
    compressed_data = compress_func()
    end_compress = time.perf_counter()
    compress_time = end_compress - start_compress
    print(f"  Compression time: {compress_time:.3f} seconds")
    compressed_path = os.path.join(output_path, "compressed_output", "compressed.npz")
    compressed_file_size_bytes = os.path.getsize(compressed_path)
    print(f"  Compressed size: {compressed_file_size_bytes / (1024 * 1024):.3f} MB")

    # 2. Decompression
    start_decompress = time.perf_counter()
    decompressed_data = decompress_func()
    end_decompress = time.perf_counter()
    decompress_time = end_decompress - start_decompress
    print(f"  Decompression time: {decompress_time:.3f} seconds")

    # 3. Save Decompressed Data (with both data and names)
    decompressed_path = os.path.join(output_path, "decompressed_output", "decompressed.npz")       
    print(f"  Decompressed file saved to: {decompressed_path}")
    

    # 4. Error Analysis
    decompressed_data = np.load(decompressed_path)['data']
    if decompressed_data.shape != data_original.shape:
        raise ValueError(f"Shape mismatch after decompression for {name}!")
    
    diff = data_original.astype(np.float64) - decompressed_data.astype(np.float64)
    mse = np.mean(diff ** 2)
    rmse = np.sqrt(mse)
    max_abs_err = np.max(np.abs(diff))
    
    data_range = np.max(data_original) - np.min(data_original)
    psnr = 20 * np.log10(data_range) - 10 * np.log10(mse) if mse > 0 else float('inf')

    print(f"  -> Done. RMSE: {rmse:.2e}, Max Error: {max_abs_err:.2e}")

    # 5. Create and return a single result object
    return BenchmarkResult(
        name=name,
        size_mb=compressed_file_size_bytes / (1024 * 1024),
        compress_time_sec=compress_time,
        decompress_time_sec=decompress_time,
        rmse=rmse,
        max_err=max_abs_err,
        psnr=psnr
    )


def downcast_benchmark_analyze(
    name: str, 
    output_path: str,
    data_original: np.ndarray, 
    names: np.ndarray
) -> BenchmarkResult:
    """
    Performs a single benchmark: compress, decompress, analyze, and save artifacts.
    """
    print(f"\nBenchmarking: {name}")
    
    # 1. Compression
    start_compress = time.perf_counter()
    # Data Type Downcasting (float64 -> float32)
    data_float32 = data_original.astype(np.float32)
    end_compress = time.perf_counter()
    compress_time = end_compress - start_compress
    print(f"  Compression time: {compress_time:.3f} seconds")
    compressed_path = os.path.join(output_path, "compressed.npz")
    
    np.savez(
        compressed_path,
        data=data_float32,
        names=names,
    )

    print(f"  Truncated file saved to: {compressed_path}")
    compressed_file_size_bytes = os.path.getsize(compressed_path)
    print(f"  Truncated size: {compressed_file_size_bytes / (1024 * 1024):.3f} MB")  

    decompressed_path = os.path.join(output_path, "decompressed.npz")
    
    np.savez(
        decompressed_path,
        data=data_float32,
        names=names,
    )
    decompress_time = 0.0 ## No actual decompression needed for downcasting

    diff = data_original - data_float32
    mse = np.mean(diff ** 2)
    rmse = np.sqrt(mse)
    max_abs_err = np.max(np.abs(diff))
    
    data_range = np.max(data_original) - np.min(data_original)
    psnr = 20 * np.log10(data_range) - 10 * np.log10(mse) if mse > 0 else float('inf')

    print(f"  -> Done. RMSE: {rmse:.2e}, Max Error: {max_abs_err:.2e}")

    # 5. Create and return a single result object
    return BenchmarkResult(
        name=name,
        size_mb=compressed_file_size_bytes / (1024 * 1024),
        compress_time_sec=compress_time,
        decompress_time_sec=decompress_time,
        rmse=rmse,
        max_err=max_abs_err,
        psnr=psnr
    )

def sz_benchmark_analyze(
    name: str, 
    output_path: str,
    data_original: np.ndarray, 
    names: np.ndarray
) -> BenchmarkResult:
    """
    Performs a single benchmark: compress, decompress, analyze, and save artifacts.
    """
    print(f"\nBenchmarking: {name}")
    
    # 1. Compression
    start_compress = time.perf_counter()
    compressed_data = data_original.tobytes()  # Simulating compression
    end_compress = time.perf_counter()
    compress_time = end_compress - start_compress
    print(f"  Compression time: {compress_time:.3f} seconds")
    
    compressed_path = os.path.join(output_path, "compressed.bin")
    with open(compressed_path, 'wb') as f:
        f.write(compressed_data)
    
    compressed_file_size_bytes = os.path.getsize(compressed_path)
    print(f"  Compressed size: {compressed_file_size_bytes / (1024 * 1024):.3f} MB")

    # 2. Decompression
    start_decompress = time.perf_counter()
    decompressed_data = np.frombuffer(compressed_data, dtype=data_original.dtype).reshape(data_original.shape)
    end_decompress = time.perf_counter()
    decompress_time = end_decompress - start_decompress
    print(f"  Decompression time: {decompress_time:.3f} seconds")

    # 3. Save Decompressed Data (with both data and names)
    decompressed_path = os.path.join(output_path, "decompressed.npz")
    
    np.savez(
        decompressed_path,
        data=decompressed_data,
        names=names,
    )
        
    print(f"  Decompressed file saved to: {decompressed_path}")
    
    # 4. Error Analysis
    if decompressed_data.shape != data_original.shape:
        raise ValueError(f"Shape mismatch after decompression for {name}!")
    
    diff = data_original.astype(np.float64) - decompressed_data.astype(np.float64)
    mse = np.mean(diff ** 2)
    rmse = np.sqrt(mse)
    max_abs_err = np.max(np.abs(diff))
    
    data_range = np.max(data_original) - np.min(data_original)
    psnr = 20 * np.log10(data_range) - 10 * np.log10(mse)