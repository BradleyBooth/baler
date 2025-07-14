# compare.py

import os
import time
import abc
from dataclasses import dataclass
import numpy as np

# External library imports
import zfpy
import blosc2
from external.sz3.pysz import pysz


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


class Benchmark(abc.ABC):
    """
    Abstract Base Class for a compression benchmark.
    This class defines the template for running a benchmark.
    """

    def __init__(self, name: str, output_dir: str, data_original: np.ndarray, names_original: np.ndarray):
        self.name = name
        self.output_dir = output_dir
        self.data_original = data_original
        self.names_original = names_original
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self) -> BenchmarkResult:
        """
        Executes the full benchmark process: compress, decompress, and analyze.
        This is the public-facing method to run a benchmark.
        """
        print(f"\nBenchmarking: {self.name}")

        # 1. Compression
        start_compress = time.perf_counter()
        compressed_data = self._compress()
        end_compress = time.perf_counter()
        compress_time = end_compress - start_compress
        print(f"  Compression time: {compress_time:.3f} seconds")

        compressed_path = self._save_compressed(compressed_data)
        compressed_file_size_bytes = os.path.getsize(compressed_path)
        print(f"  Compressed size: {compressed_file_size_bytes / (1024 * 1024):.3f} MB")

        # 2. Decompression
        start_decompress = time.perf_counter()
        decompressed_data = self._decompress(compressed_data)
        end_decompress = time.perf_counter()
        decompress_time = end_decompress - start_decompress
        print(f"  Decompression time: {decompress_time:.3f} seconds")
        
        self._save_decompressed(decompressed_data)

        # 3. Error Analysis
        metrics = self._analyze_errors(decompressed_data)
        print(f"  -> Done. RMSE: {metrics['rmse']:.2e}, Max Error: {metrics['max_err']:.2e}")
        
        # 4. Create and return result object
        return BenchmarkResult(
            name=self.name,
            size_mb=compressed_file_size_bytes / (1024 * 1024),
            compress_time_sec=compress_time,
            decompress_time_sec=decompress_time,
            **metrics
        )

    def _analyze_errors(self, decompressed_data: np.ndarray) -> dict:
        """Performs error analysis and returns a dictionary of metrics."""
        if decompressed_data.shape != self.data_original.shape:
            raise ValueError(f"Shape mismatch after decompression for {self.name}!")
    
        diff = self.data_original.astype(np.float64) - decompressed_data.astype(np.float64)
        mse = np.mean(diff ** 2)
        rmse = np.sqrt(mse)
        max_abs_err = np.max(np.abs(diff))
        
        data_range = np.max(self.data_original) - np.min(self.data_original)
        psnr = 20 * np.log10(data_range) - 10 * np.log10(mse) if mse > 0 else float('inf')

        return {"rmse": rmse, "max_err": max_abs_err, "psnr": psnr}

    def _save_decompressed(self, decompressed_data: np.ndarray):
        """Saves the decompressed data and names to an NPZ file for inspection."""
        path = os.path.join(self.output_dir, "decompressed.npz")
        np.savez(path, data=decompressed_data, names=self.names_original)
        print(f"  Decompressed file saved to: {path}")

    @abc.abstractmethod
    def _compress(self):
        """Subclasses must implement this. Should return the compressed data."""
        pass

    @abc.abstractmethod
    def _decompress(self, compressed_data) -> np.ndarray:
        """Subclasses must implement this. Should return the decompressed numpy array."""
        pass
    
    def _save_compressed(self, compressed_data) -> str:
        """Default method to save compressed data as a binary blob."""
        path = os.path.join(self.output_dir, "compressed.bin")
        with open(path, 'wb') as f:
            f.write(compressed_data)
        return path


# --- Specific Benchmark Implementations ---

class BalerBenchmark(Benchmark):
    """Benchmark for the main 'baler' autoencoder compression."""
    def __init__(self, name: str, output_path: str, compress_func, decompress_func, data_original, names_original):
        # Baler manages its own output directory, so we pass the project's output_path
        super().__init__(name, output_path, data_original, names_original)
        self.compress_func = compress_func
        self.decompress_func = decompress_func

    def _compress(self):
        # Baler's compression function saves its own file and doesn't return data
        self.compress_func()
        return None  # No data to return

    def _decompress(self, compressed_data):
        # Baler's decompression also works on files, then we load the result
        self.decompress_func()
        decompressed_path = os.path.join(self.output_dir, "decompressed_output", "decompressed.npz")
        return np.load(decompressed_path)['data']

    def _save_compressed(self, compressed_data):
        # # This is a no-op because _compress() already saved the file.
        # # We just need to return the path for size calculation.
        # return os.path.join(self.output_dir, "compressed_output", "compressed.npz")
    
        npz_path = os.path.join(self.output_dir, "compressed_output", "compressed.npz")
        with np.load(npz_path) as data:
            for key in data.files:
                if key == 'data':
                    # print(f"Key: {key}, Shape: {data[key].shape}, Dtype: {data[key].dtype}")
                    # with np.printoptions(threshold=np.inf):
                    #     print(data[key][:10])
                    compressed_data = data[key]

        # for the purposes of benchmarking, we need baler compressed data without the normalization features
        no_norm_output_path = os.path.join(self.output_dir, "compressed_output", "no_norm_compressed.npz")
        np.savez(
            no_norm_output_path,
            data=compressed_data,
            names=self.names_original,
        )
        return no_norm_output_path

    def _save_decompressed(self, decompressed_data):
        # This is also a no-op because _decompress() already handled it.
        path = os.path.join(self.output_dir, "decompressed_output", "decompressed.npz")
        print(f"  Decompressed file already saved to: {path}")


class DowncastBenchmark(Benchmark):
    """
    Benchmark for downcasting data to a specified NumPy data type (e.g., float32, float16).
    """
    def __init__(self, output_dir: str, data_original: np.ndarray, names_original: np.ndarray, target_dtype: np.dtype):
        """
        Initializes the benchmark for a specific target data type.
        
        Args:
            output_dir (str): Directory to save benchmark artifacts.
            data_original (np.ndarray): The original, high-precision data.
            names_original (np.ndarray): The names corresponding to the data features.
            target_dtype (np.dtype): The NumPy data type to cast to (e.g., np.float32, np.float16).
        """
        self.target_dtype = target_dtype
        # Automatically generate the name based on the target type
        name = f"Downcast (to {np.dtype(self.target_dtype).name})"
        super().__init__(name, output_dir, data_original, names_original)

    def _compress(self):
        """Performs the downcasting. This is our 'compression' step."""
        return self.data_original.astype(self.target_dtype)

    def _decompress(self, compressed_data):
        """Decompression is a no-op; the compressed data is the final data."""
        return compressed_data

    def _save_compressed(self, compressed_data: np.ndarray) -> str:
        """Override to save as .npz since the 'compressed' data is just a NumPy array."""
        path = os.path.join(self.output_dir, "compressed.npz")
        np.savez(path, data=compressed_data, names=self.names_original)
        return path


class ZFPBenchmark(Benchmark):
    """
    A benchmark class specifically for evaluating the ZFP compression algorithm.
    This class extends the base `Benchmark` to implement compression and decompression
    using the `zfpy` library. It is configured with a dictionary of ZFP-specific
    parameters, such as precision, rate, or tolerance, which are directly passed
    to the `zfpy.compress_numpy` function.
    The name of the benchmark is automatically generated based on the provided
    ZFP parameters to ensure clear identification of the results.
        output_dir (str): The directory where benchmark artifacts (like plots and data) will be saved.
        data_original (np.ndarray): The original, uncompressed data array to be used for the benchmark.
        names_original (np.ndarray): An array of names corresponding to the features in `data_original`.
        zfp_params (dict): A dictionary containing the parameters for ZFP compression. These are passed
                           directly to `zfpy.compress_numpy`. Example: `{'precision': 22}` or `{'rate': 8.0}`.
    Attributes:
        zfp_params (dict): The dictionary of ZFP parameters used for compression.
    """
    def __init__(self, output_dir: str, data_original: np.ndarray, names_original: np.ndarray, zfp_params: dict):
        """
        Initializes the benchmark for a specific set of ZFP parameters.
        
        Args:
            output_dir (str): Directory to save benchmark artifacts.
            data_original (np.ndarray): The original, high-precision data.
            names_original (np.ndarray): The names corresponding to the data features.
            zfp_params (dict): A dictionary of parameters to pass to zfpy.
                               Example: {'precision': 22} or {'rate': 8.0}
        """
        if not zfp_params:
            raise ValueError("zfp_params dictionary cannot be empty for ZFPBenchmark.")
            
        self.zfp_params = zfp_params
        
        # Automatically generate a descriptive name from the parameters
        param_str = ", ".join([f"{k}={v}" for k, v in self.zfp_params.items()])
        name = f"ZFP({param_str})"
        
        super().__init__(name, output_dir, data_original, names_original)

    def _compress(self):
        """Compresses data using the stored ZFP parameters."""
        # The ** operator unpacks the dictionary into keyword arguments
        # e.g., {'rate': 8.0} becomes rate=8.0
        return zfpy.compress_numpy(self.data_original, **self.zfp_params)

    def _decompress(self, compressed_data):
        """Decompression does not require the original parameters."""
        return zfpy.decompress_numpy(compressed_data)


class BloscBenchmark(Benchmark):
    """Benchmark for Blosc2 compression."""
    def __init__(self, name: str, output_dir: str, data_original, names_original, cparams: dict):
        super().__init__(name, output_dir, data_original, names_original)
        self.cparams = cparams

    def _compress(self):
        return blosc2.pack_array2(self.data_original, cparams=self.cparams)

    def _decompress(self, compressed_data):
        return blosc2.unpack_array2(compressed_data)

# TODO Need to get sz3 working properly and understand configuration options
# class SZ3Benchmark(Benchmark):
#     """Benchmark for SZ3 compression."""
    
#     def __init__(self, output_dir: str, data_original: np.ndarray, names_original: np.ndarray, **kwargs):
#         """
#         Initializes the SZ3 benchmark.
        
#         Args:
#             output_dir: Directory for artifacts.
#             data_original: The original data array.
#             names_original: The original names array.
#             **kwargs: SZ3 parameters like mode, abs_val, rel_val.
#                       Example: mode='ABS', abs_val=1e-5
#         """
#         if not kwargs or 'mode' not in kwargs:
#             raise ValueError("SZ3Benchmark requires a 'mode' parameter (e.g., mode='REL', rel_val=1e-4).")

#         # Automatically generate a descriptive name from the parameters
#         param_str = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
#         name = f"SZ3({param_str})"
        
#         super().__init__(name, output_dir, data_original, names_original)
        
#         # Store user-friendly kwargs
#         self.sz_kwargs = kwargs
        
#         # 1. Get the directory containing this script (e.g., /home/bb/baler/modules)
#         this_dir = os.path.dirname(os.path.abspath(__file__))

#         # 2. Construct a path from this script's location to the library
#         #    The path from 'modules/' is '../external/sz3/lib/libSZ3c.so'
#         relative_path_from_script = os.path.join('..', '..', 'external', 'sz3', 'lib', 'libSZ3c.so')

#         # 3. Join the script's directory with the relative path to get the final path
#         sz_library_path = os.path.join(this_dir, relative_path_from_script)

#         # 4. It's good practice to normalize it to clean up any ".."
#         sz_library_path = os.path.normpath(sz_library_path)

#         # 5. Now, instantiate the SZ compressor with the guaranteed correct path
#         self.sz_compressor = pysz.SZ(szpath=sz_library_path)
        
#         # Map our friendly string modes to the library's integer codes
#         self.mode_map = {
#             'ABS': 0,
#             'REL': 1,
#             'ABS_AND_REL': 2,
#             'ABS_OR_REL': 3,
#             'PSNR': 4,
#             # Add other modes if needed
#         }

#     def _compress(self):
#         """Compresses data using the instantiated SZ compressor object."""
#         # Get the integer error mode
#         mode_str = self.sz_kwargs.get('mode')
#         eb_mode = self.mode_map.get(mode_str)
#         if eb_mode is None:
#             raise ValueError(f"Unknown SZ3 mode: {mode_str}")
            
#         # Get error bound values from kwargs, with defaults
#         eb_abs = self.sz_kwargs.get('abs_val', 0.0)
#         eb_rel = self.sz_kwargs.get('rel_val', 0.0)
#         eb_pwr = self.sz_kwargs.get('pwr_val', 0.0)
        
#         # The library's compress method returns a tuple: (compressed_data, ratio)
#         # We only need the data, as our framework calculates size from the saved file.
#         compressed_data, _ = self.sz_compressor.compress(
#             self.data_original, 
#             eb_mode, 
#             eb_abs, 
#             eb_rel, 
#             eb_pwr
#         )
#         return compressed_data

#     def _decompress(self, compressed_data):
#         """Decompresses data using the instantiated SZ compressor object."""
#         return self.sz_compressor.decompress(
#             compressed_data, 
#             self.data_original.shape, 
#             self.data_original.dtype
#         )