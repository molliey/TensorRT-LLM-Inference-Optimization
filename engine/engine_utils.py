import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import torch
from typing import Dict, Any, List, Optional, Tuple, Union
import os
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class EngineUtils:
    def __init__(self, engine_path: Optional[str] = None):
        self.engine_path = engine_path
        self.runtime = None
        self.engine = None
        self.context = None
        self.stream = None
        self.bindings = []
        self.binding_addrs = {}
        self.input_specs = []
        self.output_specs = []
        
        if engine_path and os.path.exists(engine_path):
            self.load_engine(engine_path)
    
    def load_engine(self, engine_path: str) -> bool:
        try:
            logger.info(f"Loading TensorRT engine from: {engine_path}")
            
            # Initialize TensorRT runtime
            trt_logger = trt.Logger(trt.Logger.WARNING)
            self.runtime = trt.Runtime(trt_logger)
            
            # Load engine
            with open(engine_path, 'rb') as f:
                engine_data = f.read()
            
            self.engine = self.runtime.deserialize_cuda_engine(engine_data)
            if self.engine is None:
                raise RuntimeError("Failed to deserialize engine")
            
            # Create execution context
            self.context = self.engine.create_execution_context()
            
            # Create CUDA stream
            self.stream = cuda.Stream()
            
            # Setup bindings
            self._setup_bindings()
            
            self.engine_path = engine_path
            logger.info("Engine loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load engine: {e}")
            return False
    
    def _setup_bindings(self):
        self.bindings = []
        self.binding_addrs = {}
        self.input_specs = []
        self.output_specs = []
        
        for i in range(self.engine.num_bindings):
            binding_name = self.engine.get_binding_name(i)
            binding_shape = self.engine.get_binding_shape(i)
            binding_dtype = self.engine.get_binding_dtype(i)
            is_input = self.engine.binding_is_input(i)
            
            # Convert TensorRT dtype to numpy dtype
            if binding_dtype == trt.DataType.FLOAT:
                dtype = np.float32
            elif binding_dtype == trt.DataType.HALF:
                dtype = np.float16
            elif binding_dtype == trt.DataType.INT32:
                dtype = np.int32
            elif binding_dtype == trt.DataType.INT8:
                dtype = np.int8
            else:
                dtype = np.float32
            
            # Calculate size
            size = np.prod(binding_shape) * np.dtype(dtype).itemsize
            
            # Allocate GPU memory
            device_mem = cuda.mem_alloc(size)
            self.bindings.append(device_mem)
            self.binding_addrs[binding_name] = device_mem
            
            binding_info = {
                'name': binding_name,
                'shape': list(binding_shape),
                'dtype': dtype,
                'size': size,
                'device_ptr': device_mem
            }
            
            if is_input:
                self.input_specs.append(binding_info)
            else:
                self.output_specs.append(binding_info)
        
        logger.info(f"Setup {len(self.input_specs)} inputs and {len(self.output_specs)} outputs")
    
    def set_input_shape(self, input_name: str, shape: List[int]) -> bool:
        try:
            binding_index = self.engine.get_binding_index(input_name)
            self.context.set_binding_shape(binding_index, shape)
            
            # Update input spec
            for spec in self.input_specs:
                if spec['name'] == input_name:
                    spec['shape'] = shape
                    # Reallocate memory if needed
                    new_size = np.prod(shape) * np.dtype(spec['dtype']).itemsize
                    if new_size > spec['size']:
                        spec['device_ptr'].free()
                        spec['device_ptr'] = cuda.mem_alloc(new_size)
                        spec['size'] = new_size
                        self.binding_addrs[input_name] = spec['device_ptr']
                    break
            
            return True
        except Exception as e:
            logger.error(f"Failed to set input shape: {e}")
            return False
    
    def infer(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        try:
            # Set dynamic shapes if needed
            for input_name, input_data in inputs.items():
                if input_data.shape != self._get_binding_shape(input_name):
                    self.set_input_shape(input_name, list(input_data.shape))
            
            # Copy inputs to GPU
            for input_name, input_data in inputs.items():
                device_ptr = self.binding_addrs[input_name]
                cuda.memcpy_htod_async(device_ptr, input_data.astype(self._get_binding_dtype(input_name)), self.stream)
            
            # Execute inference
            binding_ptrs = [int(binding) for binding in self.bindings]
            success = self.context.execute_async_v2(binding_ptrs, self.stream.handle)
            
            if not success:
                raise RuntimeError("Inference execution failed")
            
            # Copy outputs from GPU
            outputs = {}
            for output_spec in self.output_specs:
                output_name = output_spec['name']
                output_shape = self._get_current_binding_shape(output_name)
                output_dtype = output_spec['dtype']
                
                output_data = np.empty(output_shape, dtype=output_dtype)
                cuda.memcpy_dtoh_async(output_data, output_spec['device_ptr'], self.stream)
                outputs[output_name] = output_data
            
            # Synchronize stream
            self.stream.synchronize()
            
            return outputs
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise
    
    def infer_batch(self, batch_inputs: List[Dict[str, np.ndarray]]) -> List[Dict[str, np.ndarray]]:
        results = []
        for inputs in batch_inputs:
            result = self.infer(inputs)
            results.append(result)
        return results
    
    def benchmark_inference(self, 
                          inputs: Dict[str, np.ndarray], 
                          num_runs: int = 100,
                          warmup_runs: int = 10) -> Dict[str, float]:
        logger.info(f"Benchmarking inference with {num_runs} runs...")
        
        # Warmup
        for _ in range(warmup_runs):
            self.infer(inputs)
        
        # Benchmark
        import time
        cuda.Context.synchronize()
        start_time = time.time()
        
        for _ in range(num_runs):
            self.infer(inputs)
        
        cuda.Context.synchronize()
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / num_runs
        throughput = num_runs / total_time
        
        return {
            "avg_inference_time_ms": avg_time * 1000,
            "throughput_inferences_per_sec": throughput,
            "total_time_sec": total_time,
            "num_runs": num_runs
        }
    
    def profile_inference(self, 
                         inputs: Dict[str, np.ndarray],
                         profile_path: str = "profile.json") -> Dict[str, Any]:
        try:
            # Enable profiling
            profiler = trt.Profiler()
            self.context.profiler = profiler
            
            # Run inference with profiling
            result = self.infer(inputs)
            
            # Get profiling results
            profile_data = {
                "layers": [],
                "total_time_ms": 0
            }
            
            # Note: Actual profiling data extraction depends on TensorRT version
            # This is a placeholder for profiling functionality
            
            # Save profile data
            with open(profile_path, 'w') as f:
                json.dump(profile_data, f, indent=2)
            
            logger.info(f"Profiling data saved to: {profile_path}")
            return profile_data
            
        except Exception as e:
            logger.error(f"Profiling failed: {e}")
            return {"error": str(e)}
    
    def _get_binding_shape(self, binding_name: str) -> List[int]:
        binding_index = self.engine.get_binding_index(binding_name)
        return list(self.engine.get_binding_shape(binding_index))
    
    def _get_current_binding_shape(self, binding_name: str) -> List[int]:
        binding_index = self.engine.get_binding_index(binding_name)
        return list(self.context.get_binding_shape(binding_index))
    
    def _get_binding_dtype(self, binding_name: str):
        for spec in self.input_specs + self.output_specs:
            if spec['name'] == binding_name:
                return spec['dtype']
        return np.float32
    
    def get_engine_info(self) -> Dict[str, Any]:
        if self.engine is None:
            return {"error": "Engine not loaded"}
        
        info = {
            "engine_path": self.engine_path,
            "max_batch_size": getattr(self.engine, 'max_batch_size', 'N/A'),
            "num_bindings": self.engine.num_bindings,
            "num_layers": getattr(self.engine, 'num_layers', 'N/A'),
            "device_memory_size": getattr(self.engine, 'device_memory_size', 'N/A'),
            "workspace_size": getattr(self.engine, 'workspace_size', 'N/A'),
            "inputs": [],
            "outputs": []
        }
        
        for spec in self.input_specs:
            info["inputs"].append({
                "name": spec['name'],
                "shape": spec['shape'],
                "dtype": str(spec['dtype'])
            })
        
        for spec in self.output_specs:
            info["outputs"].append({
                "name": spec['name'],
                "shape": spec['shape'],
                "dtype": str(spec['dtype'])
            })
        
        return info
    
    def save_engine_info(self, info_path: str):
        info = self.get_engine_info()
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        logger.info(f"Engine info saved to: {info_path}")
    
    def cleanup(self):
        logger.info("Cleaning up engine resources...")
        
        try:
            # Free GPU memory
            for binding in self.bindings:
                if hasattr(binding, 'free'):
                    binding.free()
            
            # Cleanup CUDA resources
            if self.stream:
                del self.stream
            
            # Cleanup TensorRT resources
            if self.context:
                del self.context
            if self.engine:
                del self.engine
            if self.runtime:
                del self.runtime
            
            self.bindings.clear()
            self.binding_addrs.clear()
            self.input_specs.clear()
            self.output_specs.clear()
            
            logger.info("Cleanup completed")
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
    
    def __del__(self):
        self.cleanup()

    @staticmethod
    def validate_engine(engine_path: str) -> bool:
        try:
            trt_logger = trt.Logger(trt.Logger.ERROR)
            runtime = trt.Runtime(trt_logger)
            
            with open(engine_path, 'rb') as f:
                engine_data = f.read()
            
            engine = runtime.deserialize_cuda_engine(engine_data)
            success = engine is not None
            
            if engine:
                del engine
            del runtime
            
            return success
            
        except Exception as e:
            logger.error(f"Engine validation failed: {e}")
            return False
    
    @staticmethod
    def compare_engines(engine1_path: str, engine2_path: str) -> Dict[str, Any]:
        try:
            utils1 = EngineUtils(engine1_path)
            utils2 = EngineUtils(engine2_path)
            
            info1 = utils1.get_engine_info()
            info2 = utils2.get_engine_info()
            
            comparison = {
                "engine1": info1,
                "engine2": info2,
                "differences": {}
            }
            
            # Compare key metrics
            for key in ['num_bindings', 'num_layers', 'device_memory_size']:
                if key in info1 and key in info2:
                    if info1[key] != info2[key]:
                        comparison["differences"][key] = {
                            "engine1": info1[key],
                            "engine2": info2[key]
                        }
            
            utils1.cleanup()
            utils2.cleanup()
            
            return comparison
            
        except Exception as e:
            return {"error": str(e)}