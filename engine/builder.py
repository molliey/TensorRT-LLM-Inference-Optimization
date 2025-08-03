import tensorrt as trt
import numpy as np
import os
from typing import Optional, Dict, Any, List
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class TensorRTBuilder:
    def __init__(self, 
                 max_workspace_size: int = 1 << 30,  # 1GB
                 precision: str = "fp16",
                 max_batch_size: int = 8,
                 verbose: bool = False):
        self.max_workspace_size = max_workspace_size
        self.precision = precision
        self.max_batch_size = max_batch_size
        self.verbose = verbose
        
        # Initialize TensorRT components
        self.logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
        self.builder = trt.Builder(self.logger)
        self.network = None
        self.config = None
        self.engine = None
        self.engine_path = None
    
    def build_engine_from_onnx(self, 
                              onnx_path: str,
                              engine_path: str,
                              optimization_profiles: Optional[List[Dict]] = None) -> str:
        logger.info(f"Building TensorRT engine from ONNX: {onnx_path}")
        
        # Create network
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        self.network = self.builder.create_network(network_flags)
        
        # Create ONNX parser
        parser = trt.OnnxParser(self.network, self.logger)
        
        # Parse ONNX model
        with open(onnx_path, 'rb') as model_file:
            success = parser.parse(model_file.read())
            if not success:
                error_msgs = []
                for idx in range(parser.num_errors):
                    error_msgs.append(parser.get_error(idx))
                raise RuntimeError(f"Failed to parse ONNX model: {error_msgs}")
        
        logger.info("ONNX model parsed successfully")
        
        # Create builder configuration
        self.config = self.builder.create_builder_config()
        self.config.max_workspace_size = self.max_workspace_size
        
        # Set precision
        if self.precision == "fp16":
            if self.builder.platform_has_fast_fp16:
                self.config.set_flag(trt.BuilderFlag.FP16)
                logger.info("FP16 precision enabled")
            else:
                logger.warning("FP16 not supported on this platform, using FP32")
        elif self.precision == "int8":
            if self.builder.platform_has_fast_int8:
                self.config.set_flag(trt.BuilderFlag.INT8)
                logger.info("INT8 precision enabled")
            else:
                logger.warning("INT8 not supported on this platform, using FP32")
        
        # Setup optimization profiles for dynamic shapes
        if optimization_profiles:
            self._setup_optimization_profiles(optimization_profiles)
        else:
            self._setup_default_optimization_profile()
        
        # Build engine
        logger.info("Building TensorRT engine...")
        self.engine = self.builder.build_engine(self.network, self.config)
        
        if self.engine is None:
            raise RuntimeError("Failed to build TensorRT engine")
        
        # Serialize and save engine
        os.makedirs(os.path.dirname(engine_path), exist_ok=True)
        with open(engine_path, 'wb') as f:
            f.write(self.engine.serialize())
        
        self.engine_path = engine_path
        logger.info(f"TensorRT engine saved to: {engine_path}")
        
        return engine_path
    
    def _setup_optimization_profiles(self, profiles: List[Dict]):
        for i, profile_config in enumerate(profiles):
            profile = self.builder.create_optimization_profile()
            
            for input_name, shapes in profile_config.items():
                min_shape = shapes.get('min', [1, 1])
                opt_shape = shapes.get('opt', [1, 512])
                max_shape = shapes.get('max', [8, 1024])
                
                profile.set_shape(input_name, min_shape, opt_shape, max_shape)
                logger.info(f"Profile {i} - {input_name}: min={min_shape}, opt={opt_shape}, max={max_shape}")
            
            self.config.add_optimization_profile(profile)
    
    def _setup_default_optimization_profile(self):
        profile = self.builder.create_optimization_profile()
        
        # Get input tensor info
        input_tensor = self.network.get_input(0)
        input_name = input_tensor.name
        
        # Define shapes for input_ids (batch_size, sequence_length)
        min_shape = [1, 1]      # Minimum: batch=1, seq=1
        opt_shape = [4, 512]    # Optimal: batch=4, seq=512
        max_shape = [8, 1024]   # Maximum: batch=8, seq=1024
        
        profile.set_shape(input_name, min_shape, opt_shape, max_shape)
        self.config.add_optimization_profile(profile)
        
        logger.info(f"Default optimization profile set for {input_name}")
        logger.info(f"Shapes - min: {min_shape}, opt: {opt_shape}, max: {max_shape}")
    
    def enable_dla(self, dla_core: int = 0):
        if self.builder.num_DLA_cores > 0:
            self.config.default_device_type = trt.DeviceType.DLA
            self.config.DLA_core = dla_core
            self.config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
            logger.info(f"DLA core {dla_core} enabled with GPU fallback")
        else:
            logger.warning("No DLA cores available")
    
    def enable_tactic_sources(self, sources: List[str]):
        tactic_source_map = {
            'cublas': trt.TacticSource.CUBLAS,
            'cublaslt': trt.TacticSource.CUBLAS_LT,
            'cudnn': trt.TacticSource.CUDNN,
            'edge_mask_convolutions': trt.TacticSource.EDGE_MASK_CONVOLUTIONS,
            'jit_convolutions': trt.TacticSource.JIT_CONVOLUTIONS
        }
        
        tactic_sources = 0
        for source in sources:
            if source.lower() in tactic_source_map:
                tactic_sources |= 1 << int(tactic_source_map[source.lower()])
        
        self.config.set_tactic_sources(tactic_sources)
        logger.info(f"Enabled tactic sources: {sources}")
    
    def add_calibration_cache(self, cache_path: str):
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                cache_data = f.read()
            self.config.set_quantization_flag(trt.QuantizationFlag.CALIBRATE_BEFORE_FUSION)
            logger.info(f"Loaded calibration cache from: {cache_path}")
        else:
            logger.warning(f"Calibration cache not found: {cache_path}")
    
    def get_engine_info(self) -> Dict[str, Any]:
        if self.engine is None:
            return {"error": "Engine not built"}
        
        info = {
            "engine_path": self.engine_path,
            "max_batch_size": self.engine.max_batch_size,
            "num_bindings": self.engine.num_bindings,
            "num_layers": self.engine.num_layers,
            "workspace_size": self.max_workspace_size,
            "precision": self.precision,
            "bindings": []
        }
        
        # Get binding information
        for i in range(self.engine.num_bindings):
            binding_name = self.engine.get_binding_name(i)
            binding_shape = self.engine.get_binding_shape(i)
            binding_dtype = self.engine.get_binding_dtype(i)
            is_input = self.engine.binding_is_input(i)
            
            info["bindings"].append({
                "name": binding_name,
                "shape": list(binding_shape),
                "dtype": str(binding_dtype),
                "is_input": is_input
            })
        
        return info
    
    def cleanup(self):
        if self.engine:
            del self.engine
        if self.config:
            del self.config
        if self.network:
            del self.network
        if self.builder:
            del self.builder
        
        logger.info("TensorRT components cleaned up")