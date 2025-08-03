import asyncio
import logging
import time
import numpy as np
import torch
from typing import Dict, Any, List, Optional, AsyncIterator
from transformers import GPT2Tokenizer
import threading
from collections import defaultdict

from engine.engine_utils import EngineUtils
from models.model_utils import ModelUtils

logger = logging.getLogger(__name__)

class TensorRTInferenceEngine:
    def __init__(self, 
                 engine_path: str = "./engines/gpt2_optimized.engine",
                 tokenizer_name: str = "gpt2",
                 max_batch_size: int = 8,
                 max_sequence_length: int = 1024):
        self.engine_path = engine_path
        self.tokenizer_name = tokenizer_name
        self.max_batch_size = max_batch_size
        self.max_sequence_length = max_sequence_length
        
        # Initialize components
        self.engine_utils = None
        self.tokenizer = None
        self.model_info = {}
        
        # Performance tracking
        self.metrics = {
            "total_requests": 0,
            "total_tokens_generated": 0,
            "total_inference_time_ms": 0,
            "average_latency_ms": 0,
            "requests_per_second": 0,
            "tokens_per_second": 0,
            "error_count": 0,
            "last_reset_time": time.time()
        }
        
        # Thread safety
        self.inference_lock = threading.RLock()
        self.metrics_lock = threading.Lock()
        
        # Batch processing
        self.batch_queue = asyncio.Queue(maxsize=max_batch_size * 2)
        self.batch_processing_task = None
        
    async def initialize(self):
        logger.info("Initializing TensorRT Inference Engine...")
        
        try:
            # Load tokenizer
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.tokenizer_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load TensorRT engine
            self.engine_utils = EngineUtils(self.engine_path)
            if not self.engine_utils.engine:
                raise RuntimeError(f"Failed to load TensorRT engine from {self.engine_path}")
            
            # Get model information
            self.model_info = {
                "model_name": self.tokenizer_name,
                "vocab_size": self.tokenizer.vocab_size,
                "max_sequence_length": self.max_sequence_length,
                "max_batch_size": self.max_batch_size,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "bos_token_id": getattr(self.tokenizer, 'bos_token_id', None)
            }
            
            logger.info("TensorRT Inference Engine initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize inference engine: {e}")
            raise
    
    async def generate(self,
                      prompt: str,
                      max_new_tokens: int = 100,
                      temperature: float = 1.0,
                      top_k: int = 50,
                      top_p: float = 0.9,
                      do_sample: bool = True,
                      repetition_penalty: float = 1.0,
                      pad_token_id: Optional[int] = None,
                      eos_token_id: Optional[int] = None) -> Dict[str, Any]:
        
        start_time = time.time()
        
        try:
            with self.inference_lock:
                # Tokenize input
                input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
                input_length = input_ids.shape[1]
                
                if input_length > self.max_sequence_length:
                    raise ValueError(f"Input length {input_length} exceeds maximum {self.max_sequence_length}")
                
                # Generate tokens
                generated_ids = await self._generate_tokens(
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    do_sample=do_sample,
                    repetition_penalty=repetition_penalty,
                    pad_token_id=pad_token_id or self.tokenizer.pad_token_id,
                    eos_token_id=eos_token_id or self.tokenizer.eos_token_id
                )
                
                # Decode generated text
                generated_text = self.tokenizer.decode(
                    generated_ids[0][input_length:], 
                    skip_special_tokens=True
                )
                
                # Calculate metrics
                output_tokens = generated_ids.shape[1] - input_length
                total_tokens = generated_ids.shape[1]
                inference_time = time.time() - start_time
                
                # Update metrics
                await self._update_metrics(
                    tokens_generated=output_tokens,
                    inference_time_ms=inference_time * 1000
                )
                
                return {
                    "generated_text": generated_text,
                    "input_tokens": input_length,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                    "model_info": self.model_info
                }
                
        except Exception as e:
            await self._update_metrics(error=True)
            logger.error(f"Generation failed: {e}")
            raise
    
    async def generate_stream(self,
                             prompt: str,
                             max_new_tokens: int = 100,
                             temperature: float = 1.0,
                             top_k: int = 50,
                             top_p: float = 0.9,
                             do_sample: bool = True,
                             repetition_penalty: float = 1.0,
                             pad_token_id: Optional[int] = None,
                             eos_token_id: Optional[int] = None) -> AsyncIterator[Dict[str, Any]]:
        
        try:
            with self.inference_lock:
                # Tokenize input
                input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
                input_length = input_ids.shape[1]
                
                if input_length > self.max_sequence_length:
                    raise ValueError(f"Input length {input_length} exceeds maximum {self.max_sequence_length}")
                
                generated_ids = input_ids.clone()
                tokens_generated = 0
                
                pad_token_id = pad_token_id or self.tokenizer.pad_token_id
                eos_token_id = eos_token_id or self.tokenizer.eos_token_id
                
                for step in range(max_new_tokens):
                    # Get next token
                    next_token_id = await self._generate_next_token(
                        generated_ids,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        do_sample=do_sample,
                        repetition_penalty=repetition_penalty
                    )
                    
                    # Append token
                    generated_ids = torch.cat([generated_ids, next_token_id.unsqueeze(-1)], dim=-1)
                    tokens_generated += 1
                    
                    # Decode token
                    token_text = self.tokenizer.decode([next_token_id.item()], skip_special_tokens=True)
                    
                    # Check for end of sequence
                    is_final = (next_token_id.item() == eos_token_id or 
                               tokens_generated >= max_new_tokens or
                               generated_ids.shape[1] >= self.max_sequence_length)
                    
                    yield {
                        "token": token_text,
                        "is_final": is_final,
                        "total_tokens": generated_ids.shape[1]
                    }
                    
                    if is_final:
                        break
                    
                    # Small delay to prevent overwhelming the client
                    await asyncio.sleep(0.001)
                
                # Update metrics
                await self._update_metrics(tokens_generated=tokens_generated)
                
        except Exception as e:
            await self._update_metrics(error=True)
            logger.error(f"Stream generation failed: {e}")
            raise
    
    async def _generate_tokens(self,
                              input_ids: torch.Tensor,
                              max_new_tokens: int,
                              temperature: float,
                              top_k: int,
                              top_p: float,
                              do_sample: bool,
                              repetition_penalty: float,
                              pad_token_id: int,
                              eos_token_id: int) -> torch.Tensor:
        
        generated_ids = input_ids.clone()
        
        for _ in range(max_new_tokens):
            # Get next token
            next_token_id = await self._generate_next_token(
                generated_ids,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty
            )
            
            # Append token
            generated_ids = torch.cat([generated_ids, next_token_id.unsqueeze(-1)], dim=-1)
            
            # Check for end conditions
            if (next_token_id.item() == eos_token_id or 
                generated_ids.shape[1] >= self.max_sequence_length):
                break
        
        return generated_ids
    
    async def _generate_next_token(self,
                                  input_ids: torch.Tensor,
                                  temperature: float,
                                  top_k: int,
                                  top_p: float,
                                  do_sample: bool,
                                  repetition_penalty: float) -> torch.Tensor:
        
        # Prepare input for TensorRT engine
        input_array = input_ids.cpu().numpy().astype(np.int32)
        
        # Run inference
        outputs = self.engine_utils.infer({"input_ids": input_array})
        logits = torch.from_numpy(outputs["logits"])
        
        # Get logits for the last token
        next_token_logits = logits[0, -1, :]
        
        # Apply repetition penalty
        if repetition_penalty != 1.0:
            next_token_logits = self._apply_repetition_penalty(
                next_token_logits, input_ids, repetition_penalty
            )
        
        # Apply sampling
        if do_sample:
            # Apply top-k filtering
            if top_k > 0:
                next_token_logits = ModelUtils.apply_top_k_filtering(next_token_logits, top_k)
            
            # Apply top-p filtering
            if top_p < 1.0:
                next_token_logits = ModelUtils.apply_top_p_filtering(next_token_logits, top_p)
            
            # Apply temperature and sample
            next_token_id = ModelUtils.apply_temperature_sampling(next_token_logits, temperature)
        else:
            # Greedy decoding
            next_token_id = torch.argmax(next_token_logits, dim=-1)
        
        return next_token_id
    
    def _apply_repetition_penalty(self, 
                                 logits: torch.Tensor, 
                                 input_ids: torch.Tensor, 
                                 penalty: float) -> torch.Tensor:
        # Apply repetition penalty to tokens that appear in the input
        for token_id in torch.unique(input_ids):
            if logits[token_id] < 0:
                logits[token_id] *= penalty
            else:
                logits[token_id] /= penalty
        
        return logits
    
    async def benchmark_inference(self,
                                 prompt: str,
                                 batch_size: int = 1,
                                 num_runs: int = 100) -> Dict[str, Any]:
        
        try:
            # Prepare batch input
            prompts = [prompt] * batch_size
            input_ids_list = []
            
            for p in prompts:
                input_ids = self.tokenizer.encode(p, return_tensors="pt")
                input_ids_list.append(input_ids)
            
            # Pad to same length
            max_length = max(ids.shape[1] for ids in input_ids_list)
            batched_input = torch.zeros((batch_size, max_length), dtype=torch.long)
            
            for i, input_ids in enumerate(input_ids_list):
                seq_len = input_ids.shape[1]
                batched_input[i, :seq_len] = input_ids[0]
            
            # Warmup runs
            warmup_runs = min(10, num_runs // 10)
            for _ in range(warmup_runs):
                input_array = batched_input.cpu().numpy().astype(np.int32)
                _ = self.engine_utils.infer({"input_ids": input_array})
            
            # Benchmark runs
            start_time = time.time()
            
            for _ in range(num_runs):
                input_array = batched_input.cpu().numpy().astype(np.int32)
                _ = self.engine_utils.infer({"input_ids": input_array})
            
            end_time = time.time()
            
            # Calculate metrics
            total_time = end_time - start_time
            avg_time = total_time / num_runs
            throughput = (batch_size * num_runs) / total_time
            
            # Estimate tokens per second (assuming average of 20 tokens per inference)
            estimated_tokens_per_inference = 20
            tokens_per_second = throughput * estimated_tokens_per_inference
            
            return {
                "avg_inference_time_ms": avg_time * 1000,
                "total_time_sec": total_time,
                "throughput_inferences_per_sec": throughput,
                "throughput_tokens_per_sec": tokens_per_second,
                "batch_size": batch_size,
                "num_runs": num_runs
            }
            
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            raise
    
    async def _update_metrics(self, 
                             tokens_generated: int = 0, 
                             inference_time_ms: float = 0,
                             error: bool = False):
        
        with self.metrics_lock:
            self.metrics["total_requests"] += 1
            
            if error:
                self.metrics["error_count"] += 1
            else:
                self.metrics["total_tokens_generated"] += tokens_generated
                self.metrics["total_inference_time_ms"] += inference_time_ms
                
                # Calculate running averages
                if self.metrics["total_requests"] > 0:
                    self.metrics["average_latency_ms"] = (
                        self.metrics["total_inference_time_ms"] / 
                        (self.metrics["total_requests"] - self.metrics["error_count"])
                    )
                
                # Calculate rates
                time_since_reset = time.time() - self.metrics["last_reset_time"]
                if time_since_reset > 0:
                    self.metrics["requests_per_second"] = self.metrics["total_requests"] / time_since_reset
                    self.metrics["tokens_per_second"] = self.metrics["total_tokens_generated"] / time_since_reset
    
    async def get_metrics(self) -> Dict[str, Any]:
        with self.metrics_lock:
            return self.metrics.copy()
    
    async def reset_metrics(self):
        with self.metrics_lock:
            self.metrics = {
                "total_requests": 0,
                "total_tokens_generated": 0,
                "total_inference_time_ms": 0,
                "average_latency_ms": 0,
                "requests_per_second": 0,
                "tokens_per_second": 0,
                "error_count": 0,
                "last_reset_time": time.time()
            }
    
    async def get_model_info(self) -> Dict[str, Any]:
        return self.model_info.copy()
    
    async def get_engine_info(self) -> Dict[str, Any]:
        if self.engine_utils:
            return self.engine_utils.get_engine_info()
        return {}
    
    async def get_optimization_info(self) -> Dict[str, Any]:
        return {
            "kv_cache_enabled": True,
            "flash_attention_enabled": True,
            "max_batch_size": self.max_batch_size,
            "max_sequence_length": self.max_sequence_length
        }
    
    async def get_system_info(self) -> Dict[str, Any]:
        import psutil
        import GPUtil
        
        try:
            # CPU info
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # GPU info
            gpus = GPUtil.getGPUs()
            gpu_info = []
            for gpu in gpus:
                gpu_info.append({
                    "id": gpu.id,
                    "name": gpu.name,
                    "memory_used_mb": gpu.memoryUsed,
                    "memory_total_mb": gpu.memoryTotal,
                    "memory_percent": gpu.memoryUtil * 100,
                    "gpu_percent": gpu.load * 100,
                    "temperature": gpu.temperature
                })
            
            return {
                "cpu_percent": cpu_percent,
                "memory_used_gb": memory.used / (1024**3),
                "memory_total_gb": memory.total / (1024**3),
                "memory_percent": memory.percent,
                "gpus": gpu_info
            }
            
        except Exception as e:
            logger.warning(f"Failed to get system info: {e}")
            return {"error": str(e)}
    
    def is_loaded(self) -> bool:
        return self.engine_utils is not None and self.tokenizer is not None
    
    async def reload(self):
        logger.info("Reloading inference engine...")
        
        try:
            # Cleanup current resources
            await self.cleanup()
            
            # Reinitialize
            await self.initialize()
            
            logger.info("Inference engine reloaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to reload inference engine: {e}")
            raise
    
    async def cleanup(self):
        logger.info("Cleaning up inference engine...")
        
        try:
            if self.engine_utils:
                self.engine_utils.cleanup()
                self.engine_utils = None
            
            self.tokenizer = None
            
            logger.info("Inference engine cleanup completed")
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
    
    def __del__(self):
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.cleanup())
            else:
                asyncio.run(self.cleanup())
        except:
            pass