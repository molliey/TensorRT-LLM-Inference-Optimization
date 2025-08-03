import asyncio
import time
import numpy as np
import psutil
import GPUtil
from typing import Dict, List, Any, Optional, Tuple
import logging
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import requests
import threading
from queue import Queue

logger = logging.getLogger(__name__)

@dataclass
class ThroughputResult:
    batch_size: int
    sequence_length: int
    num_requests: int
    total_time_sec: float
    avg_latency_ms: float
    throughput_requests_per_sec: float
    throughput_tokens_per_sec: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    error_rate: float
    gpu_utilization_avg: float
    gpu_memory_usage_mb: float
    cpu_utilization_avg: float
    memory_usage_mb: float

class ThroughputBenchmark:
    def __init__(self, 
                 server_url: str = "http://localhost:8000",
                 max_workers: int = 50):
        self.server_url = server_url.rstrip('/')
        self.max_workers = max_workers
        self.results = []
        self.system_monitor = SystemMonitor()
    
    async def run_throughput_test(self,
                                 prompts: List[str],
                                 batch_sizes: List[int] = [1, 2, 4, 8],
                                 sequence_lengths: List[int] = [128, 256, 512],
                                 num_requests: int = 100,
                                 max_new_tokens: int = 50,
                                 concurrent_requests: int = 10) -> List[ThroughputResult]:
        
        logger.info("Starting throughput benchmark...")
        
        for batch_size in batch_sizes:
            for seq_length in sequence_lengths:
                logger.info(f"Testing batch_size={batch_size}, seq_length={seq_length}")
                
                # Generate test prompts
                test_prompts = self._generate_test_prompts(prompts, seq_length, num_requests)
                
                # Run test
                result = await self._run_single_test(
                    test_prompts=test_prompts,
                    batch_size=batch_size,
                    sequence_length=seq_length,
                    max_new_tokens=max_new_tokens,
                    concurrent_requests=concurrent_requests
                )
                
                self.results.append(result)
                logger.info(f"Result: {result.throughput_requests_per_sec:.2f} req/s, "
                           f"{result.throughput_tokens_per_sec:.2f} tokens/s")
        
        return self.results
    
    async def _run_single_test(self,
                              test_prompts: List[str],
                              batch_size: int, 
                              sequence_length: int,
                              max_new_tokens: int,
                              concurrent_requests: int) -> ThroughputResult:
        
        # Start system monitoring
        self.system_monitor.start_monitoring()
        
        # Prepare requests
        requests_data = []
        for prompt in test_prompts:
            requests_data.append({
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
                "temperature": 1.0,
                "top_k": 50,
                "top_p": 0.9,
                "do_sample": True
            })
        
        # Execute concurrent requests
        latencies = []
        errors = 0
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
            # Submit all requests
            future_to_request = {
                executor.submit(self._send_request, req_data): i 
                for i, req_data in enumerate(requests_data)
            }
            
            # Collect results
            for future in as_completed(future_to_request):
                try:
                    latency, success = future.result()
                    if success:
                        latencies.append(latency)
                    else:
                        errors += 1
                except Exception as e:
                    logger.error(f"Request failed: {e}")
                    errors += 1
        
        end_time = time.time()
        
        # Stop system monitoring
        system_stats = self.system_monitor.stop_monitoring()
        
        # Calculate metrics
        total_time = end_time - start_time
        num_successful = len(latencies)
        
        if num_successful > 0:
            avg_latency = np.mean(latencies)
            p50_latency = np.percentile(latencies, 50)
            p95_latency = np.percentile(latencies, 95)
            p99_latency = np.percentile(latencies, 99)
            throughput_requests = num_successful / total_time
            
            # Estimate tokens per second (assuming average tokens per request)
            avg_tokens_per_request = max_new_tokens * 0.8  # Rough estimate
            throughput_tokens = throughput_requests * avg_tokens_per_request
        else:
            avg_latency = p50_latency = p95_latency = p99_latency = 0
            throughput_requests = throughput_tokens = 0
        
        return ThroughputResult(
            batch_size=batch_size,
            sequence_length=sequence_length,
            num_requests=len(test_prompts),
            total_time_sec=total_time,
            avg_latency_ms=avg_latency * 1000,
            throughput_requests_per_sec=throughput_requests,
            throughput_tokens_per_sec=throughput_tokens,
            p50_latency_ms=p50_latency * 1000,
            p95_latency_ms=p95_latency * 1000,
            p99_latency_ms=p99_latency * 1000,
            error_rate=errors / len(test_prompts),
            gpu_utilization_avg=system_stats.get('gpu_utilization_avg', 0),
            gpu_memory_usage_mb=system_stats.get('gpu_memory_usage_mb', 0),
            cpu_utilization_avg=system_stats.get('cpu_utilization_avg', 0),
            memory_usage_mb=system_stats.get('memory_usage_mb', 0)
        )
    
    def _send_request(self, request_data: Dict[str, Any]) -> Tuple[float, bool]:
        try:
            start_time = time.time()
            
            response = requests.post(
                f"{self.server_url}/generate",
                json=request_data,
                timeout=60
            )
            
            end_time = time.time()
            latency = end_time - start_time
            
            if response.status_code == 200:
                return latency, True
            else:
                logger.warning(f"Request failed with status {response.status_code}")
                return latency, False
                
        except Exception as e:
            logger.error(f"Request exception: {e}")
            return 0, False
    
    def _generate_test_prompts(self, 
                              base_prompts: List[str], 
                              target_length: int, 
                              num_requests: int) -> List[str]:
        test_prompts = []
        
        for i in range(num_requests):
            base_prompt = base_prompts[i % len(base_prompts)]
            
            # Adjust prompt length to approximate target sequence length
            words = base_prompt.split()
            target_words = max(1, target_length // 4)  # Rough estimate: 4 chars per word
            
            if len(words) > target_words:
                prompt = ' '.join(words[:target_words])
            else:
                # Repeat words to reach target length
                multiplier = (target_words // len(words)) + 1
                extended_words = (words * multiplier)[:target_words]
                prompt = ' '.join(extended_words)
            
            test_prompts.append(prompt)
        
        return test_prompts
    
    def save_results(self, output_path: str):
        results_data = []
        for result in self.results:
            results_data.append({
                "batch_size": result.batch_size,
                "sequence_length": result.sequence_length,
                "num_requests": result.num_requests,
                "total_time_sec": result.total_time_sec,
                "avg_latency_ms": result.avg_latency_ms,
                "throughput_requests_per_sec": result.throughput_requests_per_sec,
                "throughput_tokens_per_sec": result.throughput_tokens_per_sec,
                "p50_latency_ms": result.p50_latency_ms,
                "p95_latency_ms": result.p95_latency_ms,
                "p99_latency_ms": result.p99_latency_ms,
                "error_rate": result.error_rate,
                "gpu_utilization_avg": result.gpu_utilization_avg,
                "gpu_memory_usage_mb": result.gpu_memory_usage_mb,
                "cpu_utilization_avg": result.cpu_utilization_avg,
                "memory_usage_mb": result.memory_usage_mb
            })
        
        with open(output_path, 'w') as f:
            json.dump({
                "benchmark_type": "throughput",
                "timestamp": time.time(),
                "server_url": self.server_url,
                "results": results_data
            }, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
    
    def get_summary(self) -> Dict[str, Any]:
        if not self.results:
            return {"error": "No results available"}
        
        # Find best performing configurations
        best_throughput = max(self.results, key=lambda x: x.throughput_requests_per_sec)
        best_tokens_per_sec = max(self.results, key=lambda x: x.throughput_tokens_per_sec)
        lowest_latency = min(self.results, key=lambda x: x.avg_latency_ms)
        
        return {
            "total_tests": len(self.results),
            "best_throughput": {
                "requests_per_sec": best_throughput.throughput_requests_per_sec,
                "batch_size": best_throughput.batch_size,
                "sequence_length": best_throughput.sequence_length
            },
            "best_token_throughput": {
                "tokens_per_sec": best_tokens_per_sec.throughput_tokens_per_sec,
                "batch_size": best_tokens_per_sec.batch_size,
                "sequence_length": best_tokens_per_sec.sequence_length
            },
            "lowest_latency": {
                "avg_latency_ms": lowest_latency.avg_latency_ms,
                "batch_size": lowest_latency.batch_size,
                "sequence_length": lowest_latency.sequence_length
            },
            "average_metrics": {
                "avg_throughput_requests_per_sec": np.mean([r.throughput_requests_per_sec for r in self.results]),
                "avg_throughput_tokens_per_sec": np.mean([r.throughput_tokens_per_sec for r in self.results]),
                "avg_latency_ms": np.mean([r.avg_latency_ms for r in self.results]),
                "avg_error_rate": np.mean([r.error_rate for r in self.results])
            }
        }

class SystemMonitor:
    def __init__(self):
        self.monitoring = False
        self.stats = {
            'cpu_utilization': [],
            'memory_usage_mb': [],
            'gpu_utilization': [],
            'gpu_memory_usage_mb': []
        }
        self.monitor_thread = None
    
    def start_monitoring(self):
        self.monitoring = True
        self.stats = {
            'cpu_utilization': [],
            'memory_usage_mb': [],
            'gpu_utilization': [],
            'gpu_memory_usage_mb': []
        }
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> Dict[str, float]:
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        
        # Calculate averages
        result = {}
        if self.stats['cpu_utilization']:
            result['cpu_utilization_avg'] = np.mean(self.stats['cpu_utilization'])
        if self.stats['memory_usage_mb']:
            result['memory_usage_mb'] = np.mean(self.stats['memory_usage_mb'])
        if self.stats['gpu_utilization']:
            result['gpu_utilization_avg'] = np.mean(self.stats['gpu_utilization'])
        if self.stats['gpu_memory_usage_mb']:
            result['gpu_memory_usage_mb'] = np.mean(self.stats['gpu_memory_usage_mb'])
        
        return result
    
    def _monitor_loop(self):
        while self.monitoring:
            try:
                # CPU and Memory
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory()
                
                self.stats['cpu_utilization'].append(cpu_percent)
                self.stats['memory_usage_mb'].append(memory.used / (1024 * 1024))
                
                # GPU
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu = gpus[0]  # Use first GPU
                        self.stats['gpu_utilization'].append(gpu.load * 100)
                        self.stats['gpu_memory_usage_mb'].append(gpu.memoryUsed)
                except:
                    pass
                
                time.sleep(1)  # Monitor every second
                
            except Exception as e:
                logger.warning(f"Monitoring error: {e}")
                time.sleep(1)