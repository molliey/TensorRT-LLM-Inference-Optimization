import asyncio
import time
import numpy as np
import requests
import json
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import statistics
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

@dataclass
class LatencyResult:
    prompt_length: int
    max_new_tokens: int
    num_samples: int
    time_to_first_token_ms: float
    time_per_token_ms: float
    total_generation_time_ms: float
    tokens_generated: int
    tokens_per_second: float
    percentiles: Dict[str, float]
    std_deviation_ms: float
    error_rate: float

class LatencyBenchmark:
    def __init__(self, server_url: str = "http://localhost:8000"):
        self.server_url = server_url.rstrip('/')
        self.results = []
    
    async def run_latency_test(self,
                              test_prompts: List[str],
                              token_lengths: List[int] = [50, 100, 200],
                              num_samples: int = 50,
                              temperature: float = 1.0,
                              measure_streaming: bool = True) -> List[LatencyResult]:
        
        logger.info("Starting latency benchmark...")
        
        for max_tokens in token_lengths:
            for prompt in test_prompts:
                logger.info(f"Testing prompt length={len(prompt)}, max_tokens={max_tokens}")
                
                if measure_streaming:
                    result = await self._test_streaming_latency(
                        prompt=prompt,
                        max_new_tokens=max_tokens,
                        num_samples=num_samples,
                        temperature=temperature
                    )
                else:
                    result = await self._test_batch_latency(
                        prompt=prompt,
                        max_new_tokens=max_tokens,
                        num_samples=num_samples,
                        temperature=temperature
                    )
                
                self.results.append(result)
                logger.info(f"TTFT: {result.time_to_first_token_ms:.2f}ms, "
                           f"TPT: {result.time_per_token_ms:.2f}ms, "
                           f"TPS: {result.tokens_per_second:.2f}")
        
        return self.results
    
    async def _test_streaming_latency(self,
                                    prompt: str,
                                    max_new_tokens: int,
                                    num_samples: int,
                                    temperature: float) -> LatencyResult:
        
        ttft_times = []  # Time to first token
        tpt_times = []   # Time per token
        total_times = []
        tokens_generated_list = []
        errors = 0
        
        for i in range(num_samples):
            try:
                result = await self._single_streaming_request(
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature
                )
                
                if result:
                    ttft_times.append(result['ttft'])
                    tpt_times.extend(result['tpt_times'])
                    total_times.append(result['total_time'])
                    tokens_generated_list.append(result['tokens_generated'])
                else:
                    errors += 1
                    
            except Exception as e:
                logger.error(f"Streaming request {i+1} failed: {e}")
                errors += 1
        
        if not ttft_times:
            return self._empty_result(len(prompt), max_new_tokens, num_samples, 1.0)
        
        # Calculate metrics
        avg_ttft = np.mean(ttft_times)
        avg_tpt = np.mean(tpt_times) if tpt_times else 0
        avg_total_time = np.mean(total_times)
        avg_tokens_generated = np.mean(tokens_generated_list)
        tokens_per_second = avg_tokens_generated / (avg_total_time / 1000) if avg_total_time > 0 else 0
        
        # Calculate percentiles
        percentiles = {
            'p50': np.percentile(ttft_times, 50),
            'p90': np.percentile(ttft_times, 90),
            'p95': np.percentile(ttft_times, 95),
            'p99': np.percentile(ttft_times, 99)
        }
        
        return LatencyResult(
            prompt_length=len(prompt),
            max_new_tokens=max_new_tokens,
            num_samples=num_samples,
            time_to_first_token_ms=avg_ttft,
            time_per_token_ms=avg_tpt,
            total_generation_time_ms=avg_total_time,
            tokens_generated=int(avg_tokens_generated),
            tokens_per_second=tokens_per_second,
            percentiles=percentiles,
            std_deviation_ms=np.std(ttft_times),
            error_rate=errors / num_samples
        )
    
    async def _single_streaming_request(self,
                                      prompt: str,
                                      max_new_tokens: int,
                                      temperature: float) -> Optional[Dict[str, Any]]:
        
        request_data = {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "stream": True
        }
        
        try:
            start_time = time.time()
            first_token_time = None
            token_times = []
            last_token_time = start_time
            tokens_generated = 0
            
            response = requests.post(
                f"{self.server_url}/generate/stream",
                json=request_data,
                stream=True,
                timeout=60
            )
            
            if response.status_code != 200:
                logger.warning(f"Stream request failed with status {response.status_code}")
                return None
            
            for line in response.iter_lines(decode_unicode=True):
                if line.startswith('data: '):
                    current_time = time.time()
                    
                    try:
                        data = json.loads(line[6:])  # Remove 'data: ' prefix
                        
                        if 'error' in data:
                            logger.error(f"Stream error: {data['error']}")
                            return None
                        
                        # Record first token time
                        if first_token_time is None:
                            first_token_time = current_time
                        
                        # Record time between tokens
                        if tokens_generated > 0:
                            token_times.append((current_time - last_token_time) * 1000)
                        
                        tokens_generated += 1
                        last_token_time = current_time
                        
                        if data.get('is_final', False):
                            break
                            
                    except json.JSONDecodeError:
                        continue
            
            end_time = time.time()
            
            if first_token_time is None:
                return None
            
            return {
                'ttft': (first_token_time - start_time) * 1000,  # ms
                'tpt_times': token_times,  # ms per token
                'total_time': (end_time - start_time) * 1000,  # ms
                'tokens_generated': tokens_generated
            }
            
        except Exception as e:
            logger.error(f"Streaming request failed: {e}")
            return None
    
    async def _test_batch_latency(self,
                                prompt: str,
                                max_new_tokens: int,
                                num_samples: int,
                                temperature: float) -> LatencyResult:
        
        latencies = []
        tokens_generated_list = []
        errors = 0
        
        request_data = {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": True
        }
        
        # Run concurrent requests
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for i in range(num_samples):
                future = executor.submit(self._single_batch_request, request_data)
                futures.append(future)
            
            for future in futures:
                try:
                    result = future.result()
                    if result:
                        latencies.append(result['latency'])
                        tokens_generated_list.append(result['tokens_generated'])
                    else:
                        errors += 1
                except Exception as e:
                    logger.error(f"Batch request failed: {e}")
                    errors += 1
        
        if not latencies:
            return self._empty_result(len(prompt), max_new_tokens, num_samples, 1.0)
        
        # Calculate metrics
        avg_latency = np.mean(latencies)
        avg_tokens_generated = np.mean(tokens_generated_list)
        tokens_per_second = avg_tokens_generated / (avg_latency / 1000) if avg_latency > 0 else 0
        
        # For batch requests, we don't have TTFT and TPT separation
        # So we estimate based on total latency
        estimated_ttft = avg_latency * 0.1  # Rough estimate: 10% for first token
        estimated_tpt = (avg_latency - estimated_ttft) / max(1, avg_tokens_generated - 1)
        
        percentiles = {
            'p50': np.percentile(latencies, 50),
            'p90': np.percentile(latencies, 90),
            'p95': np.percentile(latencies, 95),
            'p99': np.percentile(latencies, 99)
        }
        
        return LatencyResult(
            prompt_length=len(prompt),
            max_new_tokens=max_new_tokens,
            num_samples=num_samples,
            time_to_first_token_ms=estimated_ttft,
            time_per_token_ms=estimated_tpt,
            total_generation_time_ms=avg_latency,
            tokens_generated=int(avg_tokens_generated),
            tokens_per_second=tokens_per_second,
            percentiles=percentiles,
            std_deviation_ms=np.std(latencies),
            error_rate=errors / num_samples
        )
    
    def _single_batch_request(self, request_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            start_time = time.time()
            
            response = requests.post(
                f"{self.server_url}/generate",
                json=request_data,
                timeout=60
            )
            
            end_time = time.time()
            latency = (end_time - start_time) * 1000  # ms
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'latency': latency,
                    'tokens_generated': data.get('output_tokens', 0)
                }
            else:
                logger.warning(f"Batch request failed with status {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Batch request exception: {e}")
            return None
    
    def _empty_result(self, 
                     prompt_length: int, 
                     max_new_tokens: int, 
                     num_samples: int, 
                     error_rate: float) -> LatencyResult:
        return LatencyResult(
            prompt_length=prompt_length,
            max_new_tokens=max_new_tokens,
            num_samples=num_samples,
            time_to_first_token_ms=0,
            time_per_token_ms=0,
            total_generation_time_ms=0,
            tokens_generated=0,
            tokens_per_second=0,
            percentiles={'p50': 0, 'p90': 0, 'p95': 0, 'p99': 0},
            std_deviation_ms=0,
            error_rate=error_rate
        )
    
    def save_results(self, output_path: str):
        results_data = []
        for result in self.results:
            results_data.append({
                "prompt_length": result.prompt_length,
                "max_new_tokens": result.max_new_tokens,
                "num_samples": result.num_samples,
                "time_to_first_token_ms": result.time_to_first_token_ms,
                "time_per_token_ms": result.time_per_token_ms,
                "total_generation_time_ms": result.total_generation_time_ms,
                "tokens_generated": result.tokens_generated,
                "tokens_per_second": result.tokens_per_second,
                "percentiles": result.percentiles,
                "std_deviation_ms": result.std_deviation_ms,
                "error_rate": result.error_rate
            })
        
        with open(output_path, 'w') as f:
            json.dump({
                "benchmark_type": "latency",
                "timestamp": time.time(),
                "server_url": self.server_url,
                "results": results_data
            }, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
    
    def plot_results(self, output_dir: str = "./benchmark/results"):
        if not self.results:
            logger.warning("No results to plot")
            return
        
        # Create plots
        self._plot_latency_distribution(output_dir)
        self._plot_latency_vs_length(output_dir)
        self._plot_throughput_comparison(output_dir)
    
    def _plot_latency_distribution(self, output_dir: str):
        plt.figure(figsize=(12, 8))
        
        # TTFT distribution
        plt.subplot(2, 2, 1)
        ttft_values = [r.time_to_first_token_ms for r in self.results]
        plt.hist(ttft_values, bins=20, alpha=0.7, color='blue')
        plt.xlabel('Time to First Token (ms)')
        plt.ylabel('Frequency')
        plt.title('TTFT Distribution')
        
        # TPT distribution
        plt.subplot(2, 2, 2)
        tpt_values = [r.time_per_token_ms for r in self.results]
        plt.hist(tpt_values, bins=20, alpha=0.7, color='green')
        plt.xlabel('Time per Token (ms)')
        plt.ylabel('Frequency')
        plt.title('TPT Distribution')
        
        # Total time distribution
        plt.subplot(2, 2, 3)
        total_values = [r.total_generation_time_ms for r in self.results]
        plt.hist(total_values, bins=20, alpha=0.7, color='red')
        plt.xlabel('Total Generation Time (ms)')
        plt.ylabel('Frequency')
        plt.title('Total Time Distribution')
        
        # Tokens per second distribution
        plt.subplot(2, 2, 4)
        tps_values = [r.tokens_per_second for r in self.results]
        plt.hist(tps_values, bins=20, alpha=0.7, color='orange')
        plt.xlabel('Tokens per Second')
        plt.ylabel('Frequency')
        plt.title('Throughput Distribution')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/latency_distributions.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_latency_vs_length(self, output_dir: str):
        plt.figure(figsize=(12, 6))
        
        # Group by max_new_tokens
        token_groups = {}
        for result in self.results:
            if result.max_new_tokens not in token_groups:
                token_groups[result.max_new_tokens] = []
            token_groups[result.max_new_tokens].append(result)
        
        # TTFT vs prompt length
        plt.subplot(1, 2, 1)
        for max_tokens, results in token_groups.items():
            prompt_lengths = [r.prompt_length for r in results]
            ttft_values = [r.time_to_first_token_ms for r in results]
            plt.scatter(prompt_lengths, ttft_values, label=f'{max_tokens} tokens', alpha=0.7)
        
        plt.xlabel('Prompt Length (characters)')
        plt.ylabel('Time to First Token (ms)')
        plt.title('TTFT vs Prompt Length')
        plt.legend()
        
        # TPT vs max tokens
        plt.subplot(1, 2, 2)
        for max_tokens, results in token_groups.items():
            max_tokens_list = [r.max_new_tokens for r in results]
            tpt_values = [r.time_per_token_ms for r in results]
            plt.scatter(max_tokens_list, tpt_values, label=f'{max_tokens} tokens', alpha=0.7)
        
        plt.xlabel('Max New Tokens')
        plt.ylabel('Time per Token (ms)')
        plt.title('TPT vs Max Tokens')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/latency_vs_length.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_throughput_comparison(self, output_dir: str):
        plt.figure(figsize=(10, 6))
        
        # Tokens per second comparison
        max_tokens = [r.max_new_tokens for r in self.results]
        tps_values = [r.tokens_per_second for r in self.results]
        
        plt.scatter(max_tokens, tps_values, alpha=0.7, s=60)
        
        # Add trend line
        z = np.polyfit(max_tokens, tps_values, 1)
        p = np.poly1d(z)
        plt.plot(max_tokens, p(max_tokens), "r--", alpha=0.8)
        
        plt.xlabel('Max New Tokens')
        plt.ylabel('Tokens per Second')
        plt.title('Throughput vs Generation Length')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/throughput_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def get_summary(self) -> Dict[str, Any]:
        if not self.results:
            return {"error": "No results available"}
        
        # Calculate summary statistics
        ttft_values = [r.time_to_first_token_ms for r in self.results]
        tpt_values = [r.time_per_token_ms for r in self.results]
        tps_values = [r.tokens_per_second for r in self.results]
        error_rates = [r.error_rate for r in self.results]
        
        return {
            "total_tests": len(self.results),
            "time_to_first_token": {
                "mean_ms": np.mean(ttft_values),
                "median_ms": np.median(ttft_values),
                "p95_ms": np.percentile(ttft_values, 95),
                "p99_ms": np.percentile(ttft_values, 99),
                "std_ms": np.std(ttft_values)
            },
            "time_per_token": {
                "mean_ms": np.mean(tpt_values),
                "median_ms": np.median(tpt_values),
                "p95_ms": np.percentile(tpt_values, 95),
                "p99_ms": np.percentile(tpt_values, 99),
                "std_ms": np.std(tpt_values)
            },
            "throughput": {
                "mean_tokens_per_sec": np.mean(tps_values),
                "median_tokens_per_sec": np.median(tps_values),
                "max_tokens_per_sec": np.max(tps_values),
                "min_tokens_per_sec": np.min(tps_values)
            },
            "reliability": {
                "avg_error_rate": np.mean(error_rates),
                "max_error_rate": np.max(error_rates),
                "successful_tests": len([r for r in self.results if r.error_rate < 0.1])
            }
        }