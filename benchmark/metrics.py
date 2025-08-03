import time
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
from datetime import datetime, timedelta
import psutil
import GPUtil

logger = logging.getLogger(__name__)

@dataclass
class BenchmarkMetrics:
    timestamp: float
    test_type: str
    configuration: Dict[str, Any]
    performance_metrics: Dict[str, float]
    system_metrics: Dict[str, float]
    error_metrics: Dict[str, float]

class MetricsCollector:
    def __init__(self, output_dir: str = "./benchmark/results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_history = []
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def collect_benchmark_metrics(self,
                                 throughput_results: List[Any] = None,
                                 latency_results: List[Any] = None,
                                 system_info: Dict[str, Any] = None) -> BenchmarkMetrics:
        
        timestamp = time.time()
        
        # Collect performance metrics
        performance_metrics = {}
        
        if throughput_results:
            performance_metrics.update(self._extract_throughput_metrics(throughput_results))
        
        if latency_results:
            performance_metrics.update(self._extract_latency_metrics(latency_results))
        
        # Collect system metrics
        system_metrics = self._collect_system_metrics()
        if system_info:
            system_metrics.update(system_info)
        
        # Collect error metrics
        error_metrics = self._collect_error_metrics(throughput_results, latency_results)
        
        # Configuration info
        configuration = {
            "has_throughput_test": throughput_results is not None,
            "has_latency_test": latency_results is not None,
            "num_throughput_tests": len(throughput_results) if throughput_results else 0,
            "num_latency_tests": len(latency_results) if latency_results else 0
        }
        
        metrics = BenchmarkMetrics(
            timestamp=timestamp,
            test_type="combined" if throughput_results and latency_results else 
                     "throughput" if throughput_results else "latency",
            configuration=configuration,
            performance_metrics=performance_metrics,
            system_metrics=system_metrics,
            error_metrics=error_metrics
        )
        
        self.metrics_history.append(metrics)
        return metrics
    
    def _extract_throughput_metrics(self, results: List[Any]) -> Dict[str, float]:
        if not results:
            return {}
        
        throughput_values = [r.throughput_requests_per_sec for r in results]
        tokens_per_sec_values = [r.throughput_tokens_per_sec for r in results]
        latency_values = [r.avg_latency_ms for r in results]
        
        return {
            "max_throughput_requests_per_sec": max(throughput_values),
            "avg_throughput_requests_per_sec": np.mean(throughput_values),
            "max_throughput_tokens_per_sec": max(tokens_per_sec_values),
            "avg_throughput_tokens_per_sec": np.mean(tokens_per_sec_values),
            "min_avg_latency_ms": min(latency_values),
            "avg_latency_ms": np.mean(latency_values),
            "throughput_p95_latency_ms": np.percentile([r.p95_latency_ms for r in results], 95),
            "throughput_p99_latency_ms": np.percentile([r.p99_latency_ms for r in results], 99)
        }
    
    def _extract_latency_metrics(self, results: List[Any]) -> Dict[str, float]:
        if not results:
            return {}
        
        ttft_values = [r.time_to_first_token_ms for r in results]
        tpt_values = [r.time_per_token_ms for r in results]
        tps_values = [r.tokens_per_second for r in results]
        
        return {
            "avg_time_to_first_token_ms": np.mean(ttft_values),
            "p95_time_to_first_token_ms": np.percentile(ttft_values, 95),
            "p99_time_to_first_token_ms": np.percentile(ttft_values, 99),
            "avg_time_per_token_ms": np.mean(tpt_values),
            "p95_time_per_token_ms": np.percentile(tpt_values, 95),
            "p99_time_per_token_ms": np.percentile(tpt_values, 99),
            "max_tokens_per_second": max(tps_values),
            "avg_tokens_per_second": np.mean(tps_values)
        }
    
    def _collect_system_metrics(self) -> Dict[str, float]:
        try:
            # CPU and Memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            metrics = {
                "cpu_utilization_percent": cpu_percent,
                "memory_used_gb": memory.used / (1024**3),
                "memory_total_gb": memory.total / (1024**3),
                "memory_utilization_percent": memory.percent
            }
            
            # GPU metrics
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Use first GPU
                    metrics.update({
                        "gpu_utilization_percent": gpu.load * 100,
                        "gpu_memory_used_mb": gpu.memoryUsed,
                        "gpu_memory_total_mb": gpu.memoryTotal,
                        "gpu_memory_utilization_percent": gpu.memoryUtil * 100,
                        "gpu_temperature_c": gpu.temperature
                    })
            except Exception as e:
                logger.warning(f"Could not collect GPU metrics: {e}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return {}
    
    def _collect_error_metrics(self, 
                              throughput_results: List[Any] = None,
                              latency_results: List[Any] = None) -> Dict[str, float]:
        
        error_rates = []
        
        if throughput_results:
            error_rates.extend([r.error_rate for r in throughput_results])
        
        if latency_results:
            error_rates.extend([r.error_rate for r in latency_results])
        
        if not error_rates:
            return {"avg_error_rate": 0.0, "max_error_rate": 0.0}
        
        return {
            "avg_error_rate": np.mean(error_rates),
            "max_error_rate": max(error_rates),
            "min_error_rate": min(error_rates),
            "total_tests_with_errors": len([r for r in error_rates if r > 0])
        }
    
    def save_metrics(self, filename: Optional[str] = None):
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_metrics_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        # Convert metrics to serializable format
        metrics_data = []
        for metric in self.metrics_history:
            metrics_data.append(asdict(metric))
        
        with open(filepath, 'w') as f:
            json.dump({
                "collection_timestamp": time.time(),
                "total_metrics": len(metrics_data),
                "metrics": metrics_data
            }, f, indent=2)
        
        logger.info(f"Metrics saved to {filepath}")
        return str(filepath)
    
    def load_metrics(self, filepath: str):
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.metrics_history = []
        for metric_data in data.get("metrics", []):
            metric = BenchmarkMetrics(**metric_data)
            self.metrics_history.append(metric)
        
        logger.info(f"Loaded {len(self.metrics_history)} metrics from {filepath}")
    
    def generate_report(self, output_filename: Optional[str] = None) -> str:
        if not self.metrics_history:
            logger.warning("No metrics to report")
            return ""
        
        if not output_filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"benchmark_report_{timestamp}.html"
        
        report_path = self.output_dir / output_filename
        
        # Generate comprehensive report
        html_content = self._generate_html_report()
        
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Report generated: {report_path}")
        return str(report_path)
    
    def _generate_html_report(self) -> str:
        # Create visualizations
        self._create_performance_plots()
        self._create_system_plots()
        self._create_comparison_plots()
        
        # Get summary statistics
        summary = self._get_summary_statistics()
        
        # Generate HTML
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>TensorRT-LLM Benchmark Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 30px 0; }}
                .metric {{ display: inline-block; margin: 10px; padding: 15px; background-color: #f9f9f9; border-radius: 5px; }}
                .plot {{ text-align: center; margin: 20px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>TensorRT-LLM Benchmark Report</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Total Benchmark Runs: {len(self.metrics_history)}</p>
            </div>
            
            <div class="section">
                <h2>Performance Summary</h2>
                {self._format_summary_metrics(summary)}
            </div>
            
            <div class="section">
                <h2>Performance Trends</h2>
                <div class="plot">
                    <img src="performance_trends.png" alt="Performance Trends" style="max-width: 100%;">
                </div>
            </div>
            
            <div class="section">
                <h2>System Resource Usage</h2>
                <div class="plot">
                    <img src="system_metrics.png" alt="System Metrics" style="max-width: 100%;">
                </div>
            </div>
            
            <div class="section">
                <h2>Performance Comparison</h2>
                <div class="plot">
                    <img src="performance_comparison.png" alt="Performance Comparison" style="max-width: 100%;">
                </div>
            </div>
            
            <div class="section">
                <h2>Detailed Metrics</h2>
                {self._format_detailed_table()}
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _create_performance_plots(self):
        if not self.metrics_history:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        timestamps = [datetime.fromtimestamp(m.timestamp) for m in self.metrics_history]
        
        # Throughput over time
        throughput_values = [m.performance_metrics.get('max_throughput_tokens_per_sec', 0) 
                           for m in self.metrics_history]
        axes[0, 0].plot(timestamps, throughput_values, marker='o')
        axes[0, 0].set_title('Max Throughput Over Time')
        axes[0, 0].set_ylabel('Tokens/sec')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Latency over time
        latency_values = [m.performance_metrics.get('avg_time_to_first_token_ms', 0) 
                         for m in self.metrics_history]
        axes[0, 1].plot(timestamps, latency_values, marker='o', color='red')
        axes[0, 1].set_title('Average TTFT Over Time')
        axes[0, 1].set_ylabel('Milliseconds')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Error rates over time
        error_rates = [m.error_metrics.get('avg_error_rate', 0) * 100 
                      for m in self.metrics_history]
        axes[1, 0].plot(timestamps, error_rates, marker='o', color='orange')
        axes[1, 0].set_title('Error Rate Over Time')
        axes[1, 0].set_ylabel('Error Rate (%)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Tokens per second distribution
        all_tps = [m.performance_metrics.get('avg_tokens_per_second', 0) 
                  for m in self.metrics_history if m.performance_metrics.get('avg_tokens_per_second', 0) > 0]
        if all_tps:
            axes[1, 1].hist(all_tps, bins=10, alpha=0.7, color='green')
        axes[1, 1].set_title('Tokens/sec Distribution')
        axes[1, 1].set_xlabel('Tokens/sec')
        axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_trends.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_system_plots(self):
        if not self.metrics_history:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # CPU utilization
        cpu_values = [m.system_metrics.get('cpu_utilization_percent', 0) 
                     for m in self.metrics_history]
        axes[0, 0].plot(cpu_values, marker='o')
        axes[0, 0].set_title('CPU Utilization')
        axes[0, 0].set_ylabel('CPU %')
        
        # Memory utilization
        memory_values = [m.system_metrics.get('memory_utilization_percent', 0) 
                        for m in self.metrics_history]
        axes[0, 1].plot(memory_values, marker='o', color='red')
        axes[0, 1].set_title('Memory Utilization')
        axes[0, 1].set_ylabel('Memory %')
        
        # GPU utilization
        gpu_values = [m.system_metrics.get('gpu_utilization_percent', 0) 
                     for m in self.metrics_history]
        axes[1, 0].plot(gpu_values, marker='o', color='green')
        axes[1, 0].set_title('GPU Utilization')
        axes[1, 0].set_ylabel('GPU %')
        
        # GPU memory utilization
        gpu_mem_values = [m.system_metrics.get('gpu_memory_utilization_percent', 0) 
                         for m in self.metrics_history]
        axes[1, 1].plot(gpu_mem_values, marker='o', color='orange')
        axes[1, 1].set_title('GPU Memory Utilization')
        axes[1, 1].set_ylabel('GPU Memory %')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'system_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_comparison_plots(self):
        if len(self.metrics_history) < 2:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Performance vs System Usage
        throughput_values = [m.performance_metrics.get('max_throughput_tokens_per_sec', 0) 
                           for m in self.metrics_history]
        gpu_values = [m.system_metrics.get('gpu_utilization_percent', 0) 
                     for m in self.metrics_history]
        
        axes[0].scatter(gpu_values, throughput_values, alpha=0.7)
        axes[0].set_xlabel('GPU Utilization %')
        axes[0].set_ylabel('Throughput (tokens/sec)')
        axes[0].set_title('Throughput vs GPU Utilization')
        
        # Latency vs Error Rate
        latency_values = [m.performance_metrics.get('avg_time_to_first_token_ms', 0) 
                         for m in self.metrics_history]
        error_values = [m.error_metrics.get('avg_error_rate', 0) * 100 
                       for m in self.metrics_history]
        
        axes[1].scatter(error_values, latency_values, alpha=0.7, color='red')
        axes[1].set_xlabel('Error Rate %')
        axes[1].set_ylabel('TTFT (ms)')
        axes[1].set_title('Latency vs Error Rate')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _get_summary_statistics(self) -> Dict[str, Any]:
        if not self.metrics_history:
            return {}
        
        # Aggregate all performance metrics
        all_throughput = [m.performance_metrics.get('max_throughput_tokens_per_sec', 0) 
                         for m in self.metrics_history if m.performance_metrics.get('max_throughput_tokens_per_sec', 0) > 0]
        all_latency = [m.performance_metrics.get('avg_time_to_first_token_ms', 0) 
                      for m in self.metrics_history if m.performance_metrics.get('avg_time_to_first_token_ms', 0) > 0]
        all_errors = [m.error_metrics.get('avg_error_rate', 0) 
                     for m in self.metrics_history]
        
        summary = {
            "total_benchmarks": len(self.metrics_history),
            "date_range": {
                "start": datetime.fromtimestamp(min(m.timestamp for m in self.metrics_history)).isoformat(),
                "end": datetime.fromtimestamp(max(m.timestamp for m in self.metrics_history)).isoformat()
            }
        }
        
        if all_throughput:
            summary["throughput"] = {
                "max_tokens_per_sec": max(all_throughput),
                "avg_tokens_per_sec": np.mean(all_throughput),
                "min_tokens_per_sec": min(all_throughput),
                "std_tokens_per_sec": np.std(all_throughput)
            }
        
        if all_latency:
            summary["latency"] = {
                "min_ttft_ms": min(all_latency),
                "avg_ttft_ms": np.mean(all_latency),
                "max_ttft_ms": max(all_latency),
                "std_ttft_ms": np.std(all_latency)
            }
        
        if all_errors:
            summary["reliability"] = {
                "avg_error_rate": np.mean(all_errors),
                "max_error_rate": max(all_errors),
                "benchmarks_with_errors": len([e for e in all_errors if e > 0])
            }
        
        return summary
    
    def _format_summary_metrics(self, summary: Dict[str, Any]) -> str:
        html = "<div style='display: flex; flex-wrap: wrap;'>"
        
        for category, metrics in summary.items():
            if isinstance(metrics, dict):
                html += f"<div class='metric'><h3>{category.title()}</h3>"
                for key, value in metrics.items():
                    if isinstance(value, float):
                        html += f"<p><strong>{key}:</strong> {value:.2f}</p>"
                    else:
                        html += f"<p><strong>{key}:</strong> {value}</p>"
                html += "</div>"
        
        html += "</div>"
        return html
    
    def _format_detailed_table(self) -> str:
        html = "<table><tr>"
        html += "<th>Timestamp</th><th>Test Type</th><th>Max Throughput</th><th>Avg Latency</th><th>Error Rate</th><th>GPU Usage</th>"
        html += "</tr>"
        
        for metric in self.metrics_history:
            timestamp = datetime.fromtimestamp(metric.timestamp).strftime('%Y-%m-%d %H:%M:%S')
            throughput = metric.performance_metrics.get('max_throughput_tokens_per_sec', 0)
            latency = metric.performance_metrics.get('avg_time_to_first_token_ms', 0)
            error_rate = metric.error_metrics.get('avg_error_rate', 0) * 100
            gpu_usage = metric.system_metrics.get('gpu_utilization_percent', 0)
            
            html += f"<tr>"
            html += f"<td>{timestamp}</td>"
            html += f"<td>{metric.test_type}</td>"
            html += f"<td>{throughput:.1f} tokens/sec</td>"
            html += f"<td>{latency:.1f} ms</td>"
            html += f"<td>{error_rate:.1f}%</td>"
            html += f"<td>{gpu_usage:.1f}%</td>"
            html += f"</tr>"
        
        html += "</table>"
        return html
    
    def compare_benchmarks(self, 
                          baseline_metrics: 'BenchmarkMetrics',
                          current_metrics: 'BenchmarkMetrics') -> Dict[str, Any]:
        
        comparison = {
            "timestamp_comparison": {
                "baseline": datetime.fromtimestamp(baseline_metrics.timestamp).isoformat(),
                "current": datetime.fromtimestamp(current_metrics.timestamp).isoformat()
            },
            "performance_comparison": {},
            "system_comparison": {},
            "error_comparison": {}
        }
        
        # Compare performance metrics
        for key in set(baseline_metrics.performance_metrics.keys()) | set(current_metrics.performance_metrics.keys()):
            baseline_val = baseline_metrics.performance_metrics.get(key, 0)
            current_val = current_metrics.performance_metrics.get(key, 0)
            
            if baseline_val > 0:
                change_percent = ((current_val - baseline_val) / baseline_val) * 100
                comparison["performance_comparison"][key] = {
                    "baseline": baseline_val,
                    "current": current_val,
                    "change_percent": change_percent,
                    "improved": change_percent > 0 if "latency" not in key.lower() else change_percent < 0
                }
        
        return comparison