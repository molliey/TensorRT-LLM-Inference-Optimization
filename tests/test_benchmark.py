import pytest
import asyncio
import tempfile
import json
import os
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import numpy as np

from benchmark import ThroughputBenchmark, LatencyBenchmark, MetricsCollector


class TestThroughputBenchmark:
    
    @pytest.fixture
    def mock_requests(self):
        with patch('benchmark.throughput_test.requests') as mock:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "generated_text": "Test response",
                "output_tokens": 20
            }
            mock.post.return_value = mock_response
            yield mock
    
    @pytest.fixture
    def benchmark(self):
        return ThroughputBenchmark(server_url="http://localhost:8000")
    
    def test_init(self, benchmark):
        assert benchmark.server_url == "http://localhost:8000"
        assert benchmark.max_workers == 50
        assert benchmark.results == []
        assert benchmark.system_monitor is not None
    
    @pytest.mark.asyncio
    async def test_run_throughput_test(self, benchmark, mock_requests):
        prompts = ["Test prompt 1", "Test prompt 2"]
        batch_sizes = [1, 2]
        sequence_lengths = [128, 256]
        
        with patch.object(benchmark.system_monitor, 'start_monitoring'), \
             patch.object(benchmark.system_monitor, 'stop_monitoring', return_value={}):
            
            results = await benchmark.run_throughput_test(
                prompts=prompts,
                batch_sizes=batch_sizes,
                sequence_lengths=sequence_lengths,
                num_requests=10,
                max_new_tokens=50,
                concurrent_requests=2
            )
            
            assert len(results) == len(batch_sizes) * len(sequence_lengths)
            
            for result in results:
                assert hasattr(result, 'batch_size')
                assert hasattr(result, 'sequence_length')
                assert hasattr(result, 'throughput_requests_per_sec')
                assert hasattr(result, 'throughput_tokens_per_sec')
                assert hasattr(result, 'avg_latency_ms')
                assert result.batch_size in batch_sizes
                assert result.sequence_length in sequence_lengths
    
    def test_send_request_success(self, benchmark, mock_requests):
        request_data = {
            "prompt": "Test prompt",
            "max_new_tokens": 50
        }
        
        latency, success = benchmark._send_request(request_data)
        
        assert isinstance(latency, float)
        assert latency > 0
        assert success is True
        mock_requests.post.assert_called_once()
    
    def test_send_request_failure(self, benchmark):
        with patch('benchmark.throughput_test.requests.post') as mock_post:
            mock_post.side_effect = Exception("Connection error")
            
            request_data = {"prompt": "Test prompt"}
            latency, success = benchmark._send_request(request_data)
            
            assert latency == 0
            assert success is False
    
    def test_generate_test_prompts(self, benchmark):
        base_prompts = ["Short prompt", "This is a longer prompt with more words"]
        
        test_prompts = benchmark._generate_test_prompts(
            base_prompts=base_prompts,
            target_length=100,
            num_requests=5
        )
        
        assert len(test_prompts) == 5
        assert all(isinstance(prompt, str) for prompt in test_prompts)
        assert all(len(prompt) > 0 for prompt in test_prompts)
    
    def test_save_results(self, benchmark):
        # Create mock results
        from benchmark.throughput_test import ThroughputResult
        
        mock_result = ThroughputResult(
            batch_size=1,
            sequence_length=128,
            num_requests=10,
            total_time_sec=1.0,
            avg_latency_ms=100.0,
            throughput_requests_per_sec=10.0,
            throughput_tokens_per_sec=200.0,
            p50_latency_ms=95.0,
            p95_latency_ms=150.0,
            p99_latency_ms=200.0,
            error_rate=0.0,
            gpu_utilization_avg=80.0,
            gpu_memory_usage_mb=1000.0,
            cpu_utilization_avg=50.0,
            memory_usage_mb=2000.0
        )
        
        benchmark.results = [mock_result]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_path = f.name
        
        try:
            benchmark.save_results(output_path)
            
            assert os.path.exists(output_path)
            
            with open(output_path, 'r') as f:
                data = json.load(f)
            
            assert "benchmark_type" in data
            assert data["benchmark_type"] == "throughput"
            assert "results" in data
            assert len(data["results"]) == 1
            
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)
    
    def test_get_summary(self, benchmark):
        # Test with no results
        summary = benchmark.get_summary()
        assert "error" in summary
        
        # Test with results
        from benchmark.throughput_test import ThroughputResult
        
        mock_results = [
            ThroughputResult(1, 128, 10, 1.0, 100.0, 10.0, 200.0, 95.0, 150.0, 200.0, 0.0, 80.0, 1000.0, 50.0, 2000.0),
            ThroughputResult(2, 256, 10, 2.0, 200.0, 5.0, 100.0, 190.0, 250.0, 300.0, 0.1, 85.0, 1200.0, 60.0, 2500.0)
        ]
        
        benchmark.results = mock_results
        summary = benchmark.get_summary()
        
        assert "total_tests" in summary
        assert summary["total_tests"] == 2
        assert "best_throughput" in summary
        assert "best_token_throughput" in summary
        assert "lowest_latency" in summary
        assert "average_metrics" in summary


class TestLatencyBenchmark:
    
    @pytest.fixture
    def mock_requests(self):
        with patch('benchmark.latency_test.requests') as mock:
            # Mock streaming response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.iter_lines.return_value = [
                'data: {"token": "Hello", "is_final": false, "total_tokens": 1}',
                'data: {"token": " world", "is_final": false, "total_tokens": 2}',
                'data: {"token": "!", "is_final": true, "total_tokens": 3}'
            ]
            
            # Mock regular response
            mock_regular_response = Mock()
            mock_regular_response.status_code = 200
            mock_regular_response.json.return_value = {
                "generated_text": "Hello world!",
                "output_tokens": 3
            }
            
            mock.post.return_value = mock_response
            yield mock
    
    @pytest.fixture
    def benchmark(self):
        return LatencyBenchmark(server_url="http://localhost:8000")
    
    def test_init(self, benchmark):
        assert benchmark.server_url == "http://localhost:8000"
        assert benchmark.results == []
    
    @pytest.mark.asyncio
    async def test_run_latency_test_streaming(self, benchmark, mock_requests):
        test_prompts = ["Hello", "How are you?"]
        token_lengths = [10, 20]
        
        results = await benchmark.run_latency_test(
            test_prompts=test_prompts,
            token_lengths=token_lengths,
            num_samples=5,
            measure_streaming=True
        )
        
        assert len(results) == len(test_prompts) * len(token_lengths)
        
        for result in results:
            assert hasattr(result, 'time_to_first_token_ms')
            assert hasattr(result, 'time_per_token_ms')
            assert hasattr(result, 'tokens_per_second')
            assert hasattr(result, 'percentiles')
            assert result.prompt_length > 0
    
    @pytest.mark.asyncio
    async def test_single_streaming_request(self, benchmark, mock_requests):
        result = await benchmark._single_streaming_request(
            prompt="Test prompt",
            max_new_tokens=10,
            temperature=1.0
        )
        
        assert result is not None
        assert 'ttft' in result
        assert 'tpt_times' in result
        assert 'total_time' in result
        assert 'tokens_generated' in result
        
        assert result['ttft'] > 0
        assert result['total_time'] > 0
        assert result['tokens_generated'] > 0
    
    def test_single_batch_request(self, benchmark):
        with patch('benchmark.latency_test.requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "generated_text": "Test response",
                "output_tokens": 5
            }
            mock_post.return_value = mock_response
            
            request_data = {
                "prompt": "Test prompt",
                "max_new_tokens": 10
            }
            
            result = benchmark._single_batch_request(request_data)
            
            assert result is not None
            assert 'latency' in result
            assert 'tokens_generated' in result
            assert result['latency'] > 0
            assert result['tokens_generated'] == 5
    
    def test_get_summary(self, benchmark):
        # Test with no results
        summary = benchmark.get_summary()
        assert "error" in summary
        
        # Test with results
        from benchmark.latency_test import LatencyResult
        
        mock_results = [
            LatencyResult(
                prompt_length=10,
                max_new_tokens=50,
                num_samples=10,
                time_to_first_token_ms=50.0,
                time_per_token_ms=10.0,
                total_generation_time_ms=500.0,
                tokens_generated=45,
                tokens_per_second=90.0,
                percentiles={'p50': 48.0, 'p90': 55.0, 'p95': 60.0, 'p99': 70.0},
                std_deviation_ms=5.0,
                error_rate=0.0
            ),
            LatencyResult(
                prompt_length=20,
                max_new_tokens=100,
                num_samples=10,
                time_to_first_token_ms=60.0,
                time_per_token_ms=12.0,
                total_generation_time_ms=1200.0,
                tokens_generated=95,
                tokens_per_second=79.2,
                percentiles={'p50': 58.0, 'p90': 65.0, 'p95': 70.0, 'p99': 80.0},
                std_deviation_ms=6.0,
                error_rate=0.1
            )
        ]
        
        benchmark.results = mock_results
        summary = benchmark.get_summary()
        
        assert "total_tests" in summary
        assert summary["total_tests"] == 2
        assert "time_to_first_token" in summary
        assert "time_per_token" in summary
        assert "throughput" in summary
        assert "reliability" in summary


class TestMetricsCollector:
    
    @pytest.fixture
    def collector(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            yield MetricsCollector(output_dir=temp_dir)
    
    def test_init(self, collector):
        assert collector.output_dir.exists()
        assert collector.metrics_history == []
    
    def test_collect_benchmark_metrics(self, collector):
        # Mock results
        mock_throughput_results = [
            Mock(throughput_requests_per_sec=10.0, throughput_tokens_per_sec=200.0, 
                 avg_latency_ms=100.0, p95_latency_ms=150.0, p99_latency_ms=200.0, error_rate=0.0)
        ]
        
        mock_latency_results = [
            Mock(time_to_first_token_ms=50.0, time_per_token_ms=10.0, 
                 tokens_per_second=100.0, error_rate=0.0)
        ]
        
        with patch('psutil.cpu_percent', return_value=50.0), \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('GPUtil.getGPUs', return_value=[]):
            
            mock_memory.return_value = Mock(used=1000000000, total=2000000000, percent=50.0)
            
            metrics = collector.collect_benchmark_metrics(
                throughput_results=mock_throughput_results,
                latency_results=mock_latency_results
            )
            
            assert metrics.timestamp > 0
            assert metrics.test_type == "combined"
            assert len(metrics.performance_metrics) > 0
            assert len(metrics.system_metrics) > 0
            assert len(metrics.error_metrics) > 0
    
    def test_extract_throughput_metrics(self, collector):
        mock_results = [
            Mock(throughput_requests_per_sec=10.0, throughput_tokens_per_sec=200.0, 
                 avg_latency_ms=100.0, p95_latency_ms=150.0, p99_latency_ms=200.0)
        ]
        
        metrics = collector._extract_throughput_metrics(mock_results)
        
        assert "max_throughput_requests_per_sec" in metrics
        assert "avg_throughput_requests_per_sec" in metrics
        assert "max_throughput_tokens_per_sec" in metrics
        assert "min_avg_latency_ms" in metrics
        assert metrics["max_throughput_requests_per_sec"] == 10.0
        assert metrics["max_throughput_tokens_per_sec"] == 200.0
    
    def test_extract_latency_metrics(self, collector):
        mock_results = [
            Mock(time_to_first_token_ms=50.0, time_per_token_ms=10.0, tokens_per_second=100.0)
        ]
        
        metrics = collector._extract_latency_metrics(mock_results)
        
        assert "avg_time_to_first_token_ms" in metrics
        assert "avg_time_per_token_ms" in metrics
        assert "max_tokens_per_second" in metrics
        assert metrics["avg_time_to_first_token_ms"] == 50.0
        assert metrics["avg_time_per_token_ms"] == 10.0
        assert metrics["max_tokens_per_second"] == 100.0
    
    def test_collect_system_metrics(self, collector):
        with patch('psutil.cpu_percent', return_value=75.0), \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('GPUtil.getGPUs') as mock_gpus:
            
            mock_memory.return_value = Mock(
                used=4000000000, total=8000000000, percent=50.0
            )
            
            mock_gpu = Mock()
            mock_gpu.load = 0.8
            mock_gpu.memoryUsed = 2000
            mock_gpu.memoryTotal = 4000
            mock_gpu.memoryUtil = 0.5
            mock_gpu.temperature = 70
            mock_gpus.return_value = [mock_gpu]
            
            metrics = collector._collect_system_metrics()
            
            assert "cpu_utilization_percent" in metrics
            assert "memory_used_gb" in metrics
            assert "gpu_utilization_percent" in metrics
            assert "gpu_memory_used_mb" in metrics
            assert "gpu_temperature_c" in metrics
            
            assert metrics["cpu_utilization_percent"] == 75.0
            assert metrics["gpu_utilization_percent"] == 80.0
            assert metrics["gpu_memory_used_mb"] == 2000
    
    def test_save_and_load_metrics(self, collector):
        # Create mock metrics
        from benchmark.metrics import BenchmarkMetrics
        
        mock_metrics = BenchmarkMetrics(
            timestamp=1234567890.0,
            test_type="throughput",
            configuration={"test": True},
            performance_metrics={"throughput": 100.0},
            system_metrics={"cpu": 50.0},
            error_metrics={"error_rate": 0.0}
        )
        
        collector.metrics_history = [mock_metrics]
        
        # Save metrics
        filepath = collector.save_metrics("test_metrics.json")
        assert os.path.exists(filepath)
        
        # Load metrics
        new_collector = MetricsCollector(output_dir=collector.output_dir)
        new_collector.load_metrics(filepath)
        
        assert len(new_collector.metrics_history) == 1
        loaded_metric = new_collector.metrics_history[0]
        assert loaded_metric.test_type == "throughput"
        assert loaded_metric.timestamp == 1234567890.0
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_create_performance_plots(self, mock_close, mock_savefig, collector):
        # Create mock metrics
        from benchmark.metrics import BenchmarkMetrics
        
        mock_metrics = [
            BenchmarkMetrics(
                timestamp=1234567890.0,
                test_type="throughput",
                configuration={},
                performance_metrics={"max_throughput_tokens_per_sec": 100.0, "avg_time_to_first_token_ms": 50.0, "avg_tokens_per_second": 80.0},
                system_metrics={},
                error_metrics={"avg_error_rate": 0.0}
            ),
            BenchmarkMetrics(
                timestamp=1234567900.0,
                test_type="latency",
                configuration={},
                performance_metrics={"max_throughput_tokens_per_sec": 120.0, "avg_time_to_first_token_ms": 45.0, "avg_tokens_per_second": 90.0},
                system_metrics={},
                error_metrics={"avg_error_rate": 0.1}
            )
        ]
        
        collector.metrics_history = mock_metrics
        
        # Should not raise any exceptions
        collector._create_performance_plots()
        
        # Verify plots were saved
        mock_savefig.assert_called()
        mock_close.assert_called()


@pytest.mark.integration
class TestBenchmarkIntegration:
    
    @pytest.mark.asyncio
    async def test_full_benchmark_pipeline(self):
        """Test the complete benchmark pipeline with mocked server"""
        with patch('benchmark.throughput_test.requests.post') as mock_post:
            # Mock successful responses
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "generated_text": "Test response",
                "output_tokens": 10
            }
            mock_post.return_value = mock_response
            
            # Run throughput benchmark
            throughput_benchmark = ThroughputBenchmark()
            
            with patch.object(throughput_benchmark.system_monitor, 'start_monitoring'), \
                 patch.object(throughput_benchmark.system_monitor, 'stop_monitoring', return_value={}):
                
                throughput_results = await throughput_benchmark.run_throughput_test(
                    prompts=["Test prompt"],
                    batch_sizes=[1],
                    sequence_lengths=[128],
                    num_requests=5,
                    concurrent_requests=2
                )
                
                assert len(throughput_results) == 1
                assert throughput_results[0].batch_size == 1
                assert throughput_results[0].sequence_length == 128
    
    def test_metrics_collection_integration(self):
        """Test metrics collection with mock benchmark results"""
        with tempfile.TemporaryDirectory() as temp_dir:
            collector = MetricsCollector(output_dir=temp_dir)
            
            # Create mock results
            mock_throughput_results = [
                Mock(throughput_requests_per_sec=10.0, throughput_tokens_per_sec=200.0, 
                     avg_latency_ms=100.0, p95_latency_ms=150.0, p99_latency_ms=200.0, error_rate=0.0)
            ]
            
            with patch('psutil.cpu_percent', return_value=50.0), \
                 patch('psutil.virtual_memory') as mock_memory:
                
                mock_memory.return_value = Mock(used=1000000000, total=2000000000, percent=50.0)
                
                # Collect metrics
                metrics = collector.collect_benchmark_metrics(
                    throughput_results=mock_throughput_results
                )
                
                assert metrics is not None
                assert len(collector.metrics_history) == 1
                
                # Save and verify
                filepath = collector.save_metrics()
                assert os.path.exists(filepath)
                
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                assert "metrics" in data
                assert len(data["metrics"]) == 1