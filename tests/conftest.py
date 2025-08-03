import pytest
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import torch


@pytest.fixture(scope="session")
def test_data_dir():
    """Create a temporary directory for test data"""
    temp_dir = tempfile.mkdtemp(prefix="tensorrt_llm_test_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="session")
def mock_cuda_available():
    """Mock CUDA availability for tests"""
    with patch('torch.cuda.is_available', return_value=True), \
         patch('torch.cuda.device_count', return_value=1), \
         patch('torch.cuda.get_device_name', return_value="Mock GPU"):
        yield


@pytest.fixture(scope="function")
def mock_tensorrt():
    """Mock TensorRT components for testing"""
    with patch('tensorrt.Logger') as mock_logger, \
         patch('tensorrt.Builder') as mock_builder, \
         patch('tensorrt.Runtime') as mock_runtime, \
         patch('tensorrt.OnnxParser') as mock_parser:
        
        # Configure mocks
        mock_logger.return_value = Mock()
        mock_builder.return_value = Mock()
        mock_runtime.return_value = Mock()
        mock_parser.return_value = Mock()
        
        yield {
            'logger': mock_logger,
            'builder': mock_builder,
            'runtime': mock_runtime,
            'parser': mock_parser
        }


@pytest.fixture(scope="function")
def mock_transformers():
    """Mock transformers components for testing"""
    with patch('transformers.GPT2LMHeadModel') as mock_model, \
         patch('transformers.GPT2Tokenizer') as mock_tokenizer, \
         patch('transformers.GPT2Config') as mock_config:
        
        # Configure mock model
        mock_model_instance = Mock()
        mock_model_instance.eval.return_value = mock_model_instance
        mock_model_instance.parameters.return_value = [Mock()]
        mock_model_instance.modules.return_value = []
        mock_model.from_pretrained.return_value = mock_model_instance
        
        # Configure mock tokenizer
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.vocab_size = 50257
        mock_tokenizer_instance.pad_token = "<|endoftext|>"
        mock_tokenizer_instance.eos_token = "<|endoftext|>"
        mock_tokenizer_instance.pad_token_id = 50256
        mock_tokenizer_instance.eos_token_id = 50256
        mock_tokenizer_instance.encode.return_value = [15496, 11, 995]  # "Hello, world"
        mock_tokenizer_instance.decode.return_value = "Hello, world"
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        # Configure mock config
        mock_config_instance = Mock()
        mock_config_instance.vocab_size = 50257
        mock_config_instance.n_positions = 1024
        mock_config_instance.n_embd = 768
        mock_config_instance.n_layer = 12
        mock_config_instance.n_head = 12
        mock_config.from_pretrained.return_value = mock_config_instance
        
        yield {
            'model': mock_model,
            'tokenizer': mock_tokenizer,
            'config': mock_config,
            'model_instance': mock_model_instance,
            'tokenizer_instance': mock_tokenizer_instance,
            'config_instance': mock_config_instance
        }


@pytest.fixture(scope="function")
def mock_onnx():
    """Mock ONNX components for testing"""
    with patch('onnx.load') as mock_load, \
         patch('onnx.checker.check_model') as mock_check, \
         patch('onnx.save') as mock_save, \
         patch('onnxruntime.InferenceSession') as mock_session:
        
        # Configure mock ONNX model
        mock_model = Mock()
        mock_model.opset_import = [Mock()]
        mock_model.opset_import[0].version = 14
        mock_model.graph.input = []
        mock_model.graph.output = []
        mock_load.return_value = mock_model
        
        # Configure mock ONNX Runtime session
        mock_session_instance = Mock()
        mock_session_instance.get_inputs.return_value = [Mock()]
        mock_session_instance.get_inputs.return_value[0].name = "input_ids"
        mock_session_instance.run.return_value = [torch.randn(1, 10, 50257).numpy()]
        mock_session.return_value = mock_session_instance
        
        yield {
            'load': mock_load,
            'check': mock_check,
            'save': mock_save,
            'session': mock_session,
            'model': mock_model,
            'session_instance': mock_session_instance
        }


@pytest.fixture(scope="function")
def mock_pycuda():
    """Mock PyCUDA components for testing"""
    with patch('pycuda.driver.mem_alloc') as mock_mem_alloc, \
         patch('pycuda.driver.memcpy_htod_async') as mock_memcpy_htod, \
         patch('pycuda.driver.memcpy_dtoh_async') as mock_memcpy_dtoh, \
         patch('pycuda.driver.Stream') as mock_stream, \
         patch('pycuda.driver.Context') as mock_context:
        
        # Configure mocks
        mock_memory = Mock()
        mock_memory.free = Mock()
        mock_mem_alloc.return_value = mock_memory
        
        mock_stream_instance = Mock()
        mock_stream_instance.synchronize = Mock()
        mock_stream.return_value = mock_stream_instance
        
        mock_context.synchronize = Mock()
        
        yield {
            'mem_alloc': mock_mem_alloc,
            'memcpy_htod': mock_memcpy_htod,
            'memcpy_dtoh': mock_memcpy_dtoh,
            'stream': mock_stream,
            'context': mock_context,
            'memory': mock_memory,
            'stream_instance': mock_stream_instance
        }


@pytest.fixture(scope="function")
def mock_system_monitoring():
    """Mock system monitoring components"""
    with patch('psutil.cpu_percent', return_value=50.0) as mock_cpu, \
         patch('psutil.virtual_memory') as mock_memory, \
         patch('GPUtil.getGPUs') as mock_gpus:
        
        # Configure mock memory
        mock_memory_info = Mock()
        mock_memory_info.used = 4000000000  # 4GB
        mock_memory_info.total = 8000000000  # 8GB
        mock_memory_info.percent = 50.0
        mock_memory.return_value = mock_memory_info
        
        # Configure mock GPU
        mock_gpu = Mock()
        mock_gpu.id = 0
        mock_gpu.name = "Mock GPU"
        mock_gpu.load = 0.75  # 75% utilization
        mock_gpu.memoryUsed = 2000  # 2GB
        mock_gpu.memoryTotal = 4000  # 4GB
        mock_gpu.memoryUtil = 0.5  # 50%
        mock_gpu.temperature = 70  # 70°C
        mock_gpus.return_value = [mock_gpu]
        
        yield {
            'cpu': mock_cpu,
            'memory': mock_memory,
            'gpus': mock_gpus,
            'memory_info': mock_memory_info,
            'gpu': mock_gpu
        }


@pytest.fixture
def sample_model_config():
    """Sample model configuration for testing"""
    return {
        "model_name": "gpt2",
        "vocab_size": 50257,
        "n_positions": 1024,
        "n_embd": 768,
        "n_layer": 12,
        "n_head": 12,
        "intermediate_size": 3072,
        "max_position_embeddings": 1024
    }


@pytest.fixture
def sample_engine_config():
    """Sample engine configuration for testing"""
    return {
        "precision": "fp16",
        "max_workspace_size": 1073741824,  # 1GB
        "max_batch_size": 8,
        "max_sequence_length": 1024,
        "enable_kv_cache": True,
        "enable_flash_attention": True
    }


@pytest.fixture
def sample_server_config():
    """Sample server configuration for testing"""
    return {
        "host": "0.0.0.0",
        "port": 8000,
        "workers": 1,
        "log_level": "info",
        "max_batch_size": 8,
        "max_sequence_length": 1024
    }


@pytest.fixture
def mock_file_operations():
    """Mock file operations for testing"""
    mock_files = {}
    
    def mock_open(filename, mode='r', *args, **kwargs):
        if 'w' in mode:
            mock_files[filename] = ""
            return Mock(__enter__=Mock(return_value=Mock(write=lambda x: None)), __exit__=Mock())
        elif 'r' in mode:
            content = mock_files.get(filename, '{"test": "data"}')
            mock_file = Mock()
            mock_file.read.return_value = content
            mock_file.__enter__ = Mock(return_value=mock_file)
            mock_file.__exit__ = Mock()
            return mock_file
        else:
            return Mock()
    
    with patch('builtins.open', side_effect=mock_open), \
         patch('os.path.exists', return_value=True), \
         patch('os.makedirs'), \
         patch('os.path.getsize', return_value=1024):
        
        yield mock_files


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment"""
    # Set environment variables for testing
    os.environ['TESTING'] = '1'
    os.environ['LOG_LEVEL'] = 'DEBUG'
    
    yield
    
    # Cleanup
    if 'TESTING' in os.environ:
        del os.environ['TESTING']


@pytest.fixture
def event_loop():
    """Create an event loop for async tests"""
    import asyncio
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on conditions"""
    for item in items:
        # Add gpu marker to tests that require CUDA
        if "cuda" in item.name.lower() or "gpu" in item.name.lower():
            item.add_marker(pytest.mark.gpu)
        
        # Add slow marker to integration tests
        if "integration" in item.name.lower():
            item.add_marker(pytest.mark.slow)
        
        # Skip integration tests by default unless explicitly requested
        if item.get_closest_marker("integration") and not config.getoption("--integration"):
            item.add_marker(pytest.mark.skip(reason="Integration tests skipped by default"))


def pytest_addoption(parser):
    """Add custom command line options"""
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="Run integration tests"
    )
    parser.addoption(
        "--gpu",
        action="store_true", 
        default=False,
        help="Run GPU tests"
    )


# Custom assertions
def assert_model_output_shape(output, batch_size, seq_length, vocab_size):
    """Assert that model output has correct shape"""
    if hasattr(output, 'logits'):
        output = output.logits
    
    assert output.shape[0] == batch_size, f"Expected batch size {batch_size}, got {output.shape[0]}"
    assert output.shape[1] == seq_length, f"Expected sequence length {seq_length}, got {output.shape[1]}"
    assert output.shape[2] == vocab_size, f"Expected vocab size {vocab_size}, got {output.shape[2]}"


def assert_benchmark_result_valid(result):
    """Assert that benchmark result has valid structure"""
    required_fields = [
        'avg_inference_time_ms', 'throughput_requests_per_sec', 
        'throughput_tokens_per_sec', 'error_rate'
    ]
    
    for field in required_fields:
        assert hasattr(result, field), f"Missing field: {field}"
        value = getattr(result, field)
        assert isinstance(value, (int, float)), f"Field {field} should be numeric, got {type(value)}"
        assert value >= 0, f"Field {field} should be non-negative, got {value}"


# Export custom assertions
pytest.assert_model_output_shape = assert_model_output_shape
pytest.assert_benchmark_result_valid = assert_benchmark_result_valid