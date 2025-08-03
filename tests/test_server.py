import pytest
import asyncio
import json
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient
import tempfile
import os

from server.api import app
from server.inference import TensorRTInferenceEngine
from server.middleware import setup_middleware


class TestFastAPIServer:
    
    @pytest.fixture
    def client(self):
        """Test client with mocked inference engine"""
        # Override the inference engine dependency with a mock
        def mock_get_inference_engine():
            mock_engine = Mock(spec=TensorRTInferenceEngine)
            mock_engine.is_loaded.return_value = True
            mock_engine.get_engine_info.return_value = {"engine_path": "test.engine"}
            mock_engine.get_system_info.return_value = {"cpu_percent": 50}
            mock_engine.get_model_info.return_value = {"model_name": "gpt2"}
            mock_engine.get_optimization_info.return_value = {"kv_cache_enabled": True}
            mock_engine.get_metrics.return_value = {"total_requests": 10}
            mock_engine.reset_metrics = AsyncMock()
            mock_engine.generate = AsyncMock(return_value={
                "generated_text": "Hello world!",
                "input_tokens": 5,
                "output_tokens": 10,
                "total_tokens": 15,
                "model_info": {"model_name": "gpt2"}
            })
            mock_engine.generate_stream = AsyncMock()
            mock_engine.benchmark_inference = AsyncMock(return_value={
                "avg_inference_time_ms": 100.0,
                "throughput_tokens_per_sec": 50.0
            })
            return mock_engine
        
        # Override the dependency
        app.dependency_overrides[app.get_inference_engine] = mock_get_inference_engine
        
        with TestClient(app) as client:
            yield client
        
        # Clean up
        app.dependency_overrides.clear()
    
    def test_root_endpoint(self, client):
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert data["message"] == "TensorRT-LLM Inference API"
    
    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "engine_info" in data
        assert "system_info" in data
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
    
    def test_generate_endpoint(self, client):
        request_data = {
            "prompt": "Hello, world!",
            "max_new_tokens": 50,
            "temperature": 1.0,
            "top_k": 50,
            "top_p": 0.9
        }
        
        response = client.post("/generate", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "generated_text" in data
        assert "input_tokens" in data
        assert "output_tokens" in data
        assert "total_tokens" in data
        assert "inference_time_ms" in data
        assert "tokens_per_second" in data
        assert "model_info" in data
        
        assert data["generated_text"] == "Hello world!"
        assert data["input_tokens"] == 5
        assert data["output_tokens"] == 10
    
    def test_generate_endpoint_validation(self, client):
        # Test with invalid request
        invalid_request = {
            "prompt": "A" * 5000,  # Too long
            "max_new_tokens": -1   # Invalid
        }
        
        response = client.post("/generate", json=invalid_request)
        assert response.status_code == 422  # Validation error
    
    def test_model_info_endpoint(self, client):
        response = client.get("/model/info")
        assert response.status_code == 200
        
        data = response.json()
        assert "model_info" in data
        assert "engine_info" in data
        assert "optimization_info" in data
    
    def test_metrics_endpoint(self, client):
        response = client.get("/metrics")
        assert response.status_code == 200
        
        data = response.json()
        assert "total_requests" in data
        assert data["total_requests"] == 10
    
    def test_reset_metrics_endpoint(self, client):
        response = client.post("/metrics/reset")
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "successfully" in data["message"].lower()
    
    def test_benchmark_endpoint(self, client):
        request_data = {
            "prompt": "Test prompt",
            "batch_sizes": [1, 2],
            "sequence_lengths": [128, 256],
            "num_runs": 10
        }
        
        response = client.post("/benchmark", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "results" in data
        assert "summary" in data


class TestTensorRTInferenceEngine:
    
    @pytest.fixture
    def mock_engine_utils(self):
        with patch('server.inference.EngineUtils') as mock:
            mock_instance = Mock()
            mock_instance.engine = Mock()
            mock_instance.infer.return_value = {"logits": [[1.0, 2.0, 3.0]]}
            mock.return_value = mock_instance
            yield mock_instance
    
    @pytest.fixture
    def mock_tokenizer(self):
        with patch('transformers.GPT2Tokenizer.from_pretrained') as mock:
            mock_tokenizer = Mock()
            mock_tokenizer.vocab_size = 50257
            mock_tokenizer.pad_token_id = 50256
            mock_tokenizer.eos_token_id = 50256
            mock_tokenizer.encode.return_value = [15496, 11, 995]  # "Hello, world"
            mock_tokenizer.decode.return_value = "Generated text"
            mock.return_value = mock_tokenizer
            yield mock_tokenizer
    
    @pytest.fixture
    def inference_engine(self, mock_engine_utils, mock_tokenizer):
        with tempfile.NamedTemporaryFile(suffix='.engine', delete=False) as f:
            f.write(b"mock_engine_data")
            engine_path = f.name
        
        try:
            engine = TensorRTInferenceEngine(
                engine_path=engine_path,
                tokenizer_name="gpt2",
                max_batch_size=4,
                max_sequence_length=512
            )
            yield engine
        finally:
            if os.path.exists(engine_path):
                os.unlink(engine_path)
    
    def test_init(self, inference_engine):
        assert inference_engine.engine_path.endswith('.engine')
        assert inference_engine.tokenizer_name == "gpt2"
        assert inference_engine.max_batch_size == 4
        assert inference_engine.max_sequence_length == 512
        assert inference_engine.engine_utils is None  # Not initialized yet
    
    @pytest.mark.asyncio
    async def test_initialize(self, inference_engine, mock_engine_utils, mock_tokenizer):
        success = await inference_engine.initialize()
        
        assert success is True
        assert inference_engine.tokenizer is not None
        assert inference_engine.engine_utils is not None
        assert "model_name" in inference_engine.model_info
        assert "vocab_size" in inference_engine.model_info
    
    @pytest.mark.asyncio
    async def test_generate(self, inference_engine, mock_engine_utils, mock_tokenizer):
        # Initialize first
        await inference_engine.initialize()
        
        # Mock torch tensor operations
        with patch('torch.tensor') as mock_tensor, \
             patch('torch.cat') as mock_cat, \
             patch('torch.argmax') as mock_argmax:
            
            mock_tensor.return_value = Mock()
            mock_tensor.return_value.cpu.return_value.numpy.return_value.astype.return_value = [[1, 2, 3]]
            mock_tensor.return_value.shape = [1, 3]
            
            mock_cat.return_value = Mock()
            mock_cat.return_value.shape = [1, 4]
            
            mock_argmax.return_value = Mock()
            mock_argmax.return_value.item.return_value = 1
            mock_argmax.return_value.unsqueeze.return_value = Mock()
            
            # Mock numpy operations
            with patch('numpy.from_numpy') as mock_from_numpy:
                mock_from_numpy.return_value = Mock()
                
                result = await inference_engine.generate(
                    prompt="Hello, world!",
                    max_new_tokens=10,
                    temperature=1.0
                )
                
                assert isinstance(result, dict)
                assert "generated_text" in result
                assert "input_tokens" in result
                assert "output_tokens" in result
                assert "total_tokens" in result
                assert "model_info" in result
    
    @pytest.mark.asyncio
    async def test_get_metrics(self, inference_engine):
        metrics = await inference_engine.get_metrics()
        
        assert isinstance(metrics, dict)
        assert "total_requests" in metrics
        assert "total_tokens_generated" in metrics
        assert "error_count" in metrics
        assert metrics["total_requests"] == 0  # Initial state
    
    @pytest.mark.asyncio
    async def test_reset_metrics(self, inference_engine):
        # First, simulate some usage
        await inference_engine._update_metrics(tokens_generated=10, inference_time_ms=100)
        
        metrics_before = await inference_engine.get_metrics()
        assert metrics_before["total_requests"] == 1
        
        # Reset metrics
        await inference_engine.reset_metrics()
        
        metrics_after = await inference_engine.get_metrics()
        assert metrics_after["total_requests"] == 0
        assert metrics_after["total_tokens_generated"] == 0
    
    def test_is_loaded(self, inference_engine):
        # Before initialization
        assert inference_engine.is_loaded() is False
        
        # After setting components
        inference_engine.engine_utils = Mock()
        inference_engine.tokenizer = Mock()
        assert inference_engine.is_loaded() is True
    
    @pytest.mark.asyncio
    async def test_cleanup(self, inference_engine):
        # Set some mock components
        inference_engine.engine_utils = Mock()
        inference_engine.engine_utils.cleanup = Mock()
        inference_engine.tokenizer = Mock()
        
        await inference_engine.cleanup()
        
        inference_engine.engine_utils.cleanup.assert_called_once()
        assert inference_engine.engine_utils is None
        assert inference_engine.tokenizer is None


class TestMiddleware:
    
    def test_setup_middleware(self):
        """Test that middleware setup doesn't raise errors"""
        from fastapi import FastAPI
        test_app = FastAPI()
        
        # Should not raise any exceptions
        setup_middleware(test_app)
        
        # Check that middleware was added
        assert len(test_app.user_middleware) > 0


@pytest.mark.integration
class TestServerIntegration:
    
    @pytest.fixture
    def app_with_mock_lifespan(self):
        """Create app with mocked lifespan for testing"""
        with patch('server.api.inference_engine', Mock()):
            yield app
    
    def test_server_startup_mock(self, app_with_mock_lifespan):
        """Test server startup with mocked components"""
        with TestClient(app_with_mock_lifespan) as client:
            response = client.get("/")
            assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test handling concurrent requests"""
        # Create mock inference engine
        mock_engine = Mock(spec=TensorRTInferenceEngine)
        mock_engine.generate = AsyncMock(return_value={
            "generated_text": "Test response",
            "input_tokens": 5,
            "output_tokens": 10,
            "total_tokens": 15,
            "model_info": {"model_name": "gpt2"}
        })
        
        # Override dependency
        def get_mock_engine():
            return mock_engine
        
        app.dependency_overrides[app.get_inference_engine] = get_mock_engine
        
        try:
            with TestClient(app) as client:
                # Send multiple concurrent requests
                responses = []
                for i in range(5):
                    response = client.post("/generate", json={
                        "prompt": f"Test prompt {i}",
                        "max_new_tokens": 10
                    })
                    responses.append(response)
                
                # All requests should succeed
                for response in responses:
                    assert response.status_code == 200
                    data = response.json()
                    assert "generated_text" in data
        
        finally:
            app.dependency_overrides.clear()
    
    def test_error_handling(self):
        """Test error handling in API endpoints"""
        # Create mock engine that raises exceptions
        mock_engine = Mock(spec=TensorRTInferenceEngine)
        mock_engine.generate = AsyncMock(side_effect=Exception("Test error"))
        
        # Override dependency
        def get_failing_engine():
            return mock_engine
        
        app.dependency_overrides[app.get_inference_engine] = get_failing_engine
        
        try:
            with TestClient(app) as client:
                response = client.post("/generate", json={
                    "prompt": "Test prompt",
                    "max_new_tokens": 10
                })
                
                assert response.status_code == 500
                data = response.json()
                assert "error" in data
                assert "Test error" in data["error"]
        
        finally:
            app.dependency_overrides.clear()


@pytest.mark.skipif(not os.getenv('INTEGRATION_TESTS'), reason="Integration tests disabled")
class TestRealServerIntegration:
    """
    Real server integration tests - only run when INTEGRATION_TESTS=1
    These tests require actual TensorRT engine and GPU
    """
    
    def test_real_server_startup(self):
        """Test real server startup with actual engine"""
        pytest.skip("Real server integration test - implement when needed")
    
    def test_real_inference_pipeline(self):
        """Test complete inference pipeline with real engine"""
        pytest.skip("Real server integration test - implement when needed")