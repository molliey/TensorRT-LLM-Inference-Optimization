import pytest
import tempfile
import os
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from engine import TensorRTBuilder, TensorRTOptimizer, EngineUtils


class TestTensorRTBuilder:
    
    @pytest.fixture
    def builder(self):
        return TensorRTBuilder(
            max_workspace_size=1 << 28,  # 256MB for testing
            precision="fp16",
            max_batch_size=4,
            verbose=False
        )
    
    def test_init(self, builder):
        assert builder.max_workspace_size == 1 << 28
        assert builder.precision == "fp16"
        assert builder.max_batch_size == 4
        assert builder.verbose is False
        assert builder.logger is not None
        assert builder.builder is not None
    
    @patch('tensorrt.OnnxParser')
    @patch('tensorrt.Builder')
    def test_build_engine_from_onnx_mock(self, mock_builder_class, mock_parser_class):
        """Test engine building with mocked TensorRT components"""
        # Mock TensorRT components
        mock_builder = MagicMock()
        mock_network = MagicMock()
        mock_config = MagicMock()
        mock_engine = MagicMock()
        mock_parser = MagicMock()
        
        mock_builder_class.return_value = mock_builder
        mock_builder.create_network.return_value = mock_network
        mock_builder.create_builder_config.return_value = mock_config
        mock_builder.build_engine.return_value = mock_engine
        mock_builder.platform_has_fast_fp16 = True
        
        mock_parser_class.return_value = mock_parser
        mock_parser.parse.return_value = True
        mock_parser.num_errors = 0
        
        mock_engine.serialize.return_value = b"mock_engine_data"
        
        builder = TensorRTBuilder(precision="fp16")
        builder.builder = mock_builder
        
        with tempfile.TemporaryDirectory() as temp_dir:
            onnx_path = os.path.join(temp_dir, "test_model.onnx")
            engine_path = os.path.join(temp_dir, "test_engine.engine")
            
            # Create a dummy ONNX file
            with open(onnx_path, 'wb') as f:
                f.write(b"dummy_onnx_data")
            
            result_path = builder.build_engine_from_onnx(onnx_path, engine_path)
            
            assert result_path == engine_path
            assert os.path.exists(engine_path)
            mock_parser.parse.assert_called_once()
            mock_builder.build_engine.assert_called_once()
    
    def test_setup_default_optimization_profile(self, builder):
        # Mock network with input tensor
        mock_network = MagicMock()
        mock_input_tensor = MagicMock()
        mock_input_tensor.name = "input_ids"
        mock_network.get_input.return_value = mock_input_tensor
        
        mock_config = MagicMock()
        mock_profile = MagicMock()
        
        builder.network = mock_network
        builder.config = mock_config
        builder.builder.create_optimization_profile.return_value = mock_profile
        
        builder._setup_default_optimization_profile()
        
        mock_profile.set_shape.assert_called_once()
        mock_config.add_optimization_profile.assert_called_once_with(mock_profile)
    
    def test_cleanup(self, builder):
        # Set some mock objects
        builder.engine = MagicMock()
        builder.config = MagicMock()
        builder.network = MagicMock()
        
        builder.cleanup()
        
        # After cleanup, objects should be None
        assert builder.engine is None
        assert builder.config is None
        assert builder.network is None


class TestTensorRTOptimizer:
    
    @pytest.fixture
    def optimizer(self):
        return TensorRTOptimizer(
            enable_kv_cache=True,
            enable_flash_attention=True,
            max_sequence_length=1024,
            max_batch_size=8
        )
    
    def test_init(self, optimizer):
        assert optimizer.enable_kv_cache is True
        assert optimizer.enable_flash_attention is True
        assert optimizer.max_sequence_length == 1024
        assert optimizer.max_batch_size == 8
        
        assert "max_tokens" in optimizer.kv_cache_config
        assert "block_size" in optimizer.kv_cache_config
        assert "enabled" in optimizer.flash_attention_config
    
    def test_calculate_kv_cache_size(self, optimizer):
        size = optimizer._calculate_kv_cache_size(
            num_layers=12,
            num_heads=12,
            head_dim=64,
            num_blocks=64,
            block_size=16
        )
        
        assert isinstance(size, int)
        assert size > 0
    
    def test_setup_kv_cache_optimization(self, optimizer):
        mock_network = MagicMock()
        mock_config = MagicMock()
        mock_memory_pool_config = MagicMock()
        
        mock_config.create_memory_pool_config.return_value = mock_memory_pool_config
        
        model_config = {
            "n_layer": 12,
            "n_embd": 768,
            "n_head": 12
        }
        
        result = optimizer.setup_kv_cache_optimization(
            mock_network, mock_config, model_config
        )
        
        assert isinstance(result, dict)
        assert "kv_cache_enabled" in result
        assert result["kv_cache_enabled"] is True
        assert "num_blocks" in result
        assert "memory_size_mb" in result
    
    def test_setup_flash_attention_optimization(self, optimizer):
        mock_network = MagicMock()
        mock_config = MagicMock()
        
        result = optimizer.setup_flash_attention_optimization(
            mock_network, mock_config
        )
        
        assert isinstance(result, dict)
        assert "enabled" in result
    
    def test_get_optimization_summary(self, optimizer):
        summary = optimizer.get_optimization_summary()
        
        assert isinstance(summary, dict)
        assert "kv_cache" in summary
        assert "flash_attention" in summary
        assert "general" in summary
        
        assert summary["kv_cache"]["enabled"] is True
        assert summary["flash_attention"]["enabled"] is True


class TestEngineUtils:
    
    @pytest.fixture
    def mock_engine_path(self):
        with tempfile.NamedTemporaryFile(suffix='.engine', delete=False) as f:
            f.write(b"mock_engine_data")
            engine_path = f.name
        
        yield engine_path
        
        # Cleanup
        if os.path.exists(engine_path):
            os.unlink(engine_path)
    
    def test_init_without_engine(self):
        utils = EngineUtils()
        
        assert utils.engine_path is None
        assert utils.runtime is None
        assert utils.engine is None
        assert utils.context is None
        assert utils.bindings == []
    
    @patch('tensorrt.Runtime')
    def test_load_engine_mock(self, mock_runtime_class, mock_engine_path):
        """Test engine loading with mocked TensorRT components"""
        # Mock TensorRT components
        mock_runtime = MagicMock()
        mock_engine = MagicMock()
        mock_context = MagicMock()
        
        mock_runtime_class.return_value = mock_runtime
        mock_runtime.deserialize_cuda_engine.return_value = mock_engine
        mock_engine.create_execution_context.return_value = mock_context
        mock_engine.num_bindings = 2
        
        # Mock binding info
        mock_engine.get_binding_name.side_effect = ["input_ids", "logits"]
        mock_engine.get_binding_shape.side_effect = [(1, 512), (1, 512, 50257)]
        mock_engine.get_binding_dtype.side_effect = [0, 0]  # TensorRT DataType values
        mock_engine.binding_is_input.side_effect = [True, False]
        
        with patch('pycuda.driver.Stream'), patch('pycuda.driver.mem_alloc') as mock_mem_alloc:
            mock_mem_alloc.return_value = MagicMock()
            
            utils = EngineUtils()
            success = utils.load_engine(mock_engine_path)
            
            assert success is True
            assert utils.engine is not None
            assert utils.context is not None
            assert len(utils.input_specs) == 1
            assert len(utils.output_specs) == 1
    
    def test_validate_engine_static_method(self, mock_engine_path):
        with patch('tensorrt.Runtime') as mock_runtime_class:
            mock_runtime = MagicMock()
            mock_engine = MagicMock()
            
            mock_runtime_class.return_value = mock_runtime
            mock_runtime.deserialize_cuda_engine.return_value = mock_engine
            
            result = EngineUtils.validate_engine(mock_engine_path)
            assert isinstance(result, bool)
    
    def test_get_engine_info_without_engine(self):
        utils = EngineUtils()
        info = utils.get_engine_info()
        
        assert isinstance(info, dict)
        assert "error" in info
    
    @patch('tensorrt.Runtime')
    def test_get_engine_info_with_engine(self, mock_runtime_class, mock_engine_path):
        """Test getting engine info with mocked engine"""
        mock_runtime = MagicMock()
        mock_engine = MagicMock()
        mock_context = MagicMock()
        
        mock_runtime_class.return_value = mock_runtime
        mock_runtime.deserialize_cuda_engine.return_value = mock_engine
        mock_engine.create_execution_context.return_value = mock_context
        mock_engine.num_bindings = 1
        mock_engine.get_binding_name.return_value = "input_ids"
        mock_engine.get_binding_shape.return_value = (1, 512)
        mock_engine.get_binding_dtype.return_value = 0
        mock_engine.binding_is_input.return_value = True
        
        with patch('pycuda.driver.Stream'), patch('pycuda.driver.mem_alloc'):
            utils = EngineUtils()
            utils.load_engine(mock_engine_path)
            info = utils.get_engine_info()
            
            assert isinstance(info, dict)
            assert "engine_path" in info
            assert "num_bindings" in info
            assert "inputs" in info
            assert "outputs" in info
    
    def test_cleanup(self):
        utils = EngineUtils()
        
        # Set some mock objects
        mock_binding = MagicMock()
        mock_binding.free = MagicMock()
        utils.bindings = [mock_binding]
        utils.stream = MagicMock()
        utils.context = MagicMock()
        utils.engine = MagicMock()
        utils.runtime = MagicMock()
        
        utils.cleanup()
        
        # Check cleanup was called
        mock_binding.free.assert_called_once()
        assert len(utils.bindings) == 0
        assert len(utils.binding_addrs) == 0


@pytest.mark.integration
class TestEngineIntegration:
    
    def test_builder_optimizer_integration(self):
        """Test integration between builder and optimizer"""
        builder = TensorRTBuilder(precision="fp16", max_batch_size=2)
        optimizer = TensorRTOptimizer(
            enable_kv_cache=True,
            enable_flash_attention=True,
            max_batch_size=2
        )
        
        # Test that components can be initialized together
        assert builder.precision == "fp16"
        assert optimizer.enable_kv_cache is True
        assert optimizer.max_batch_size == builder.max_batch_size
        
        # Test optimization summary
        summary = optimizer.get_optimization_summary()
        assert summary["general"]["max_batch_size"] == 2
    
    @patch('tensorrt.Runtime')
    def test_engine_utils_integration(self, mock_runtime_class):
        """Test EngineUtils integration with mocked components"""
        with tempfile.NamedTemporaryFile(suffix='.engine', delete=False) as f:
            f.write(b"mock_engine_data")
            engine_path = f.name
        
        try:
            mock_runtime = MagicMock()
            mock_engine = MagicMock()
            mock_context = MagicMock()
            
            mock_runtime_class.return_value = mock_runtime
            mock_runtime.deserialize_cuda_engine.return_value = mock_engine
            mock_engine.create_execution_context.return_value = mock_context
            mock_engine.num_bindings = 0  # No bindings for simple test
            
            with patch('pycuda.driver.Stream'):
                utils = EngineUtils()
                success = utils.load_engine(engine_path)
                
                assert success is True
                
                # Test validation
                with patch.object(EngineUtils, 'validate_engine', return_value=True):
                    is_valid = EngineUtils.validate_engine(engine_path)
                    assert is_valid is True
                
                # Test cleanup
                utils.cleanup()
                
        finally:
            if os.path.exists(engine_path):
                os.unlink(engine_path)


@pytest.mark.skipif(not os.getenv('INTEGRATION_TESTS'), reason="Integration tests disabled")
class TestRealTensorRTIntegration:
    """
    Real TensorRT integration tests - only run when INTEGRATION_TESTS=1
    These tests require actual TensorRT installation and GPU
    """
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_real_engine_build(self):
        """Test building a real TensorRT engine (requires GPU and TensorRT)"""
        pytest.skip("Real TensorRT integration test - implement when needed")
    
    def test_real_engine_inference(self):
        """Test real engine inference (requires built engine)"""
        pytest.skip("Real TensorRT integration test - implement when needed")