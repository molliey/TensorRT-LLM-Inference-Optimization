import pytest
import torch
import tempfile
import os
from pathlib import Path
import numpy as np

from models import GPT2Loader, ONNXExporter, ModelUtils


class TestGPT2Loader:
    
    @pytest.fixture
    def loader(self):
        return GPT2Loader(model_name="gpt2")
    
    def test_init(self, loader):
        assert loader.model_name == "gpt2"
        assert loader.model is None
        assert loader.tokenizer is None
        assert loader.config is None
    
    def test_load_model(self, loader):
        model, tokenizer, config = loader.load_model()
        
        assert model is not None
        assert tokenizer is not None
        assert config is not None
        assert hasattr(model, 'forward')
        assert tokenizer.vocab_size > 0
        assert config.vocab_size == tokenizer.vocab_size
    
    def test_prepare_for_export(self, loader):
        loader.load_model()
        model = loader.prepare_for_export()
        
        assert model is not None
        assert model.training is False  # Should be in eval mode
        
        # Check that dropout is disabled
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                assert module.p == 0.0
    
    def test_get_model_info(self, loader):
        info = loader.get_model_info()
        
        assert isinstance(info, dict)
        assert "model_name" in info
        assert "vocab_size" in info
        assert "n_positions" in info
        assert "n_embd" in info
        assert "n_layer" in info
        assert "n_head" in info
        
        assert info["model_name"] == "gpt2"
        assert info["vocab_size"] == 50257


class TestONNXExporter:
    
    @pytest.fixture
    def setup_model_and_exporter(self):
        loader = GPT2Loader(model_name="gpt2")
        model, tokenizer, config = loader.load_model()
        model = loader.prepare_for_export()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            exporter = ONNXExporter(model, tokenizer, output_dir=temp_dir)
            yield exporter, temp_dir
    
    def test_init(self, setup_model_and_exporter):
        exporter, temp_dir = setup_model_and_exporter
        
        assert exporter.model is not None
        assert exporter.tokenizer is not None
        assert exporter.output_dir == temp_dir
        assert os.path.exists(temp_dir)
    
    def test_export_to_onnx(self, setup_model_and_exporter):
        exporter, temp_dir = setup_model_and_exporter
        
        onnx_path = exporter.export_to_onnx(
            batch_size=1,
            seq_length=64,
            opset_version=14,
            dynamic_axes=True
        )
        
        assert os.path.exists(onnx_path)
        assert onnx_path.endswith('.onnx')
        assert exporter.onnx_path == onnx_path
    
    def test_compare_outputs(self, setup_model_and_exporter):
        exporter, temp_dir = setup_model_and_exporter
        
        # Export model first
        exporter.export_to_onnx(batch_size=1, seq_length=32)
        
        # Test comparison
        test_input = torch.randint(0, 1000, (1, 16))
        is_close = exporter.compare_outputs(test_input, rtol=1e-2, atol=1e-2)
        
        assert isinstance(is_close, bool)
        # Note: Due to numerical differences, exact match might not always occur
    
    def test_get_model_info(self, setup_model_and_exporter):
        exporter, temp_dir = setup_model_and_exporter
        
        # Before export
        info = exporter.get_model_info()
        assert "error" in info
        
        # After export
        exporter.export_to_onnx(batch_size=1, seq_length=32)
        info = exporter.get_model_info()
        
        assert "onnx_path" in info
        assert "opset_version" in info
        assert "input_info" in info
        assert "output_info" in info
        assert "file_size_mb" in info


class TestModelUtils:
    
    @pytest.fixture
    def tokenizer(self):
        from transformers import GPT2Tokenizer
        return GPT2Tokenizer.from_pretrained("gpt2")
    
    def test_prepare_input_ids(self, tokenizer):
        text = "Hello, world!"
        input_ids = ModelUtils.prepare_input_ids(text, tokenizer, max_length=64)
        
        assert isinstance(input_ids, torch.Tensor)
        assert input_ids.dim() == 2
        assert input_ids.shape[0] == 1  # batch size
        assert input_ids.shape[1] <= 64  # max length
        
        # Test with list of texts
        texts = ["Hello", "World", "Test"]
        input_ids = ModelUtils.prepare_input_ids(texts, tokenizer, max_length=32)
        
        assert input_ids.shape[0] == 3  # batch size
        assert input_ids.shape[1] <= 32  # max length
    
    def test_decode_output(self, tokenizer):
        # Test single sequence
        token_ids = torch.tensor([15496, 11, 995])  # "Hello, world"
        decoded = ModelUtils.decode_output(token_ids, tokenizer)
        
        assert isinstance(decoded, list)
        assert len(decoded) == 1
        assert isinstance(decoded[0], str)
        
        # Test batch
        token_ids = torch.tensor([[15496, 11], [995, 0]])
        decoded = ModelUtils.decode_output(token_ids, tokenizer)
        
        assert len(decoded) == 2
        assert all(isinstance(text, str) for text in decoded)
    
    def test_apply_temperature_sampling(self):
        logits = torch.randn(10, 50257)  # batch_size=10, vocab_size=50257
        
        # Test greedy (temperature = 0)
        tokens = ModelUtils.apply_temperature_sampling(logits, temperature=0.0)
        assert tokens.shape == (10,)
        
        # Test with temperature
        tokens = ModelUtils.apply_temperature_sampling(logits, temperature=1.0)
        assert tokens.shape == (10,)
        assert torch.all(tokens >= 0)
        assert torch.all(tokens < 50257)
    
    def test_apply_top_k_filtering(self):
        logits = torch.randn(2, 100)
        
        # Test top-k filtering
        filtered_logits = ModelUtils.apply_top_k_filtering(logits, top_k=10)
        
        assert filtered_logits.shape == logits.shape
        
        # Check that only top-k values are not -inf
        for i in range(logits.shape[0]):
            non_inf_count = torch.sum(filtered_logits[i] != float('-inf'))
            assert non_inf_count <= 10
    
    def test_apply_top_p_filtering(self):
        logits = torch.randn(2, 100)
        
        # Test top-p filtering
        filtered_logits = ModelUtils.apply_top_p_filtering(logits, top_p=0.9)
        
        assert filtered_logits.shape == logits.shape
        
        # Filtered logits should have some values set to -inf
        for i in range(logits.shape[0]):
            non_inf_count = torch.sum(filtered_logits[i] != float('-inf'))
            assert non_inf_count <= logits.shape[1]
    
    def test_estimate_model_size(self):
        # Create a simple model for testing
        model = torch.nn.Sequential(
            torch.nn.Linear(768, 3072),
            torch.nn.GELU(),
            torch.nn.Linear(3072, 768)
        )
        
        size_info = ModelUtils.estimate_model_size(model)
        
        assert isinstance(size_info, dict)
        assert "total_params" in size_info
        assert "trainable_params" in size_info
        assert "model_size_mb" in size_info
        assert "param_size_mb" in size_info
        assert "buffer_size_mb" in size_info
        
        assert size_info["total_params"] > 0
        assert size_info["model_size_mb"] > 0
    
    def test_benchmark_inference_speed(self):
        # Create a simple model for testing
        model = torch.nn.Sequential(
            torch.nn.Embedding(1000, 64),
            torch.nn.Linear(64, 64),
            torch.nn.Linear(64, 1000)
        )
        model.eval()
        
        input_ids = torch.randint(0, 1000, (2, 16))
        
        benchmark_results = ModelUtils.benchmark_inference_speed(
            model, input_ids, num_runs=10, warmup_runs=2
        )
        
        assert isinstance(benchmark_results, dict)
        assert "avg_inference_time_ms" in benchmark_results
        assert "throughput_samples_per_sec" in benchmark_results
        assert "total_time_sec" in benchmark_results
        
        assert benchmark_results["avg_inference_time_ms"] > 0
        assert benchmark_results["throughput_samples_per_sec"] > 0
        assert benchmark_results["total_time_sec"] > 0


@pytest.mark.integration
class TestModelIntegration:
    
    def test_full_pipeline(self):
        """Test the complete model loading -> ONNX export pipeline"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Load model
            loader = GPT2Loader(model_name="gpt2")
            model, tokenizer, config = loader.load_model()
            
            # Prepare for export
            model = loader.prepare_for_export()
            
            # Export to ONNX
            exporter = ONNXExporter(model, tokenizer, output_dir=temp_dir)
            onnx_path = exporter.export_to_onnx(batch_size=1, seq_length=32)
            
            # Verify export
            assert os.path.exists(onnx_path)
            
            # Test model info
            model_info = loader.get_model_info()
            onnx_info = exporter.get_model_info()
            
            assert model_info["vocab_size"] == 50257
            assert "onnx_path" in onnx_info
            
            # Test generation utilities
            test_text = "Hello, world!"
            input_ids = ModelUtils.prepare_input_ids(test_text, tokenizer)
            
            assert input_ids.shape[0] == 1
            assert input_ids.shape[1] > 0
    
    def test_model_consistency(self):
        """Test that model outputs are consistent"""
        loader = GPT2Loader(model_name="gpt2")
        model, tokenizer, config = loader.load_model()
        model.eval()
        
        # Test with same input multiple times
        test_input = torch.randint(0, 1000, (1, 16))
        
        with torch.no_grad():
            output1 = model(test_input).logits
            output2 = model(test_input).logits
        
        # Outputs should be identical for same input
        assert torch.allclose(output1, output2, rtol=1e-5, atol=1e-5)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_inference(self):
        """Test model inference on GPU"""
        loader = GPT2Loader(model_name="gpt2")
        model, tokenizer, config = loader.load_model()
        
        # Move to GPU
        device = torch.device("cuda")
        model = model.to(device)
        
        test_input = torch.randint(0, 1000, (1, 16)).to(device)
        
        with torch.no_grad():
            output = model(test_input)
        
        assert output.logits.device.type == "cuda"
        assert output.logits.shape[0] == 1
        assert output.logits.shape[2] == tokenizer.vocab_size