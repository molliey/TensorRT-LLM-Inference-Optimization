import torch
import torch.onnx
import onnx
import onnxruntime as ort
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import Optional, Tuple, Dict, Any
import os
import logging

logger = logging.getLogger(__name__)

class ONNXExporter:
    def __init__(self, 
                 model: GPT2LMHeadModel, 
                 tokenizer: GPT2Tokenizer,
                 output_dir: str = "./models/onnx"):
        self.model = model
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.onnx_path = None
        
        os.makedirs(output_dir, exist_ok=True)
    
    def export_to_onnx(self, 
                       batch_size: int = 1,
                       seq_length: int = 512,
                       opset_version: int = 14,
                       dynamic_axes: bool = True) -> str:
        logger.info("Starting ONNX export...")
        
        self.model.eval()
        device = next(self.model.parameters()).device
        
        # Create dummy input
        dummy_input = torch.randint(
            0, self.tokenizer.vocab_size, 
            (batch_size, seq_length), 
            device=device,
            dtype=torch.long
        )
        
        # Define output path
        self.onnx_path = os.path.join(self.output_dir, "gpt2_model.onnx")
        
        # Define dynamic axes for flexible input shapes
        if dynamic_axes:
            dynamic_axes_dict = {
                'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                'logits': {0: 'batch_size', 1: 'sequence_length'}
            }
        else:
            dynamic_axes_dict = None
        
        # Export to ONNX
        with torch.no_grad():
            torch.onnx.export(
                self.model,
                dummy_input,
                self.onnx_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['input_ids'],
                output_names=['logits'],
                dynamic_axes=dynamic_axes_dict,
                verbose=False
            )
        
        logger.info(f"ONNX model exported to: {self.onnx_path}")
        
        # Verify the exported model
        self._verify_onnx_model()
        
        return self.onnx_path
    
    def _verify_onnx_model(self) -> bool:
        try:
            # Load and check ONNX model
            onnx_model = onnx.load(self.onnx_path)
            onnx.checker.check_model(onnx_model)
            
            # Test with ONNX Runtime
            ort_session = ort.InferenceSession(self.onnx_path)
            
            # Test inference
            dummy_input = torch.randint(0, 100, (1, 10)).numpy().astype('int64')
            ort_inputs = {ort_session.get_inputs()[0].name: dummy_input}
            ort_outputs = ort_session.run(None, ort_inputs)
            
            logger.info("ONNX model verification successful")
            logger.info(f"Input shape: {dummy_input.shape}")
            logger.info(f"Output shape: {ort_outputs[0].shape}")
            
            return True
            
        except Exception as e:
            logger.error(f"ONNX model verification failed: {e}")
            return False
    
    def optimize_onnx_model(self) -> str:
        try:
            from onnxoptimizer import optimize
            
            logger.info("Optimizing ONNX model...")
            
            # Load original model
            onnx_model = onnx.load(self.onnx_path)
            
            # Apply optimizations
            optimized_model = optimize(onnx_model, [
                'eliminate_deadend',
                'eliminate_identity',
                'eliminate_nop_dropout',
                'eliminate_nop_pad',
                'eliminate_unused_initializer',
                'extract_constant_to_initializer',
                'fuse_add_bias_into_conv',
                'fuse_bn_into_conv',
                'fuse_consecutive_concats',
                'fuse_consecutive_reduce_unsqueeze',
                'fuse_consecutive_squeezes',
                'fuse_consecutive_transposes',
                'fuse_matmul_add_bias_into_gemm',
                'fuse_pad_into_conv',
                'fuse_transpose_into_gemm',
            ])
            
            # Save optimized model
            optimized_path = self.onnx_path.replace('.onnx', '_optimized.onnx')
            onnx.save(optimized_model, optimized_path)
            
            logger.info(f"Optimized ONNX model saved to: {optimized_path}")
            return optimized_path
            
        except ImportError:
            logger.warning("onnxoptimizer not available, skipping optimization")
            return self.onnx_path
        except Exception as e:
            logger.error(f"ONNX optimization failed: {e}")
            return self.onnx_path
    
    def compare_outputs(self, test_input: torch.Tensor, rtol: float = 1e-3, atol: float = 1e-3) -> bool:
        logger.info("Comparing PyTorch and ONNX outputs...")
        
        # PyTorch inference
        self.model.eval()
        with torch.no_grad():
            pytorch_output = self.model(test_input).logits
        
        # ONNX Runtime inference
        ort_session = ort.InferenceSession(self.onnx_path)
        ort_inputs = {ort_session.get_inputs()[0].name: test_input.cpu().numpy()}
        onnx_output = ort_session.run(None, ort_inputs)[0]
        
        # Compare outputs
        pytorch_output_np = pytorch_output.cpu().numpy()
        diff = torch.abs(torch.from_numpy(onnx_output) - torch.from_numpy(pytorch_output_np))
        max_diff = torch.max(diff).item()
        
        is_close = torch.allclose(
            torch.from_numpy(onnx_output), 
            torch.from_numpy(pytorch_output_np), 
            rtol=rtol, 
            atol=atol
        )
        
        logger.info(f"Max difference: {max_diff}")
        logger.info(f"Outputs match within tolerance: {is_close}")
        
        return is_close
    
    def get_model_info(self) -> Dict[str, Any]:
        if self.onnx_path and os.path.exists(self.onnx_path):
            onnx_model = onnx.load(self.onnx_path)
            
            return {
                "onnx_path": self.onnx_path,
                "opset_version": onnx_model.opset_import[0].version,
                "input_info": [(input.name, [d.dim_value for d in input.type.tensor_type.shape.dim]) 
                              for input in onnx_model.graph.input],
                "output_info": [(output.name, [d.dim_value for d in output.type.tensor_type.shape.dim]) 
                               for output in onnx_model.graph.output],
                "file_size_mb": os.path.getsize(self.onnx_path) / (1024 * 1024)
            }
        else:
            return {"error": "ONNX model not found"}