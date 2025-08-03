import torch
import numpy as np
from transformers import GPT2Tokenizer
from typing import List, Dict, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)

class ModelUtils:
    @staticmethod
    def prepare_input_ids(
        text: Union[str, List[str]], 
        tokenizer: GPT2Tokenizer,
        max_length: int = 512,
        padding: bool = True,
        truncation: bool = True
    ) -> torch.Tensor:
        if isinstance(text, str):
            text = [text]
        
        encoded = tokenizer(
            text,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors="pt"
        )
        
        return encoded['input_ids']
    
    @staticmethod
    def decode_output(
        token_ids: torch.Tensor,
        tokenizer: GPT2Tokenizer,
        skip_special_tokens: bool = True
    ) -> List[str]:
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)
        
        decoded_texts = []
        for ids in token_ids:
            text = tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
            decoded_texts.append(text)
        
        return decoded_texts
    
    @staticmethod
    def apply_temperature_sampling(
        logits: torch.Tensor, 
        temperature: float = 1.0
    ) -> torch.Tensor:
        if temperature == 0.0:
            return torch.argmax(logits, dim=-1)
        
        logits = logits / temperature
        probabilities = torch.softmax(logits, dim=-1)
        return torch.multinomial(probabilities, num_samples=1).squeeze(-1)
    
    @staticmethod
    def apply_top_k_filtering(
        logits: torch.Tensor, 
        top_k: int = 50
    ) -> torch.Tensor:
        if top_k > 0:
            values, indices = torch.topk(logits, top_k, dim=-1)
            logits_filtered = torch.full_like(logits, float('-inf'))
            logits_filtered.scatter_(-1, indices, values)
            return logits_filtered
        return logits
    
    @staticmethod
    def apply_top_p_filtering(
        logits: torch.Tensor, 
        top_p: float = 0.9
    ) -> torch.Tensor:
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            # Scatter sorted indices to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
            logits = logits.masked_fill(indices_to_remove, float('-inf'))
        
        return logits
    
    @staticmethod
    def generate_text(
        model: torch.nn.Module,
        tokenizer: GPT2Tokenizer,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
        device: Optional[torch.device] = None
    ) -> str:
        model.eval()
        
        if device is None:
            device = next(model.parameters()).device
        
        # Tokenize input
        input_ids = ModelUtils.prepare_input_ids(prompt, tokenizer, padding=False)
        input_ids = input_ids.to(device)
        
        generated_ids = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Get model predictions
                outputs = model(generated_ids)
                logits = outputs.logits[:, -1, :]  # Get last token logits
                
                # Apply filtering
                if do_sample:
                    if top_k > 0:
                        logits = ModelUtils.apply_top_k_filtering(logits, top_k)
                    if top_p < 1.0:
                        logits = ModelUtils.apply_top_p_filtering(logits, top_p)
                    
                    # Sample next token
                    next_token = ModelUtils.apply_temperature_sampling(logits, temperature)
                else:
                    # Greedy decoding
                    next_token = torch.argmax(logits, dim=-1)
                
                # Append to sequence
                generated_ids = torch.cat([generated_ids, next_token.unsqueeze(-1)], dim=-1)
                
                # Check for EOS token
                if next_token.item() == tokenizer.eos_token_id:
                    break
        
        # Decode generated text
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        # Remove original prompt from generated text
        generated_text = generated_text[len(prompt):].strip()
        
        return generated_text
    
    @staticmethod
    def calculate_perplexity(
        model: torch.nn.Module,
        tokenizer: GPT2Tokenizer,
        text: str,
        device: Optional[torch.device] = None
    ) -> float:
        model.eval()
        
        if device is None:
            device = next(model.parameters()).device
        
        # Tokenize text
        input_ids = ModelUtils.prepare_input_ids(text, tokenizer, padding=False)
        input_ids = input_ids.to(device)
        
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            perplexity = torch.exp(loss)
        
        return perplexity.item()
    
    @staticmethod
    def estimate_model_size(model: torch.nn.Module) -> Dict[str, Any]:
        param_count = sum(p.numel() for p in model.parameters())
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        
        total_size = param_size + buffer_size
        
        return {
            "total_params": param_count,
            "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "model_size_mb": total_size / (1024 * 1024),
            "param_size_mb": param_size / (1024 * 1024),
            "buffer_size_mb": buffer_size / (1024 * 1024)
        }
    
    @staticmethod
    def benchmark_inference_speed(
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        num_runs: int = 100,
        warmup_runs: int = 10
    ) -> Dict[str, float]:
        model.eval()
        device = next(model.parameters()).device
        input_ids = input_ids.to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(input_ids)
        
        # Benchmark
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        import time
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(input_ids)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / num_runs
        throughput = num_runs / total_time
        
        return {
            "avg_inference_time_ms": avg_time * 1000,
            "throughput_samples_per_sec": throughput,
            "total_time_sec": total_time
        }