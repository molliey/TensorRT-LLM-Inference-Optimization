import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class GPT2Loader:
    def __init__(self, model_name: str = "gpt2", cache_dir: Optional[str] = None):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.model = None
        self.tokenizer = None
        self.config = None
    
    def load_model(self) -> Tuple[GPT2LMHeadModel, GPT2Tokenizer, GPT2Config]:
        logger.info(f"Loading GPT2 model: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            padding_side="left"
        )
        
        # Add padding token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model configuration
        self.config = GPT2Config.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir
        )
        
        # Load model
        self.model = GPT2LMHeadModel.from_pretrained(
            self.model_name,
            config=self.config,
            cache_dir=self.cache_dir,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Set model to evaluation mode
        self.model.eval()
        
        logger.info(f"Model loaded successfully. Config: {self.config}")
        return self.model, self.tokenizer, self.config
    
    def prepare_for_export(self) -> GPT2LMHeadModel:
        if self.model is None:
            self.load_model()
        
        # Prepare model for ONNX export
        self.model.eval()
        
        # Remove dropout for inference
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
        
        return self.model
    
    def get_model_info(self) -> dict:
        if self.config is None:
            self.load_model()
        
        return {
            "model_name": self.model_name,
            "vocab_size": self.config.vocab_size,
            "n_positions": self.config.n_positions,
            "n_embd": self.config.n_embd,
            "n_layer": self.config.n_layer,
            "n_head": self.config.n_head,
            "intermediate_size": getattr(self.config, 'n_inner', self.config.n_embd * 4),
            "max_position_embeddings": self.config.n_positions
        }