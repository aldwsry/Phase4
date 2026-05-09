"""
Model loading and management utilities
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import Tuple, Optional
from pathlib import Path
from .config import ModelConfig, CompressionConfig

class ModelManager:
    """Manages model loading, saving, and management"""
    
    def __init__(self, config: CompressionConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.loaded_model_name = None
        
    def load_base_model(self, model_name: Optional[str] = None) -> Tuple:
        """Load base model and tokenizer"""
        requested_model_name = model_name or self.config.model.base_model
        candidate_model_names = [requested_model_name]

        # Fall back to a smaller causal LM on CPU-only environments.
        if not torch.cuda.is_available() and requested_model_name != "distilgpt2":
            candidate_model_names.append("distilgpt2")
        
        last_error = None
        for candidate_model_name in candidate_model_names:
            try:
                if self.config.verbose:
                    print(f"Loading base model: {candidate_model_name}")
                    print(f"Device: {self.device}")

                # Load tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    candidate_model_name,
                    trust_remote_code=True,
                    padding_side="left"
                )

                # Add padding token if missing
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

                # Load model with optimizations
                model_kwargs = {
                    "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
                    "device_map": "auto" if torch.cuda.is_available() else None,
                    "trust_remote_code": True,
                    "low_cpu_mem_usage": True,
                }

                self.model = AutoModelForCausalLM.from_pretrained(
                    candidate_model_name,
                    **model_kwargs
                )

                # Move to device if not using device_map
                if not torch.cuda.is_available():
                    self.model = self.model.to(self.device)

                self.loaded_model_name = candidate_model_name

                if self.config.verbose:
                    print("Model loaded successfully")
                    print(f"Loaded model: {self.loaded_model_name}")
                    print(f"Model size: {self._get_model_size(self.model):.2f} GB")

                return self.model, self.tokenizer

            except Exception as exc:
                last_error = exc
                if self.config.verbose and candidate_model_name != candidate_model_names[-1]:
                    print(f"Warning: failed to load {candidate_model_name}: {exc}")
                    print("Falling back to a smaller CPU-friendly model...")

        raise RuntimeError(f"Unable to load a base model. Last error: {last_error}")
    
    def load_compressed_model(self, model_path: str, bits: int = 4) -> Tuple:
        """Load quantized/compressed model"""
        if self.config.verbose:
            print(f"Loading compressed model from: {model_path}")
        
        # Load quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=bits == 4,
            load_in_8bit=bits == 8,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        ) if torch.cuda.is_available() else None
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config if bnb_config else None,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        self.model = model
        self.tokenizer = tokenizer
        
        if self.config.verbose:
            print(f"Compressed model loaded successfully")
        
        return model, tokenizer
    
    def save_model(self, model_path: str) -> None:
        """Save model and tokenizer"""
        if self.model is None:
            raise ValueError("No model loaded. Load or train a model first.")
        
        output_path = Path(model_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if self.config.verbose:
            print(f"Saving model to: {model_path}")
        
        self.model.save_pretrained(str(output_path))
        self.tokenizer.save_pretrained(str(output_path))
        
        if self.config.verbose:
            print(f"Model saved successfully")
    
    @staticmethod
    def _get_model_size(model: torch.nn.Module) -> float:
        """Calculate model size in GB"""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb / 1024
    
    def get_model_size(self) -> float:
        """Get current model size in GB"""
        if self.model is None:
            raise ValueError("No model loaded")
        return self._get_model_size(self.model)
    
    def get_num_parameters(self) -> int:
        """Get total number of parameters"""
        if self.model is None:
            raise ValueError("No model loaded")
        return sum(p.numel() for p in self.model.parameters())
    
    def get_model_info(self) -> dict:
        """Get model information"""
        if self.model is None:
            raise ValueError("No model loaded")
        
        return {
            "num_parameters": self.get_num_parameters(),
            "model_size_gb": self.get_model_size(),
            "vocab_size": self.tokenizer.vocab_size if self.tokenizer else None,
            "max_position_embeddings": getattr(self.model.config, "max_position_embeddings", None),
            "hidden_size": getattr(self.model.config, "hidden_size", None),
            "num_hidden_layers": getattr(self.model.config, "num_hidden_layers", None),
            "num_attention_heads": getattr(self.model.config, "num_attention_heads", None),
        }
