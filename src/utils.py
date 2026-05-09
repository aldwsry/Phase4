"""
Utility functions for compression pipeline
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional
import time
import psutil
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_seed(seed: int) -> None:
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def get_gpu_memory_usage() -> Dict[str, float]:
    """Get GPU memory usage in GB"""
    if not torch.cuda.is_available():
        return {"allocated": 0.0, "reserved": 0.0, "total": 0.0}
    
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    return {
        "allocated": allocated,
        "reserved": reserved,
        "total": total,
    }

def get_cpu_memory_usage() -> Dict[str, float]:
    """Get CPU memory usage in GB"""
    process = psutil.Process()
    mem_info = process.memory_info()
    
    return {
        "rss": mem_info.rss / 1024**3,  # Resident set size
        "vms": mem_info.vms / 1024**3,  # Virtual memory size
    }

def measure_latency(model, tokenizer, inputs: str, num_runs: int = 3) -> Dict[str, float]:
    """Measure model inference latency"""
    model.eval()
    
    # Warmup
    with torch.no_grad():
        inputs_encoded = tokenizer(inputs, return_tensors="pt").to(model.device)
        _ = model(**inputs_encoded)
    
    # Measure latency
    latencies = []
    for _ in range(num_runs):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        with torch.no_grad():
            inputs_encoded = tokenizer(inputs, return_tensors="pt").to(model.device)
            _ = model(**inputs_encoded)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        latencies.append(time.time() - start_time)
    
    return {
        "mean": np.mean(latencies),
        "std": np.std(latencies),
        "min": np.min(latencies),
        "max": np.max(latencies),
    }

def measure_throughput(model, tokenizer, num_tokens: int = 100) -> float:
    """Measure tokens per second"""
    model.eval()
    
    # Generate tokens and measure throughput
    input_text = "The quick brown fox"
    inputs_encoded = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs_encoded,
            max_new_tokens=num_tokens,
            do_sample=False,
        )
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed_time = time.time() - start_time
    
    throughput = num_tokens / elapsed_time
    return throughput

class MetricsTracker:
    """Track training and evaluation metrics"""
    
    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}
    
    def add(self, key: str, value: float) -> None:
        """Add metric value"""
        if key not in self.metrics:
            self.metrics[key] = []
        self.metrics[key].append(value)
    
    def get(self, key: str) -> List[float]:
        """Get metric values"""
        return self.metrics.get(key, [])
    
    def get_last(self, key: str) -> Optional[float]:
        """Get last metric value"""
        values = self.metrics.get(key, [])
        return values[-1] if values else None
    
    def get_mean(self, key: str) -> float:
        """Get mean metric value"""
        values = self.metrics.get(key, [])
        return np.mean(values) if values else 0.0
    
    def get_all(self) -> Dict[str, List[float]]:
        """Get all metrics"""
        return self.metrics.copy()
    
    def to_dict(self) -> Dict[str, Dict[str, float]]:
        """Convert to dictionary with statistics"""
        result = {}
        for key, values in self.metrics.items():
            result[key] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "last": values[-1] if values else None,
            }
        return result

def calculate_model_sparsity(model: torch.nn.Module) -> float:
    """Calculate overall sparsity (percentage of zero weights)"""
    total_params = 0
    zero_params = 0
    
    for param in model.parameters():
        total_params += param.numel()
        zero_params += (param == 0).sum().item()
    
    return (zero_params / total_params) * 100 if total_params > 0 else 0.0

def calculate_layer_sparsity(model: torch.nn.Module) -> Dict[str, float]:
    """Calculate sparsity per layer"""
    sparsity_by_layer = {}
    
    for name, param in model.named_parameters():
        if param.dim() > 1:  # Skip bias and layer norm
            zero_params = (param == 0).sum().item()
            total_params = param.numel()
            sparsity = (zero_params / total_params) * 100 if total_params > 0 else 0.0
            sparsity_by_layer[name] = sparsity
    
    return sparsity_by_layer

def get_attention_patterns(model: torch.nn.Module) -> List[str]:
    """Get attention head configuration"""
    config = model.config
    patterns = []
    
    if hasattr(config, 'num_attention_heads'):
        patterns.append(f"Attention Heads: {config.num_attention_heads}")
    if hasattr(config, 'num_key_value_heads'):
        patterns.append(f"Key-Value Heads: {config.num_key_value_heads}")
    if hasattr(config, 'attention_dropout'):
        patterns.append(f"Attention Dropout: {config.attention_dropout}")
    
    return patterns

class Timer:
    """Context manager for timing operations"""
    
    def __init__(self, name: str = "Operation", verbose: bool = True):
        self.name = name
        self.verbose = verbose
        self.start_time = None
        self.elapsed = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.time() - self.start_time
        if self.verbose:
            print(f"{self.name} took {self.elapsed:.2f} seconds")
