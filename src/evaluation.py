"""
Evaluation metrics and benchmarking
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics"""
    accuracy: float
    perplexity: float
    memory_mb: float
    throughput_tokens_per_sec: float
    latency_ms: float
    energy_joules_per_token: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'accuracy': self.accuracy,
            'perplexity': self.perplexity,
            'memory_mb': self.memory_mb,
            'throughput_tps': self.throughput_tokens_per_sec,
            'latency_ms': self.latency_ms,
            'energy_j_per_token': self.energy_joules_per_token,
        }

class ModelEvaluator:
    """Evaluate model performance"""
    
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
    def compute_perplexity(self, texts: List[str], batch_size: int = 8) -> float:
        """Compute perplexity on text samples"""
        self.model.eval()
        
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                # Tokenize
                encodings = self.tokenizer(
                    batch_texts,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=encodings['input_ids'],
                    attention_mask=encodings['attention_mask'],
                    labels=encodings['input_ids']
                )
                
                loss = outputs.loss
                total_loss += loss.item() * encodings['input_ids'].numel()
                total_tokens += encodings['input_ids'].numel()
        
        mean_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(mean_loss)).item()
        
        return perplexity
    
    def measure_inference_speed(self, prompt: str, num_tokens: int = 100) -> Tuple[float, float]:
        """Measure inference speed (throughput and latency)"""
        self.model.eval()
        
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        # Warmup
        with torch.no_grad():
            self.model.generate(input_ids, max_new_tokens=10)
        
        # Measure
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=num_tokens,
                do_sample=False
            )
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.time() - start_time
        
        throughput = num_tokens / elapsed  # tokens per second
        latency = elapsed * 1000 / num_tokens  # milliseconds per token
        
        return throughput, latency
    
    def get_model_memory(self) -> float:
        """Get model memory usage in MB"""
        total_memory = 0
        
        for param in self.model.parameters():
            total_memory += param.data.element_size() * param.data.nelement()
        
        for buffer in self.model.buffers():
            total_memory += buffer.data.element_size() * buffer.data.nelement()
        
        return total_memory / (1024 ** 2)  # Convert to MB
    
    def evaluate(self, 
                 test_texts: List[str],
                 prompt: str = "The quick brown fox",
                 num_gen_tokens: int = 100,
                 batch_size: int = 8) -> EvaluationMetrics:
        """Run full evaluation"""
        
        # Compute perplexity
        perplexity = self.compute_perplexity(test_texts, batch_size)
        
        # Measure inference speed
        throughput, latency = self.measure_inference_speed(prompt, num_gen_tokens)
        
        # Get memory
        memory_mb = self.get_model_memory()
        
        # Placeholder accuracy (would use benchmark)
        accuracy = 0.0
        
        return EvaluationMetrics(
            accuracy=accuracy,
            perplexity=perplexity,
            memory_mb=memory_mb,
            throughput_tokens_per_sec=throughput,
            latency_ms=latency,
        )
    
    def evaluate_with_benchmarks(self, 
                                benchmark_names: List[str] = None,
                                num_samples: int = 100) -> Dict[str, Dict]:
        """
        Evaluate using standard benchmarks
        
        Args:
            benchmark_names: List of benchmarks ('mmlu', 'hellaswag', 'truthfulqa')
            num_samples: Number of samples per benchmark
        
        Returns:
            Dictionary with benchmark results
        """
        try:
            from .data_loader import DatasetEvaluator
        except ImportError:
            logger.warning("DatasetEvaluator not available")
            return {}
        
        benchmark_names = benchmark_names or ['mmlu', 'hellaswag', 'truthfulqa']
        
        logger.info(f"Evaluating on benchmarks: {benchmark_names}")
        
        evaluator = DatasetEvaluator(self.model, self.tokenizer, 
                                     device=self.device)
        
        if len(benchmark_names) == 1:
            benchmark = benchmark_names[0]
            if benchmark == 'mmlu':
                return {'mmlu': evaluator.evaluate_mmlu(num_samples)}
            elif benchmark == 'hellaswag':
                return {'hellaswag': evaluator.evaluate_hellaswag(num_samples)}
            elif benchmark == 'truthfulqa':
                return {'truthfulqa': evaluator.evaluate_truthfulqa(num_samples)}
        else:
            return evaluator.evaluate_all(num_samples)
    
    def evaluate_with_datasets(self, 
                              dataset_names: List[str] = None) -> Dict[str, Dict]:
        """
        Load and evaluate on full datasets
        
        Args:
            dataset_names: Datasets to use ('mmlu', 'hellaswag', 'truthfulqa', 'wikitext')
        
        Returns:
            Dictionary with evaluation results
        """
        try:
            from .data_loader import DataLoader
        except ImportError:
            logger.warning("DataLoader not available")
            return {}
        
        dataset_names = dataset_names or ['mmlu', 'hellaswag', 'truthfulqa']
        
        logger.info(f"Loading datasets: {dataset_names}")
        
        loader = DataLoader()
        results = {}
        
        if 'mmlu' in dataset_names:
            logger.info("Loading MMLU dataset...")
            results['mmlu'] = loader.load_mmlu()
        
        if 'hellaswag' in dataset_names:
            logger.info("Loading HellaSwag dataset...")
            results['hellaswag'] = loader.load_hellaswag()
        
        if 'truthfulqa' in dataset_names:
            logger.info("Loading TruthfulQA dataset...")
            results['truthfulqa'] = loader.load_truthfulqa()
        
        if 'wikitext' in dataset_names:
            logger.info("Loading WikiText dataset...")
            results['wikitext'] = loader.load_wikitext()
        
        logger.info(f"✓ Loaded {len(results)} datasets")
        return results

class ComparisonAnalyzer:
    """Analyze and compare compression results"""
    
    @staticmethod
    def calculate_compression_ratio(original_size_mb: float, compressed_size_mb: float) -> float:
        """Calculate compression ratio"""
        return original_size_mb / compressed_size_mb if compressed_size_mb > 0 else 0
    
    @staticmethod
    def calculate_speedup(baseline_latency: float, compressed_latency: float) -> float:
        """Calculate speedup"""
        return baseline_latency / compressed_latency if compressed_latency > 0 else 0
    
    @staticmethod
    def calculate_accuracy_retention(baseline_acc: float, compressed_acc: float) -> float:
        """Calculate accuracy retention percentage"""
        if baseline_acc == 0:
            return 0
        return (compressed_acc / baseline_acc) * 100
    
    @staticmethod
    def analyze_pareto_frontier(results: List[Dict]) -> List[Dict]:
        """Find Pareto optimal results"""
        # Maximize accuracy and speedup, minimize latency
        pareto_points = []
        
        for i, result in enumerate(results):
            is_dominated = False
            for j, other in enumerate(results):
                if i != j:
                    # Check if other dominates this result
                    if (other['accuracy'] >= result['accuracy'] and
                        other['speedup'] >= result['speedup'] and
                        other['latency_ms'] <= result['latency_ms']):
                        is_dominated = True
                        break
            
            if not is_dominated:
                pareto_points.append(result)
        
        return pareto_points
