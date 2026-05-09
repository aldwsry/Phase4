"""
Configuration module for compression pipeline
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import yaml
import json

@dataclass
class HardwareConfig:
    """Hardware configuration for target device"""
    device_name: str = "raspberry_pi_4"
    memory_limit_gb: float = 4.0
    num_cores: int = 4
    frequency_mhz: int = 1500
    has_gpu: bool = False
    
@dataclass
class ModelConfig:
    """Base model configuration"""
    base_model: str = "distilgpt2"
    model_type: str = "gpt2"
    max_seq_length: int = 1024
    vocab_size: int = 50257
    
@dataclass
class StageNASConfig:
    """Stage 1: Neural Architecture Search configuration"""
    enabled: bool = True
    search_type: str = "genetic"  # 'genetic', 'evolutionary', 'random'
    population_size: int = 32
    generations: int = 50
    mutation_rate: float = 0.3
    crossover_rate: float = 0.7
    target_speedup: float = 1.3
    hardware_constraint: str = "latency"  # 'latency', 'memory', 'energy'
    search_depth_ratios: list = field(default_factory=lambda: [0.8, 0.9, 1.0])
    search_width_ratios: list = field(default_factory=lambda: [0.8, 0.9, 1.0])

@dataclass
class StagePruningConfig:
    """Stage 2: Structured Pruning configuration"""
    enabled: bool = True
    method: str = "structured"  # 'structured', 'unstructured', 'semi_structured'
    algorithm: str = "thanos"  # 'thanos', 'wanda', 'sparsegpt'
    target_sparsity: float = 0.3
    pruning_ratio: float = 0.3
    prune_head: bool = True
    prune_mlp: bool = True
    prune_layers: bool = True
    layer_reduction_ratio: float = 0.2
    recalibration_samples: int = 128

@dataclass
class StageDistillationConfig:
    """Stage 3: Knowledge Distillation configuration"""
    enabled: bool = True
    teacher_model: str = "distilgpt2"
    student_model: Optional[str] = None
    distillation_type: str = "logit"  # 'logit', 'feature', 'self'
    temperature: float = 3.0
    alpha: float = 0.5  # weight of KD loss vs CE loss
    learning_rate: float = 5e-5
    batch_size: int = 16
    num_epochs: int = 3
    warmup_steps: int = 500
    eval_steps: int = 500
    num_distill_samples: int = 10000

@dataclass
class StageQuantizationConfig:
    """Stage 4: Post-Training Quantization configuration"""
    enabled: bool = True
    method: str = "ptq"  # 'ptq', 'qat'
    bits: int = 4  # 1, 2, 3, 4, 8
    quantization_type: str = "symmetric"
    algorithm: str = "awq"  # 'awq', 'gptq', 'smoothquant'
    calib_samples: int = 128
    device: str = "cpu"
    save_format: str = "gguf"  # 'gguf', 'onnx', 'torch'
    mixed_precision: bool = False

@dataclass
class DataConfig:
    """Data configuration for benchmarks and training"""
    data_dir: str = "./data"
    raw_data_dir: str = "./data/raw"
    processed_data_dir: str = "./data/processed"
    use_mock_data: bool = False  # Use mock data when real datasets unavailable
    num_samples_per_benchmark: int = 100
    distillation_data_samples: int = 10000
    calibration_samples: int = 128

@dataclass
class EvaluationConfig:
    """Evaluation configuration"""
    benchmarks: list = field(default_factory=lambda: ["mmlu", "hellaswag", "truthfulqa"])
    num_shots: Dict[str, int] = field(default_factory=lambda: {
        "mmlu": 5,
        "hellaswag": 0,
        "truthfulqa": 0
    })
    batch_size: int = 32
    num_samples: Optional[int] = None  # None = use all
    compute_energy: bool = True
    energy_device: str = "raspberry_pi_4"
    save_results: bool = True
    result_dir: str = "./results/"

@dataclass
class CompressionConfig:
    """Master configuration for compression pipeline"""
    
    # Core configs
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    
    # Data configuration
    data: DataConfig = field(default_factory=DataConfig)
    
    # Stage configurations
    stage1_nas: StageNASConfig = field(default_factory=StageNASConfig)
    stage2_pruning: StagePruningConfig = field(default_factory=StagePruningConfig)
    stage3_distillation: StageDistillationConfig = field(default_factory=StageDistillationConfig)
    stage4_quantization: StageQuantizationConfig = field(default_factory=StageQuantizationConfig)
    
    # Evaluation
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    # Pipeline settings
    pipeline_stages: list = field(default_factory=lambda: [1, 2, 3, 4])
    output_dir: str = "./results/models/"
    checkpoint_dir: str = "./results/checkpoints/"
    device: str = "cuda"
    seed: int = 42
    verbose: bool = True
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'CompressionConfig':
        """Load configuration from YAML file"""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'CompressionConfig':
        """Load configuration from dictionary"""
        # TODO: Implement proper deserialization
        return cls()
    
    def to_yaml(self, output_path: str) -> None:
        """Save configuration to YAML file"""
        config_dict = {
            'hardware': self.hardware.__dict__,
            'model': self.model.__dict__,
            'stage1_nas': self.stage1_nas.__dict__,
            'stage2_pruning': self.stage2_pruning.__dict__,
            'stage3_distillation': self.stage3_distillation.__dict__,
            'stage4_quantization': self.stage4_quantization.__dict__,
            'evaluation': self.evaluation.__dict__,
        }
        with open(output_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'hardware': self.hardware.__dict__,
            'model': self.model.__dict__,
            'stage1_nas': self.stage1_nas.__dict__,
            'stage2_pruning': self.stage2_pruning.__dict__,
            'stage3_distillation': self.stage3_distillation.__dict__,
            'stage4_quantization': self.stage4_quantization.__dict__,
            'evaluation': self.evaluation.__dict__,
        }
