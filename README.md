# Advanced Model Compression for IoT Deployment

## Assignment Description

This project is the Phase 4 deliverable for a generative AI course assignment on deploying language models under edge-device constraints. The goal is to compress a causal language model so it can run on resource-limited hardware while still supporting realistic benchmark evaluation and sector-oriented demonstrations.

## Objectives

The notebook and supporting code are designed to:

1. Evaluate a baseline model on real benchmark datasets.
2. Apply a four-stage compression pipeline: NAS, pruning, distillation, and quantization.
3. Measure compression impact on size, latency, accuracy, and estimated energy use.
4. Simulate Raspberry Pi 4 style deployment conditions on a CPU-only machine.
5. Produce plots, metric tables, checkpoints, and saved model artifacts.

## Overview

We implements a four-stage compression pipeline for deploying a small causal language model on edge-class hardware:

1. Hardware-Aware Neural Architecture Search
2. Structured Pruning
3. Knowledge Distillation
4. Post-Training Quantization

The notebook focuses on real-world sector applications and uses DistilGPT2 as the base model because larger instruction-tuned models were not practical in the available environment.

## Notebook Workflow

The main deliverable is [proj.ipynb](proj.ipynb), which performs the following steps:

1. Loads the base model and tokenizer.
2. Evaluates the baseline on real benchmark data.
3. Runs NAS, pruning, distillation, and quantization.
4. Re-evaluates the quantized model on the same benchmark suite.
5. Saves model artifacts, metrics, and visualizations under `results/`.

## Base Model and Target Profile

- Base model: DistilGPT2
- Parameters: 81,912,576
- Baseline model size: 0.3110 GB
- Target device profile: Raspberry Pi 4 class, 4 GB RAM, CPU-only simulation
- Quantization: 4-bit PTQ with AWQ-style weight quantization
- Saved format: GGUF-compatible model directory

## System Requirements and Dependencies

Recommended environment:

- Python 3.9 or newer
- CPU-only execution is supported and is the notebook default in this workspace
- Internet access for first-time Hugging Face dataset and model downloads
- Sufficient disk space for local model copies and generated results 

Primary dependencies are listed in [requirements.txt](requirements.txt). The notebook uses:

- `torch`
- `transformers`
- `datasets`
- `accelerate`
- `bitsandbytes`
- `auto-gptq`
- `optimum`
- `numpy`
- `pandas`
- `scipy`
- `matplotlib`
- `seaborn`
- `tqdm`
- `ipykernel`
- `ipywidgets`

## Real Data and Benchmarks

The notebook uses real Hugging Face datasets only, and the README keeps the dataset links here for easy access:

- MMLU from [cais/mmlu](https://huggingface.co/datasets/cais/mmlu)
- HellaSwag from [hellaswag](https://huggingface.co/datasets/hellaswag)
- TruthfulQA from [truthful_qa](https://huggingface.co/datasets/truthful_qa)
- WikiText from [wikitext](https://huggingface.co/datasets/wikitext)

Baseline evaluation uses 50 samples per benchmark in the recorded run. The combined baseline accuracy is 37.33%, and the quantized model reaches 30.67% average accuracy.

## Measured Results

| Metric | Baseline | Quantized |
|---|---:|---:|
| Model size | 0.3110 GB | 0.0389 GB |
| Average benchmark accuracy | 37.33% | 30.67% |
| MMLU accuracy | 28.00% | 32.00% |
| HellaSwag accuracy | 34.00% | 22.00% |
| TruthfulQA accuracy | 50.00% | 38.00% |
| Inference latency | 2.41 s | 2.07 s |
| Estimated power profile | 4.2 W | 1.8 W |
| Compression ratio | - | 8.0x |

## Stage Summary

| Stage | Recorded outcome |
|---|---|
| NAS | Best fitness 1.1840, depth ratio 0.80, width ratio 0.80 |
| Pruning | Target sparsity 30%, actual sparsity 0.00% |
| Distillation | Final loss 0.052443 after 1 epoch |
| Quantization | 8.0x compression, 87.5% size reduction |

## Energy and Cost Analysis

The notebook estimates Raspberry Pi 4 style operating costs from measured notebook latency:

- Baseline annual cost per device: $2.82
- Quantized annual cost per device: $1.03
- Estimated savings per device: $1.78 per year
- Estimated energy reduction: 63.3%

The sector-level cost table and plots are written to `results/metrics/` and `results/visualizations/`.

## Installation and Setup

```bash
cd e:\PROJ
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

To open the notebook:

```bash
jupyter notebook proj.ipynb
```

The first execution may download the DistilGPT2 model and the benchmark datasets from Hugging Face.

## Usage Examples and Demonstrations

The notebook demonstrates the full pipeline end to end:

1. Load the base model and tokenizer.
2. Run baseline benchmark evaluation.
3. Execute NAS, pruning, distillation, and quantization.
4. Re-evaluate the compressed model.
5. Save the final model and results artifacts.

Representative code path:

```python
from src.models import ModelManager
from src.config import CompressionConfig
from src.data_loader import DataLoader

config = CompressionConfig()
model_manager = ModelManager(config)
base_model, tokenizer = model_manager.load_base_model()

loader = DataLoader(data_dir="./data")
benchmarks = loader.load_all_benchmarks(num_samples_per_benchmark=50)
```

The notebook also includes sector-oriented demonstrations for healthcare, education, smart cities, agriculture, telecommunications, environment, and sports.

## Known Issues and Limitations

- The model is not instruction-tuned, so outputs are not expected to behave like a chat assistant.
- CPU-only execution is slower than GPU-accelerated training or evaluation.
- The pruning stage did not produce sparsity in the recorded run, so the current results should be treated as a diagnostic baseline rather than a final optimized pruning outcome.
- Quantization improves model size and energy profile, but it does not preserve all benchmark accuracy.
- The notebook depends on Hugging Face availability for first-run downloads.

## License Information

Academic use only. Respect the licenses and usage terms of the base model and the Hugging Face datasets referenced above. This repository is provided for educational purposes as part of the course assignment.

## Supported Sectors

The notebook frames the compression pipeline around these deployments:

- Healthcare
- Education
- Smart Cities
- Environment and Climate
- Agriculture
- Telecommunications
- Sports and Recreation

## Repository Layout

```
PROJ/
├── proj.ipynb
├── src/
├── models/
├── results/
├── requirements.txt
└── README.md
```

## Setup

```bash
cd e:\PROJ
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook proj.ipynb
```

If you want to run the notebook on a clean machine, make sure internet access is available so the Hugging Face datasets can be downloaded on first run.

## Notes

- The notebook is CPU-only by design.
- The model is optimized for efficiency, not conversational reasoning.
- The pruning stage did not produce sparsity in the recorded run, so the README reflects that measured behavior rather than an idealized result.

## Team

- Wahaq Almutairi
- Abdullah Aldwsry
- Meshari Alshammari
- Hossam Baroudi
- Mazen Hamze

Alfaisal University - College of Engineering
