# Results

This directory contains model outputs and evaluation results.

## Contents

### Model Checkpoints

| Directory | Description |
|-----------|-------------|
| `qwen25_energy_finetuned/` | LoRA adapter weights (~50MB) |
| `qwen25_energy_merged/` | Merged model weights (optional) |

### Evaluation Results

| File | Description |
|------|-------------|
| `comprehensive_evaluation_results.json` | Full evaluation with all metrics |
| `evaluation_results.json` | Basic Q&A evaluation outputs |

## Key Metrics

From `comprehensive_evaluation_results.json`:

| Metric | Value |
|--------|-------|
| **Average Quality Score** | 0.864 |
| **Technical Accuracy** | 0.800 |
| **Coherence** | 0.760 |
| **Hallucination Risk** | 0.000 |
| **Avg Latency** | 37.16s |
| **Throughput** | 13.94 tok/s |

## Loading the Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base + LoRA adapters
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
model = PeftModel.from_pretrained(base_model, "results/qwen25_energy_finetuned")
tokenizer = AutoTokenizer.from_pretrained("results/qwen25_energy_finetuned")
```

See [docs/PERFORMANCE.md](../docs/PERFORMANCE.md) for detailed performance analysis.
