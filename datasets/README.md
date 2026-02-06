# Datasets

This directory contains all training and evaluation data for the Energy AI project.

## Files

| File | Size | Description |
|------|------|-------------|
| `oil_gas_data.csv` | 159 KB | Raw source data from petroleum engineering domain |
| `oil_gas_data.jsonl` | 168 KB | Converted JSONL format |
| `energy_data_expanded.jsonl` | 499 KB | Augmented dataset with paraphrasing |
| `energy_data_instruct.jsonl` | 967 KB | Instruction-formatted data |
| `energy_data_finetuning.jsonl` | 5.76 MB | Final training dataset (chat format) |
| `energy_data_completion.jsonl` | 662 KB | Completion format (alternative) |
| `energy_data_pretrain.jsonl` | 586 KB | Pre-training format |

## Data Format

### Training Data (Chat Format)

```json
{
  "messages": [
    {"role": "system", "content": "You are an expert petroleum engineer..."},
    {"role": "user", "content": "What is permeability anisotropy?"},
    {"role": "assistant", "content": "Permeability anisotropy refers to..."}
  ]
}
```

## Statistics

- **Total Examples**: ~12,000
- **Token Distribution**: Mean 180, Max 512
- **Topics**: Drilling, Reservoir, Production, Completion, Safety, Environmental

## Usage

```python
import json

# Load training data
with open('datasets/energy_data_finetuning.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]
```

See [docs/DATASET.md](../docs/DATASET.md) for complete documentation.
