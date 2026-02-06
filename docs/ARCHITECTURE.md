# System Architecture

## Overview

This document provides a detailed technical overview of the **Energy AI - Petroleum Engineering Assistant** system architecture. The project implements a domain-specific LLM using Parameter-Efficient Fine-Tuning (PEFT) with LoRA adapters.

---

## 🏗️ High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ENERGY AI SYSTEM ARCHITECTURE                        │
└─────────────────────────────────────────────────────────────────────────────┘

                           ┌──────────────────────┐
                           │   📚 RAW DATA        │
                           │   oil_gas_data.csv   │
                           │   (159 KB, 500+ rows)│
                           └──────────┬───────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DATA PROCESSING PIPELINE                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌───────────────┐    ┌──────────────────┐    ┌─────────────────────────┐   │
│  │ CSV Import    │───▶│ Data Augmentation│───▶│ Instruction Formatting  │   │
│  │ (load_data.py)│    │ (energy_data_    │    │ (prepare_instruct_      │   │
│  │               │    │  augmentation.py)│    │  data.py)               │   │
│  └───────────────┘    └──────────────────┘    └───────────┬─────────────┘   │
│                                                           │                  │
│  Output: oil_gas_data.jsonl       Output: expanded.jsonl  │                  │
│  (168 KB)                         (499 KB)                ▼                  │
│                                              ┌─────────────────────────┐     │
│                                              │ Chat Format Dataset     │     │
│                                              │ energy_data_finetuning  │     │
│                                              │ .jsonl (5.76 MB)        │     │
│                                              │ ~12,000 examples        │     │
│                                              └─────────────────────────┘     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          FINE-TUNING PIPELINE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────┐         ┌─────────────────────────────────────┐    │
│  │ 🤖 BASE MODEL       │         │       LoRA CONFIGURATION            │    │
│  │                     │         │                                     │    │
│  │ Qwen/Qwen2.5-3B-    │◀───────▶│  • Rank (r): 16                     │    │
│  │ Instruct            │         │  • Alpha: 32                        │    │
│  │                     │         │  • Dropout: 0.05                    │    │
│  │ • 3 Billion params  │         │  • Target Modules:                  │    │
│  │ • FP16 precision    │         │    - q_proj, k_proj, v_proj, o_proj │    │
│  │ • ~6GB VRAM         │         │    - gate_proj, up_proj, down_proj  │    │
│  └─────────────────────┘         └─────────────────────────────────────┘    │
│            │                                                                 │
│            ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    TRAINING CONFIGURATION                            │    │
│  │  ┌─────────────────────────────────────────────────────────────┐    │    │
│  │  │ • Batch Size: 1 (per device)                                │    │    │
│  │  │ • Gradient Accumulation: 8 (effective batch = 8)            │    │    │
│  │  │ • Learning Rate: 2e-4                                       │    │    │
│  │  │ • Scheduler: Cosine                                         │    │    │
│  │  │ • Warmup Ratio: 3%                                          │    │    │
│  │  │ • Max Sequence Length: 512 tokens                           │    │    │
│  │  │ • Epochs: 1                                                 │    │    │
│  │  │ • FP16 Mixed Precision: ✓                                   │    │    │
│  │  │ • Gradient Checkpointing: ✓                                 │    │    │
│  │  └─────────────────────────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│            │                                                                 │
│            ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    OUTPUT: FINE-TUNED MODEL                          │    │
│  │  ./qwen25_energy_finetuned/                                          │    │
│  │  ├── adapter_config.json      (LoRA configuration)                   │    │
│  │  ├── adapter_model.safetensors (LoRA weights, ~50MB)                 │    │
│  │  ├── tokenizer.json                                                  │    │
│  │  ├── tokenizer_config.json                                           │    │
│  │  └── special_tokens_map.json                                         │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          INFERENCE PIPELINE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐     │
│  │ 👤 USER INPUT   │    │  💻 STREAMLIT    │    │  🤖 MODEL INFERENCE │     │
│  │                 │───▶│  INTERFACE       │───▶│                     │     │
│  │ "What is        │    │  (app.py)        │    │ Base Model +        │     │
│  │  permeability?" │    │                  │    │ LoRA Adapters       │     │
│  └─────────────────┘    │  • Chat history  │    │                     │     │
│                         │  • Temperature   │    │ Temperature: 0.5    │     │
│                         │  • Max tokens    │    │ Top-K: 50           │     │
│                         │  • System prompt │    │ Rep. penalty: 1.1   │     │
│                         └──────────────────┘    └──────────┬──────────┘     │
│                                                            │                 │
│                                                            ▼                 │
│                                              ┌─────────────────────────┐     │
│                                              │ 📝 RESPONSE OUTPUT      │     │
│                                              │                         │     │
│                                              │ • Technical explanation │     │
│                                              │ • Equations/formulas    │     │
│                                              │ • Industry standards    │     │
│                                              │ • Practical advice      │     │
│                                              └─────────────────────────┘     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          EVALUATION PIPELINE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    EVALUATION METRICS                                │    │
│  │                                                                      │    │
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────────────┐    │    │
│  │  │ Quality       │  │ Performance   │  │ Safety & Reliability  │    │    │
│  │  │ Metrics       │  │ Metrics       │  │ Metrics               │    │    │
│  │  │               │  │               │  │                       │    │    │
│  │  │ • Technical   │  │ • Latency     │  │ • Hallucination       │    │    │
│  │  │   Accuracy    │  │   (seconds)   │  │   Risk Score          │    │    │
│  │  │ • Coherence   │  │ • Throughput  │  │ • Standards           │    │    │
│  │  │   Score       │  │   (tok/sec)   │  │   Citation Rate       │    │    │
│  │  │ • Relevance   │  │ • Token       │  │ • Factual             │    │    │
│  │  │   Score       │  │   Efficiency  │  │   Consistency         │    │    │
│  │  │ • Complete-   │  │ • Memory      │  │                       │    │    │
│  │  │   ness        │  │   Usage (GB)  │  │                       │    │    │
│  │  └───────────────┘  └───────────────┘  └───────────────────────┘    │    │
│  │                                                                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  Output: comprehensive_evaluation_results.json                               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Component Details

### 1. Data Processing Pipeline

| Stage | Script | Input | Output | Description |
|-------|--------|-------|--------|-------------|
| Import | `load_data.py` | `oil_gas_data.csv` | `oil_gas_data.jsonl` | CSV to JSONL conversion |
| Augmentation | `energy_data_augmentation.py` | JSONL | `energy_data_expanded.jsonl` | Synonym replacement, paraphrasing |
| Formatting | `prepare_instruct_data.py` | Expanded JSONL | `energy_data_finetuning.jsonl` | Convert to chat format |

### 2. Model Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Qwen2.5-3B-Instruct                          │
├─────────────────────────────────────────────────────────────────┤
│  Parameters: 3,000,000,000 (3B)                                 │
│  Architecture: Transformer Decoder-Only                         │
│  Context Length: 32,768 tokens (max)                            │
│  Vocabulary Size: 151,936 tokens                                │
├─────────────────────────────────────────────────────────────────┤
│                       + LoRA Adapters                           │
├─────────────────────────────────────────────────────────────────┤
│  Trainable Parameters: ~16M (0.5% of total)                     │
│  Adapter Size: ~50MB                                            │
│  Target Layers: Attention + MLP projections                     │
└─────────────────────────────────────────────────────────────────┘
```

### 3. Training Infrastructure

| Resource | Minimum | Recommended | Used |
|----------|---------|-------------|------|
| GPU VRAM | 8 GB | 12+ GB | RTX 5060 (8GB) |
| RAM | 16 GB | 32 GB | 16 GB |
| Storage | 20 GB | 50 GB | 25 GB |
| Training Time | ~2h | ~1h | ~90 min |

---

## 🔄 Data Flow

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│              TOKENIZATION & ENCODING                     │
│  • Apply chat template (system + user + assistant)       │
│  • Truncate to max_length (512 tokens)                   │
│  • Convert to tensor format                              │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│                  MODEL FORWARD PASS                      │
│  • Load base weights (Qwen2.5-3B) to GPU                 │
│  • Apply LoRA adapter transformations                    │
│  • Generate token-by-token (autoregressive)              │
│  • Apply sampling (temperature=0.5, top_k=50)            │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│               RESPONSE DECODING                          │
│  • Decode generated tokens                               │
│  • Strip special tokens                                  │
│  • Format response for display                           │
└─────────────────────────────────────────────────────────┘
    │
    ▼
Expert Response on Petroleum Engineering
```

---

## 📁 File Structure

```
energy-ai/
├── 📂 Data Files
│   ├── oil_gas_data.csv              # Raw source data
│   ├── oil_gas_data.jsonl            # Converted JSONL
│   ├── energy_data_expanded.jsonl    # Augmented dataset
│   ├── energy_data_instruct.jsonl    # Instruction format
│   ├── energy_data_finetuning.jsonl  # Final training data
│   └── energy_data_completion.jsonl  # Completion format (alternative)
│
├── 📂 Training
│   ├── finetune_llama.py             # Main fine-tuning script
│   ├── prepare_instruct_data.py      # Data preparation
│   └── energy_data_augmentation.py   # Data augmentation
│
├── 📂 Evaluation
│   ├── evaluate_model.py             # Basic evaluation
│   ├── evaluation_framework.py       # Comprehensive metrics
│   └── evaluation_results.json       # Evaluation outputs
│
├── 📂 Inference
│   ├── app.py                        # Streamlit GUI
│   └── generate_responses.py         # Batch generation
│
├── 📂 Model Outputs
│   ├── qwen25_energy_finetuned/      # LoRA adapters
│   └── qwen25_energy_merged/         # Merged weights (optional)
│
└── 📂 Documentation
    ├── README.md                     # Project overview
    ├── docs/ARCHITECTURE.md          # This file
    ├── docs/DATASET.md               # Dataset documentation
    └── docs/PERFORMANCE.md           # Performance analysis
```

---

## 🔧 Technology Stack

| Category | Technology | Version |
|----------|------------|---------|
| **Language** | Python | 3.10+ |
| **Deep Learning** | PyTorch | 2.0+ |
| **Model Framework** | Hugging Face Transformers | 4.36+ |
| **PEFT** | PEFT (LoRA) | 0.7+ |
| **UI** | Streamlit | 1.30+ |
| **GPU** | CUDA | 12.1+ |

---

## 🔒 Security Considerations

1. **Model Weights**: Base model downloaded from Hugging Face (verified source)
2. **Training Data**: No PII in petroleum engineering dataset
3. **Inference**: Local deployment only (no external API calls during inference)
4. **Dependencies**: All from PyPI with version pinning

---

## 📈 Scalability Path

For production deployment, consider:

1. **Quantization**: Apply GPTQ/AWQ for 4-8x model size reduction
2. **Batching**: Implement dynamic batching for multi-user scenarios
3. **Caching**: Add KV-cache optimization for faster inference
4. **API Layer**: Wrap with FastAPI for REST endpoint access
5. **Monitoring**: Add Prometheus/Grafana for latency tracking
