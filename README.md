Energy AI - Petroleum Engineering Assistant

A fine-tuned AI assistant specialized in petroleum engineering, built with Qwen2.5-3B and LoRA fine-tuning.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

  Features

- Fine-tuned LLM: Qwen2.5-3B model fine-tuned on petroleum engineering data using LoRA
- Interactive Chat UI: Beautiful Streamlit interface for Q&A
- Domain Expertise: Specialized in drilling, reservoir engineering, production optimization
- Efficient Training: Uses LoRA for memory-efficient fine-tuning on consumer GPUs (8GB VRAM)

  Project Structure

```
energy-ai/
├── app.py                       Streamlit GUI application
├── finetune_llama.py            Fine-tuning script with LoRA
├── evaluate_model.py            Model evaluation and testing
├── energy_data_finetuning.jsonl  Training dataset (instruction format)
├── energy_data_augmentation.py  Data augmentation utilities
├── requirements.txt             Python dependencies
└── README.md                    This file
```

 🛠️ Installation

 Prerequisites

- Python 3.10+
- CUDA-capable GPU with 8GB+ VRAM
- PyTorch with CUDA support

 Setup

1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/energy-ai.git
cd energy-ai
```

2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate   Windows
 source venv/bin/activate   Linux/Mac
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Install PyTorch with CUDA (adjust for your CUDA version)
```bash
 For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

 For newer GPUs (RTX 40/50 series), use nightly with CUDA 12.8
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

  Usage

 Fine-tuning the Model

1. Prepare your dataset in JSONL format with chat messages:
```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

2. Run fine-tuning:
```bash
python finetune_llama.py
```

The script will:
- Download Qwen2.5-3B-Instruct from Hugging Face
- Apply LoRA adapters for efficient training
- Train on your dataset
- Save adapters to `./qwen25_energy_finetuned/`

 Running the Chat Interface

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

 Evaluating the Model

```bash
python evaluate_model.py
```

This runs predefined test questions and saves results to `evaluation_results.json`.

 ⚙️ Configuration

 Fine-tuning Parameters (finetune_llama.py)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MODEL_NAME` | Qwen/Qwen2.5-3B-Instruct | Base model |
| `MAX_SEQ_LENGTH` | 512 | Maximum sequence length |
| `LORA_R` | 16 | LoRA rank |
| `LORA_ALPHA` | 32 | LoRA alpha |
| `BATCH_SIZE` | 1 | Training batch size |
| `GRADIENT_ACCUMULATION` | 8 | Gradient accumulation steps |
| `LEARNING_RATE` | 2e-4 | Learning rate |
| `EPOCHS` | 1 | Number of training epochs |

 App Settings (Sidebar)

- Max Response Length: 128-512 tokens
- Temperature: 0.1-1.0 (creativity level)
- System Prompt: Customizable AI persona

  Dataset Format

The training data should be in JSONL format with chat-style messages:

```json
{
  "messages": [
    {"role": "system", "content": "You are an expert petroleum engineer..."},
    {"role": "user", "content": "What is permeability anisotropy?"},
    {"role": "assistant", "content": "Permeability anisotropy refers to..."}
  ]
}
```

 🔧 Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 8 GB | 12+ GB |
| RAM | 16 GB | 32 GB |
| Storage | 20 GB | 50 GB |

  Training Results

After fine-tuning, the model shows improved performance on petroleum engineering tasks:

-  Technical terminology understanding
-  Calculation explanations (Darcy equation, PI, etc.)
-  Drilling operations knowledge
-  Reservoir engineering concepts

  Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

  Acknowledgments

- [Qwen](https://github.com/QwenLM/Qwen) for the base model
- [Hugging Face](https://huggingface.co) for transformers and PEFT
- [Streamlit](https://streamlit.io) for the web framework

  Contact

For questions or feedback, please open an issue on GitHub.
