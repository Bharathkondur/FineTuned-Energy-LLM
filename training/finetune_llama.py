"""
Fine-tune Qwen2.5 3B on petroleum engineering dataset using LoRA.
Optimized for RTX 5060 (8GB VRAM) - uses FP16 training without quantization.
Uses manual data loading to avoid pyarrow DLL issues on Windows.
"""
import torch
import json
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType
import os

# Configuration
# Qwen2.5 3B Instruct - high quality open model that fits in 8GB VRAM
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
DATASET_FILE = "energy_data_finetuning.jsonl"
OUTPUT_DIR = "./qwen25_energy_finetuned"
MAX_SEQ_LENGTH = 512  # Reduced for VRAM efficiency
LORA_R = 16  # LoRA rank
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# Training hyperparameters - optimized for 8GB VRAM
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 8  # Effective batch size = 8
EPOCHS = 1  # Start with 1 epoch
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.03
SAVE_STEPS = 100
LOGGING_STEPS = 10


class PetroleumDataset(Dataset):
    """Custom dataset for petroleum engineering Q&A data."""
    
    def __init__(self, file_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        print(f"Loading data from {file_path}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    self.data.append(entry)
                except json.JSONDecodeError:
                    continue
        print(f"Loaded {len(self.data)} examples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        entry = self.data[idx]
        messages = entry.get("messages", [])
        
        # Format using Qwen chat template
        try:
            formatted_text = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=False
            )
        except Exception:
            # Fallback formatting for Qwen
            formatted_text = ""
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "system":
                    formatted_text += f"<|im_start|>system\n{content}<|im_end|>\n"
                elif role == "user":
                    formatted_text += f"<|im_start|>user\n{content}<|im_end|>\n"
                elif role == "assistant":
                    formatted_text += f"<|im_start|>assistant\n{content}<|im_end|>\n"
        
        # Tokenize
        encoding = self.tokenizer(
            formatted_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        
        # For causal LM, labels = input_ids (shifted internally by the model)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone()
        }


def main():
    print("=" * 60)
    print("Qwen2.5 3B Fine-Tuning for Petroleum Engineering")
    print("Using LoRA with FP16 (8GB VRAM optimized)")
    print("=" * 60)
    
    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\n✓ GPU detected: {gpu_name}")
        print(f"✓ VRAM: {gpu_memory:.1f} GB")
        
        # Clear GPU cache
        torch.cuda.empty_cache()
    else:
        print("⚠ No GPU detected! Training will be very slow.")
        return
    
    # Load tokenizer
    print("\n[1/4] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Load model in FP16
    print("[2/4] Loading model in FP16...")
    print(f"   Model: {MODEL_NAME}")
    print("   Note: Qwen2.5 3B requires ~6GB VRAM in FP16 with LoRA.")
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map={"": 0},  # Load directly to GPU 0
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    
    # Enable gradient checkpointing to save VRAM
    model.gradient_checkpointing_enable()
    
    # LoRA config - targeting Qwen2.5 attention and MLP layers
    print("[3/4] Adding LoRA adapters...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load dataset
    print(f"[4/4] Loading dataset from {DATASET_FILE}...")
    train_dataset = PetroleumDataset(DATASET_FILE, tokenizer, MAX_SEQ_LENGTH)
    
    # Configure training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        fp16=True,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=42,
        report_to="none",
        gradient_checkpointing=True,
        max_grad_norm=0.3,
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )
    
    # Estimate training time
    total_steps = (len(train_dataset) // (BATCH_SIZE * GRADIENT_ACCUMULATION)) * EPOCHS
    estimated_time = total_steps * 2  # ~2 seconds per step estimate for 3B model
    hours = estimated_time // 3600
    minutes = (estimated_time % 3600) // 60
    
    print(f"\n{'=' * 60}")
    print(f"Starting training...")
    print(f"   Total steps: {total_steps}")
    print(f"   Estimated time: ~{hours}h {minutes}m")
    print("-" * 60)
    
    # Train!
    trainer.train()
    
    # Save the model
    print("\n" + "=" * 60)
    print("Training complete! Saving model...")
    
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"✓ LoRA adapters saved to: {OUTPUT_DIR}")
    
    print("\n" + "=" * 60)
    print("Fine-tuning complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
