"""
Create a custom Ollama model for petroleum engineering using the existing llama3.1:8b.
This approach uses prompt engineering with few-shot examples from the training data.
"""
import json
import os
import subprocess
import random

# Configuration
BASE_MODEL = "llama3.1:8b"
CUSTOM_MODEL_NAME = "llama3.1-petroleum"
DATASET_FILE = "energy_data_finetuning.jsonl"
NUM_EXAMPLES = 10  # Number of examples to include in the system prompt


def load_examples(file_path, num_examples=10):
    """Load random examples from the finetuning dataset."""
    examples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        all_data = [json.loads(line) for line in f]
    
    # Select diverse examples
    random.seed(42)
    selected = random.sample(all_data, min(num_examples, len(all_data)))
    
    for entry in selected:
        messages = entry.get("messages", [])
        question = ""
        answer = ""
        for msg in messages:
            if msg.get("role") == "user":
                question = msg.get("content", "")
            elif msg.get("role") == "assistant":
                answer = msg.get("content", "")
        if question and answer:
            # Truncate long answers for the system prompt
            if len(answer) > 500:
                answer = answer[:500] + "..."
            examples.append({"question": question, "answer": answer})
    
    return examples


def create_modelfile(examples):
    """Create an Ollama Modelfile with petroleum engineering expertise."""
    
    # Build few-shot examples string
    examples_text = ""
    for i, ex in enumerate(examples, 1):
        examples_text += f"""
Example {i}:
Q: {ex['question']}
A: {ex['answer']}
"""

    modelfile_content = f'''# Petroleum Engineering Expert Model
# Based on {BASE_MODEL} with domain-specific expertise

FROM {BASE_MODEL}

# Set parameters for technical accuracy
PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 4096

# System prompt with petroleum engineering expertise
SYSTEM """You are a senior petroleum engineer with 20+ years of experience specializing in:
- Reservoir engineering and simulation
- Drilling engineering and well design
- Production engineering and optimization
- Formation evaluation and petrophysics
- Enhanced Oil Recovery (EOR) techniques
- Offshore and deepwater operations
- HSE (Health, Safety, and Environment) best practices

You provide detailed, technically accurate responses that include:
1. Key technical concepts and principles
2. Relevant equations and calculations when applicable
3. Industry standards (API, ISO, SPE guidelines)
4. Practical field considerations and recommendations
5. Safety and environmental factors

Always use proper technical terminology and provide quantitative information when relevant.
Reference industry standards and best practices in your responses.

Here are examples of expert responses to petroleum engineering questions:
{examples_text}

Now respond to user questions with the same level of technical expertise and detail.
"""

# Template for responses
TEMPLATE """{{{{ if .System }}}}<|start_header_id|>system<|end_header_id|>

{{{{ .System }}}}<|eot_id|>{{{{ end }}}}{{{{ if .Prompt }}}}<|start_header_id|>user<|end_header_id|>

{{{{ .Prompt }}}}<|eot_id|>{{{{ end }}}}<|start_header_id|>assistant<|end_header_id|>

{{{{ .Response }}}}<|eot_id|>"""
'''
    
    return modelfile_content


def main():
    print("=" * 60)
    print("Creating Custom Petroleum Engineering Model for Ollama")
    print("=" * 60)
    
    # Check if base model exists
    print(f"\n[1/4] Checking base model: {BASE_MODEL}")
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    if BASE_MODEL.split(":")[0] not in result.stdout:
        print(f"⚠ Base model {BASE_MODEL} not found. Please run: ollama pull {BASE_MODEL}")
        return
    print(f"✓ Base model found")
    
    # Load examples from dataset
    print(f"\n[2/4] Loading examples from {DATASET_FILE}")
    if not os.path.exists(DATASET_FILE):
        print(f"⚠ Dataset file not found: {DATASET_FILE}")
        return
    
    examples = load_examples(DATASET_FILE, NUM_EXAMPLES)
    print(f"✓ Loaded {len(examples)} examples")
    
    # Create Modelfile
    print("\n[3/4] Creating Modelfile...")
    modelfile_content = create_modelfile(examples)
    
    modelfile_path = "Modelfile.petroleum"
    with open(modelfile_path, 'w', encoding='utf-8') as f:
        f.write(modelfile_content)
    print(f"✓ Modelfile saved to: {modelfile_path}")
    
    # Create the custom model
    print(f"\n[4/4] Creating custom model: {CUSTOM_MODEL_NAME}")
    print("   This may take a moment...")
    
    result = subprocess.run(
        ["ollama", "create", CUSTOM_MODEL_NAME, "-f", modelfile_path],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print(f"✓ Custom model created successfully!")
        print("\n" + "=" * 60)
        print("Model Creation Complete!")
        print("=" * 60)
        print(f"\nTo use your petroleum engineering model, run:")
        print(f"   ollama run {CUSTOM_MODEL_NAME}")
        print(f"\nExample query:")
        print(f'   ollama run {CUSTOM_MODEL_NAME} "Explain the Darcy equation for fluid flow in porous media"')
    else:
        print(f"✗ Error creating model:")
        print(result.stderr)


if __name__ == "__main__":
    main()
