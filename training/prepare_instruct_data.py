"""
Transform oil/gas engineering dataset for Llama-3.1-8B-Instruct fine-tuning.
Converts classification format to instruction-response format.
"""
import json
import random

# Load the augmented data
with open('energy_data_expanded.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]

print(f"Loaded {len(data)} entries")

# Topic mapping based on labels (inferred from data analysis)
TOPIC_MAP = {
    0: "Petroleum Economics & Finance",
    1: "Reservoir Engineering & Petrophysics",
    2: "Well Intervention & Completions",
    3: "Production Engineering & Operations",
    4: "Facilities & Process Engineering",
    5: "Health, Safety & Environment (HSE)",
    6: "Drilling Engineering"
}

# System prompts for variety
SYSTEM_PROMPTS = [
    "You are an expert petroleum engineer with deep knowledge in oil and gas operations.",
    "You are a senior technical advisor specializing in upstream oil and gas engineering.",
    "You are an experienced oil and gas industry consultant with expertise across multiple disciplines.",
]

# Instruction templates for variety
INSTRUCTION_TEMPLATES = [
    "Please provide a detailed technical response to the following question:",
    "As an oil and gas engineering expert, address the following problem:",
    "Provide a comprehensive analysis and solution for:",
    "Answer the following technical question with detailed explanations:",
    "Offer your expert analysis on the following oil and gas engineering topic:",
]

def create_instruction_format(entry):
    """Convert a single entry to instruction format for Llama fine-tuning."""
    text = entry['text']
    label = entry.get('label', 1)
    topic = TOPIC_MAP.get(label, "Petroleum Engineering")
    
    # Create the instruction-response format
    system = random.choice(SYSTEM_PROMPTS)
    instruction_template = random.choice(INSTRUCTION_TEMPLATES)
    
    # Format for Llama-3.1-Instruct (chat format)
    formatted = {
        "messages": [
            {
                "role": "system",
                "content": f"{system} Your area of expertise includes {topic}."
            },
            {
                "role": "user", 
                "content": f"{instruction_template}\n\n{text}"
            },
            {
                "role": "assistant",
                "content": "[RESPONSE TO BE GENERATED]"  # Placeholder - you'll need to generate responses
            }
        ]
    }
    
    return formatted

def create_completion_format(entry):
    """Create a simpler prompt-completion format."""
    text = entry['text']
    label = entry.get('label', 1)
    topic = TOPIC_MAP.get(label, "Petroleum Engineering")
    
    return {
        "prompt": f"### Topic: {topic}\n\n### Question:\n{text}\n\n### Expert Response:\n",
        "completion": "[RESPONSE TO BE GENERATED]"  # Placeholder
    }

def create_text_only_format(entry):
    """Create text-only format for continued pretraining."""
    text = entry['text']
    label = entry.get('label', 1)
    topic = TOPIC_MAP.get(label, "Petroleum Engineering")
    
    return {
        "text": f"The following is a technical question in {topic}:\n\n{text}"
    }

# Generate all three formats
instruct_data = []
completion_data = []
text_data = []

for entry in data:
    instruct_data.append(create_instruction_format(entry))
    completion_data.append(create_completion_format(entry))
    text_data.append(create_text_only_format(entry))

# Save instruction format (for Chat fine-tuning)
with open('energy_data_instruct.jsonl', 'w', encoding='utf-8') as f:
    for item in instruct_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')
print(f"Saved {len(instruct_data)} entries to energy_data_instruct.jsonl")

# Save completion format (for completion fine-tuning)
with open('energy_data_completion.jsonl', 'w', encoding='utf-8') as f:
    for item in completion_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')
print(f"Saved {len(completion_data)} entries to energy_data_completion.jsonl")

# Save text-only format (for continued pretraining)
with open('energy_data_pretrain.jsonl', 'w', encoding='utf-8') as f:
    for item in text_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')
print(f"Saved {len(text_data)} entries to energy_data_pretrain.jsonl")

# Print sample
print("\n" + "="*60)
print("SAMPLE INSTRUCTION FORMAT:")
print("="*60)
print(json.dumps(instruct_data[0], indent=2))

print("\n" + "="*60)
print("SAMPLE COMPLETION FORMAT:")
print("="*60)
print(json.dumps(completion_data[0], indent=2))

print("\n" + "="*60)
print("IMPORTANT NOTE:")
print("="*60)
print("""
The current dataset only contains QUESTIONS, not answers.
For proper instruction fine-tuning, you need QUESTION-ANSWER pairs.

Options to generate responses:
1. Use GPT-4/Claude to generate expert responses for each question
2. Use domain experts to write responses manually
3. Use the text-only format for continued pretraining (no responses needed)

For continued pretraining (recommended for domain knowledge):
  - Use 'energy_data_pretrain.jsonl'
  - This teaches the model domain-specific terminology and concepts

For instruction fine-tuning:
  - You'll need to generate responses first
  - Consider using an LLM to generate high-quality technical responses
""")
