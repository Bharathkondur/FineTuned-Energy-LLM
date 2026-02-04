"""
Generate expert responses for oil/gas engineering questions using Ollama.
This creates a complete instruction-tuning dataset for Llama fine-tuning.
"""
import json
import requests
import time
from tqdm import tqdm

# Configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.1:8b"  # Change if using a different model
INPUT_FILE = "energy_data_expanded.jsonl"
OUTPUT_FILE = "energy_data_finetuning.jsonl"
CHECKPOINT_FILE = "generation_checkpoint.json"

# Topic mapping
TOPIC_MAP = {
    0: "Petroleum Economics & Finance",
    1: "Reservoir Engineering & Petrophysics", 
    2: "Well Intervention & Completions",
    3: "Production Engineering & Operations",
    4: "Facilities & Process Engineering",
    5: "Health, Safety & Environment (HSE)",
    6: "Drilling Engineering"
}

def generate_response(question: str, topic: str) -> str:
    """Generate an expert response using Ollama."""
    
    system_prompt = f"""You are a senior petroleum engineer with 20+ years of experience in {topic}. 
Provide detailed, technically accurate responses to oil and gas engineering questions.
Include relevant equations, industry standards (API, ISO), and practical considerations.
Be comprehensive but concise. Use proper technical terminology."""

    prompt = f"""Question: {question}

Provide a detailed expert response covering:
1. Key technical concepts and principles
2. Relevant equations or calculations (if applicable)
3. Industry best practices and standards
4. Practical considerations and recommendations

Expert Response:"""

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "system": system_prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "top_p": 0.9,
            "num_predict": 1024  # Max tokens for response
        }
    }
    
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        return result.get("response", "").strip()
    except Exception as e:
        print(f"Error generating response: {e}")
        return None

def load_checkpoint():
    """Load checkpoint to resume from last position."""
    try:
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {"last_index": -1, "completed": []}

def save_checkpoint(index, completed_indices):
    """Save checkpoint for resuming."""
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump({"last_index": index, "completed": completed_indices}, f)

def main():
    # Load data
    print("Loading questions...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    print(f"Loaded {len(data)} questions")
    
    # Load checkpoint
    checkpoint = load_checkpoint()
    start_index = checkpoint["last_index"] + 1
    completed = set(checkpoint["completed"])
    
    if start_index > 0:
        print(f"Resuming from index {start_index} ({len(completed)} already completed)")
    
    # Load existing results if any
    results = []
    try:
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            results = [json.loads(line) for line in f]
        print(f"Loaded {len(results)} existing results")
    except FileNotFoundError:
        pass
    
    # Open output file in append mode
    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
        for i in tqdm(range(start_index, len(data)), desc="Generating responses"):
            if i in completed:
                continue
                
            entry = data[i]
            question = entry['text']
            label = entry.get('label', 1)
            topic = TOPIC_MAP.get(label, "Petroleum Engineering")
            
            # Generate response
            response = generate_response(question, topic)
            
            if response:
                # Create instruction-tuning format
                result = {
                    "messages": [
                        {
                            "role": "system",
                            "content": f"You are an expert petroleum engineer specializing in {topic}. Provide detailed, technically accurate responses."
                        },
                        {
                            "role": "user",
                            "content": question
                        },
                        {
                            "role": "assistant", 
                            "content": response
                        }
                    ]
                }
                
                # Write to file
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
                f.flush()
                
                completed.add(i)
                save_checkpoint(i, list(completed))
                
                # Small delay to prevent overwhelming the model
                time.sleep(0.5)
            else:
                print(f"Failed to generate response for index {i}")
                # Continue to next question
            
            # Progress update every 50 questions
            if (i + 1) % 50 == 0:
                print(f"\nProgress: {i + 1}/{len(data)} ({(i+1)/len(data)*100:.1f}%)")
    
    print(f"\nCompleted! Generated {len(completed)} responses")
    print(f"Output saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
