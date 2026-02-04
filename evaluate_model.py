"""
Test and evaluate the fine-tuned Qwen2.5 3B model for petroleum engineering.
Includes merging LoRA adapters and running evaluation questions.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json

# Configuration
BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_PATH = "./qwen25_energy_finetuned"
MERGED_OUTPUT = "./qwen25_energy_merged"


def load_model_with_adapter():
    """Load base model and merge LoRA adapters."""
    print("=" * 60)
    print("Loading Fine-tuned Qwen2.5 3B for Petroleum Engineering")
    print("=" * 60)
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()
    
    # Load tokenizer
    print("\n[1/3] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, trust_remote_code=True)
    
    # Load base model
    print("[2/3] Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map={"": 0},
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    
    # Load and merge LoRA adapters
    print("[3/3] Loading LoRA adapters...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    
    print("✓ Model loaded successfully!")
    return model, tokenizer


def merge_and_save(model, tokenizer):
    """Merge LoRA adapters into base model and save."""
    print("\n" + "=" * 60)
    print("Merging LoRA adapters into base model...")
    print("=" * 60)
    
    # Merge adapters
    merged_model = model.merge_and_unload()
    
    # Save merged model
    print(f"Saving merged model to {MERGED_OUTPUT}...")
    merged_model.save_pretrained(MERGED_OUTPUT, safe_serialization=True)
    tokenizer.save_pretrained(MERGED_OUTPUT)
    
    print(f"✓ Merged model saved to: {MERGED_OUTPUT}")
    return merged_model


def generate_response(model, tokenizer, messages, max_new_tokens=512):
    """Generate a response using the chat template."""
    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only the new tokens
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response.strip()


def run_evaluation(model, tokenizer):
    """Run evaluation with test questions."""
    print("\n" + "=" * 60)
    print("EVALUATION: Testing Fine-tuned Model")
    print("=" * 60)
    
    system_prompt = "You are an expert petroleum engineer with deep knowledge of drilling, reservoir engineering, production optimization, and energy regulations."
    
    # Test questions covering different aspects
    test_questions = [
        # Knowledge checks
        {
            "category": "Knowledge Check",
            "question": "What is Permeability Anisotropy and why is it important in reservoir engineering?",
        },
        {
            "category": "Knowledge Check", 
            "question": "Explain how to calculate the Langmuir Stress Intensity (LSI) and its significance.",
        },
        {
            "category": "Technical",
            "question": "What factors affect the rate of penetration (ROP) during drilling operations?",
        },
        {
            "category": "Technical",
            "question": "Describe the differences between primary, secondary, and tertiary oil recovery methods.",
        },
        {
            "category": "Practical",
            "question": "How would you diagnose and address stuck pipe during drilling?",
        },
        {
            "category": "Practical",
            "question": "What are the key considerations when designing a hydraulic fracturing treatment?",
        },
        {
            "category": "Calculations",
            "question": "How do you calculate bottomhole pressure using the Darcy equation?",
        },
        {
            "category": "Regulations",
            "question": "What are the main environmental regulations that apply to offshore drilling operations?",
        },
        {
            "category": "Safety",
            "question": "What is a blowout preventer (BOP) and how does it work?",
        },
        {
            "category": "Complex",
            "question": "A well is producing 500 bbl/day of oil with a 30% water cut. The reservoir pressure has declined from 3000 psi to 2500 psi over 2 years. What analysis and interventions would you recommend?",
        },
    ]
    
    results = []
    
    for i, test in enumerate(test_questions, 1):
        print(f"\n{'─' * 60}")
        print(f"Question {i}/{len(test_questions)} [{test['category']}]")
        print(f"{'─' * 60}")
        print(f"Q: {test['question']}\n")
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": test['question']}
        ]
        
        try:
            response = generate_response(model, tokenizer, messages)
            print(f"A: {response}")
            
            results.append({
                "category": test['category'],
                "question": test['question'],
                "response": response,
                "status": "success"
            })
        except Exception as e:
            print(f"Error: {str(e)}")
            results.append({
                "category": test['category'],
                "question": test['question'],
                "response": str(e),
                "status": "error"
            })
    
    # Save results
    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'=' * 60}")
    print(f"Evaluation complete! Results saved to evaluation_results.json")
    print(f"{'=' * 60}")
    
    return results


def interactive_mode(model, tokenizer):
    """Interactive chat mode for testing."""
    print("\n" + "=" * 60)
    print("INTERACTIVE MODE - Ask questions (type 'exit' to quit)")
    print("=" * 60)
    
    system_prompt = "You are an expert petroleum engineer with deep knowledge of drilling, reservoir engineering, production optimization, and energy regulations."
    
    while True:
        question = input("\nYour question: ").strip()
        if question.lower() in ['exit', 'quit', 'q']:
            break
        if not question:
            continue
            
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
        
        print("\nGenerating response...")
        response = generate_response(model, tokenizer, messages)
        print(f"\nAnswer: {response}")


def main():
    # Load model with LoRA adapters
    model, tokenizer = load_model_with_adapter()
    
    # Option to merge and save
    print("\nDo you want to merge LoRA adapters and save? (y/n): ", end="")
    if input().strip().lower() == 'y':
        model = merge_and_save(model, tokenizer)
    
    # Run evaluation
    print("\nDo you want to run the evaluation test set? (y/n): ", end="")
    if input().strip().lower() == 'y':
        run_evaluation(model, tokenizer)
    
    # Interactive mode
    print("\nDo you want to enter interactive mode? (y/n): ", end="")
    if input().strip().lower() == 'y':
        interactive_mode(model, tokenizer)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
