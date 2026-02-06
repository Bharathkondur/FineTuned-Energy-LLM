"""
Comprehensive Evaluation Framework for Fine-Tuned Petroleum Engineering LLM.

This module provides quantitative metrics for evaluating model performance including:
- Answer Quality Scoring (factual accuracy, technical depth, coherence)
- Hallucination Detection
- Response Latency Tracking
- Token Usage Analysis
- Before/After Fine-tuning Comparison

Author: Energy AI Project
Date: 2026-02-06
"""

import torch
import json
import time
import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import statistics
from datetime import datetime


@dataclass
class EvaluationMetrics:
    """Container for all evaluation metrics."""
    question: str
    category: str
    
    # Response metrics
    response_length_tokens: int
    response_length_chars: int
    generation_time_seconds: float
    tokens_per_second: float
    
    # Quality scores (0-1 scale)
    technical_accuracy_score: float
    coherence_score: float
    relevance_score: float
    completeness_score: float
    hallucination_risk_score: float  # Lower is better
    
    # Overall
    composite_quality_score: float
    
    # Content analysis
    contains_equations: bool
    contains_industry_standards: bool
    contains_practical_recommendations: bool
    
    # Response text
    response: str


@dataclass
class BenchmarkComparison:
    """Comparison between base and fine-tuned model."""
    metric_name: str
    base_model_value: float
    finetuned_model_value: float
    improvement_percentage: float


class PetroleumEvaluator:
    """
    Comprehensive evaluator for petroleum engineering LLM responses.
    
    Evaluates:
    1. Technical accuracy using keyword matching and domain patterns
    2. Coherence and structure quality
    3. Hallucination risk assessment
    4. Response latency and throughput
    """
    
    # Domain-specific technical terms for accuracy validation
    PETROLEUM_TERMS = {
        "drilling": ["bit", "mud", "ROP", "WOB", "torque", "BHA", "casing", "cement", "BOP", "kelly", "rotary"],
        "reservoir": ["permeability", "porosity", "saturation", "pressure", "darcy", "PVT", "OOIP", "recovery", "aquifer"],
        "production": ["wellhead", "separator", "choke", "ESP", "gas lift", "artificial lift", "decline curve", "PI"],
        "completion": ["perforation", "gravel pack", "frac", "stimulation", "skin", "tubing", "packer"],
        "equations": ["Q", "k", "μ", "ΔP", "Darcy", "=", "×", "∫", "Σ"],
        "standards": ["API", "ISO", "SPE", "NORSOK", "OSHA", "EPA", "RP"],
    }
    
    # Common hallucination patterns in LLM responses
    HALLUCINATION_PATTERNS = [
        r"I don't have access to",
        r"I cannot find",
        r"As of my knowledge cutoff",
        r"I'm not sure if",
        r"This might not be accurate",
        r"hypothetically speaking",
        r"let me make up",
        r"fictional example",
    ]
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        
    def generate_response(self, question: str, system_prompt: str, max_tokens: int = 512) -> Tuple[str, float, int]:
        """Generate response and track timing metrics."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        input_length = inputs['input_ids'].shape[1]
        
        start_time = time.time()
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        generation_time = time.time() - start_time
        
        output_tokens = outputs[0][input_length:].shape[0]
        response = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        
        return response.strip(), generation_time, output_tokens
    
    def calculate_technical_accuracy(self, response: str, category: str) -> float:
        """
        Calculate technical accuracy based on domain-specific terminology usage.
        Score 0-1 where 1 is highly accurate.
        """
        response_lower = response.lower()
        
        # Get relevant terms for this category
        relevant_terms = []
        if category.lower() in ["knowledge check", "technical"]:
            relevant_terms = self.PETROLEUM_TERMS["drilling"] + self.PETROLEUM_TERMS["reservoir"]
        elif category.lower() == "practical":
            relevant_terms = self.PETROLEUM_TERMS["drilling"] + self.PETROLEUM_TERMS["production"]
        elif category.lower() == "calculations":
            relevant_terms = self.PETROLEUM_TERMS["equations"] + self.PETROLEUM_TERMS["reservoir"]
        elif category.lower() in ["regulations", "safety"]:
            relevant_terms = self.PETROLEUM_TERMS["standards"] + self.PETROLEUM_TERMS["drilling"]
        else:
            # Use all terms
            relevant_terms = [item for sublist in self.PETROLEUM_TERMS.values() for item in sublist]
        
        # Count matching terms
        matches = sum(1 for term in relevant_terms if term.lower() in response_lower)
        
        # Normalize score (expect at least 5-10 relevant terms for a good response)
        score = min(1.0, matches / 8)
        return round(score, 3)
    
    def calculate_coherence_score(self, response: str) -> float:
        """
        Calculate coherence based on structure and formatting.
        Looks for: numbered lists, headers, complete sentences, logical flow.
        """
        score = 0.0
        
        # Check for structured formatting (headers, lists)
        if re.search(r'\*\*[^*]+\*\*', response):  # Bold headers
            score += 0.2
        if re.search(r'\d+\.', response):  # Numbered lists
            score += 0.15
        if re.search(r'[-•]', response):  # Bullet points
            score += 0.1
            
        # Check for complete sentences (ends with period)
        sentences = response.split('.')
        complete_sentences = sum(1 for s in sentences if len(s.strip()) > 20)
        sentence_score = min(0.3, complete_sentences * 0.03)
        score += sentence_score
        
        # Check for logical connectors
        connectors = ["therefore", "because", "however", "additionally", "furthermore", "consequently"]
        connector_count = sum(1 for c in connectors if c in response.lower())
        score += min(0.25, connector_count * 0.05)
        
        return round(min(1.0, score), 3)
    
    def calculate_relevance_score(self, response: str, question: str) -> float:
        """Calculate how relevant the response is to the question."""
        question_words = set(question.lower().split())
        response_words = set(response.lower().split())
        
        # Remove common stop words
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "what", "how", "why", "when", "where", "and", "or", "in", "on", "to", "for"}
        question_keywords = question_words - stop_words
        response_keywords = response_words - stop_words
        
        if not question_keywords:
            return 0.5
        
        # Calculate overlap
        overlap = len(question_keywords & response_keywords)
        relevance = overlap / len(question_keywords)
        
        return round(min(1.0, relevance + 0.3), 3)  # Base bonus for any substantial response
    
    def calculate_completeness_score(self, response: str) -> float:
        """Calculate response completeness based on length and content."""
        # Length-based component
        word_count = len(response.split())
        length_score = min(0.4, word_count / 200)  # Max 0.4 for length
        
        # Content diversity (checks for multiple sections/topics)
        paragraphs = len(response.split('\n\n'))
        diversity_score = min(0.3, paragraphs * 0.1)
        
        # Check for practical elements
        practical_score = 0.0
        if re.search(r'equation|formula|calculate', response.lower()):
            practical_score += 0.1
        if re.search(r'API|ISO|standard', response):
            practical_score += 0.1
        if re.search(r'recommend|consider|ensure', response.lower()):
            practical_score += 0.1
            
        return round(min(1.0, length_score + diversity_score + practical_score), 3)
    
    def calculate_hallucination_risk(self, response: str) -> float:
        """
        Calculate hallucination risk score (lower is better).
        Checks for uncertainty language and made-up patterns.
        """
        risk_score = 0.0
        
        # Check for hallucination patterns
        for pattern in self.HALLUCINATION_PATTERNS:
            if re.search(pattern, response, re.IGNORECASE):
                risk_score += 0.15
        
        # Check for suspiciously specific but unverifiable claims
        # (e.g., very specific percentages, dates without context)
        suspicious_patterns = [
            r'\d{1,2}\.\d{5,}%',  # Overly precise percentages
            r'studies show that exactly',
            r'always|never|100%',  # Absolute statements
        ]
        for pattern in suspicious_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                risk_score += 0.1
        
        # Bonus for citing standards (reduces risk)
        if re.search(r'API RP|ISO \d+|SPE \d+', response):
            risk_score -= 0.2
            
        return round(max(0.0, min(1.0, risk_score)), 3)
    
    def evaluate_response(self, question: str, category: str, system_prompt: str) -> EvaluationMetrics:
        """Run full evaluation on a single question."""
        # Generate response with timing
        response, gen_time, output_tokens = self.generate_response(question, system_prompt)
        
        # Calculate all metrics
        technical = self.calculate_technical_accuracy(response, category)
        coherence = self.calculate_coherence_score(response)
        relevance = self.calculate_relevance_score(response, question)
        completeness = self.calculate_completeness_score(response)
        hallucination = self.calculate_hallucination_risk(response)
        
        # Composite score (weighted average)
        composite = (
            technical * 0.30 +
            coherence * 0.20 +
            relevance * 0.25 +
            completeness * 0.15 +
            (1 - hallucination) * 0.10
        )
        
        return EvaluationMetrics(
            question=question,
            category=category,
            response_length_tokens=output_tokens,
            response_length_chars=len(response),
            generation_time_seconds=round(gen_time, 3),
            tokens_per_second=round(output_tokens / gen_time, 2) if gen_time > 0 else 0,
            technical_accuracy_score=technical,
            coherence_score=coherence,
            relevance_score=relevance,
            completeness_score=completeness,
            hallucination_risk_score=hallucination,
            composite_quality_score=round(composite, 3),
            contains_equations="=" in response or "×" in response,
            contains_industry_standards=bool(re.search(r'API|ISO|SPE', response)),
            contains_practical_recommendations=bool(re.search(r'recommend|consider|ensure|should', response.lower())),
            response=response
        )


class ModelBenchmark:
    """Compare base model vs fine-tuned model performance."""
    
    def __init__(self, base_model_name: str, finetuned_path: str):
        self.base_model_name = base_model_name
        self.finetuned_path = finetuned_path
        
    def run_comparison(self, test_questions: List[Dict], system_prompt: str) -> Dict:
        """Run full comparison benchmark."""
        print("=" * 70)
        print("MODEL PERFORMANCE COMPARISON: Base vs Fine-tuned")
        print("=" * 70)
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "base_model": self.base_model_name,
            "finetuned_model": self.finetuned_path,
            "base_results": [],
            "finetuned_results": [],
            "comparison": {}
        }
        
        # Load and evaluate base model
        print("\n[1/2] Evaluating BASE model...")
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_name, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            device_map={"": 0},
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        
        base_evaluator = PetroleumEvaluator(base_model, tokenizer)
        
        for q in test_questions:
            metrics = base_evaluator.evaluate_response(q["question"], q["category"], system_prompt)
            results["base_results"].append(asdict(metrics))
            print(f"  ✓ {q['category']}: {metrics.composite_quality_score:.3f}")
        
        # Clear memory
        del base_model
        torch.cuda.empty_cache()
        
        # Load and evaluate fine-tuned model
        print("\n[2/2] Evaluating FINE-TUNED model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            device_map={"": 0},
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        finetuned_model = PeftModel.from_pretrained(base_model, self.finetuned_path)
        
        ft_evaluator = PetroleumEvaluator(finetuned_model, tokenizer)
        
        for q in test_questions:
            metrics = ft_evaluator.evaluate_response(q["question"], q["category"], system_prompt)
            results["finetuned_results"].append(asdict(metrics))
            print(f"  ✓ {q['category']}: {metrics.composite_quality_score:.3f}")
        
        # Calculate comparison metrics
        base_scores = [r["composite_quality_score"] for r in results["base_results"]]
        ft_scores = [r["composite_quality_score"] for r in results["finetuned_results"]]
        
        results["comparison"] = {
            "base_avg_quality": round(statistics.mean(base_scores), 3),
            "finetuned_avg_quality": round(statistics.mean(ft_scores), 3),
            "quality_improvement_pct": round((statistics.mean(ft_scores) - statistics.mean(base_scores)) / statistics.mean(base_scores) * 100, 2),
            "base_avg_latency": round(statistics.mean([r["generation_time_seconds"] for r in results["base_results"]]), 3),
            "finetuned_avg_latency": round(statistics.mean([r["generation_time_seconds"] for r in results["finetuned_results"]]), 3),
            "base_avg_tokens_per_sec": round(statistics.mean([r["tokens_per_second"] for r in results["base_results"]]), 2),
            "finetuned_avg_tokens_per_sec": round(statistics.mean([r["tokens_per_second"] for r in results["finetuned_results"]]), 2),
        }
        
        return results


def run_full_evaluation():
    """Main evaluation entry point."""
    print("=" * 70)
    print("PETROLEUM ENGINEERING LLM - COMPREHENSIVE EVALUATION")
    print("=" * 70)
    
    # Configuration
    BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
    ADAPTER_PATH = "./results/qwen25_energy_finetuned"
    SYSTEM_PROMPT = "You are an expert petroleum engineer with deep knowledge of drilling, reservoir engineering, production optimization, and energy regulations."
    
    # Test questions
    test_questions = [
        {"category": "Knowledge", "question": "What is permeability anisotropy and why is it important?"},
        {"category": "Technical", "question": "What factors affect rate of penetration (ROP) during drilling?"},
        {"category": "Practical", "question": "How would you diagnose stuck pipe during drilling?"},
        {"category": "Calculations", "question": "How do you calculate bottomhole pressure using Darcy's equation?"},
        {"category": "Regulations", "question": "What environmental regulations apply to offshore drilling?"},
    ]
    
    # Load model
    print("\n[1/3] Loading fine-tuned model...")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map={"": 0},
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    
    # Run evaluation
    print("[2/3] Running comprehensive evaluation...")
    evaluator = PetroleumEvaluator(model, tokenizer)
    
    all_results = []
    for i, q in enumerate(test_questions, 1):
        print(f"\nEvaluating {i}/{len(test_questions)}: [{q['category']}]")
        metrics = evaluator.evaluate_response(q["question"], q["category"], SYSTEM_PROMPT)
        all_results.append(asdict(metrics))
        
        print(f"  📊 Quality Score: {metrics.composite_quality_score:.3f}")
        print(f"  ⚡ Latency: {metrics.generation_time_seconds:.2f}s")
        print(f"  🔥 Tokens/sec: {metrics.tokens_per_second:.1f}")
        print(f"  ⚠️  Hallucination Risk: {metrics.hallucination_risk_score:.3f}")
    
    # Calculate summary statistics
    print("\n[3/3] Generating summary...")
    summary = {
        "timestamp": datetime.now().isoformat(),
        "model": f"{BASE_MODEL} + LoRA ({ADAPTER_PATH})",
        "num_questions": len(test_questions),
        "metrics_summary": {
            "avg_quality_score": round(statistics.mean([r["composite_quality_score"] for r in all_results]), 3),
            "avg_technical_accuracy": round(statistics.mean([r["technical_accuracy_score"] for r in all_results]), 3),
            "avg_coherence": round(statistics.mean([r["coherence_score"] for r in all_results]), 3),
            "avg_hallucination_risk": round(statistics.mean([r["hallucination_risk_score"] for r in all_results]), 3),
            "avg_latency_seconds": round(statistics.mean([r["generation_time_seconds"] for r in all_results]), 3),
            "avg_tokens_per_second": round(statistics.mean([r["tokens_per_second"] for r in all_results]), 2),
        },
        "detailed_results": all_results
    }
    
    # Save results to results folder
    output_file = "results/comprehensive_evaluation_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'=' * 70}")
    print("EVALUATION SUMMARY")
    print(f"{'=' * 70}")
    print(f"  📊 Average Quality Score: {summary['metrics_summary']['avg_quality_score']:.3f}")
    print(f"  🎯 Technical Accuracy:    {summary['metrics_summary']['avg_technical_accuracy']:.3f}")
    print(f"  📝 Coherence Score:       {summary['metrics_summary']['avg_coherence']:.3f}")
    print(f"  ⚠️  Hallucination Risk:   {summary['metrics_summary']['avg_hallucination_risk']:.3f}")
    print(f"  ⚡ Avg Latency:           {summary['metrics_summary']['avg_latency_seconds']:.2f}s")
    print(f"  🔥 Throughput:            {summary['metrics_summary']['avg_tokens_per_second']:.1f} tokens/sec")
    print(f"\n✅ Results saved to: {output_file}")
    
    return summary


if __name__ == "__main__":
    run_full_evaluation()
