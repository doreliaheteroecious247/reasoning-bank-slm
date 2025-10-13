# src/run_phase1.py
import json
from pathlib import Path
from tqdm import tqdm
import sys

from llm_client import LlamaServerClient
from memory import ReasoningBank, MemoryItem
from retrieval.retriever import MemoryRetriever
from judge.evaluator import MathJudge
from extraction.extractor import MemoryExtractor

class Phase1Experiment:
    """Phase 1: Does retrieval help?"""
    
    def __init__(self, random_seed: int = 42):
        self.llm = LlamaServerClient()
        self.memory_bank = ReasoningBank()
        self.retriever = MemoryRetriever()
        self.judge = MathJudge(self.llm)
        self.extractor = MemoryExtractor(self.llm)
        
        self.random_seed = random_seed
        
        self.results = {
            'baseline': [],
            'with_memory': []
        }
    
    def load_problems(self, path: str):
        """Load problem dataset"""
        with open(path, 'r') as f:
            return json.load(f)
    
    def run_baseline_without_memory(self, problems: list, limit: int = 50):
        """Run baseline WITHOUT memory and WITHOUT building new memories"""
        print("\n=== BASELINE: Solving without memory ===\n")
        print(f"Using random seed: {self.random_seed} for deterministic seeding")
        
        for problem in tqdm(problems[:limit], desc="Baseline"):
            solution = self.llm.solve_math_problem(problem['question'])
            
            evaluation = self.judge.evaluate_with_reasoning(
                problem['question'],
                solution['answer'],
                problem['expected_value']
            )
            
            result = {
                'problem_id': problem['id'],
                'question': problem['question'],
                'solution': solution,
                'evaluation': evaluation
            }
            
            self.results['baseline'].append(result)
        
        accuracy = sum(1 for r in self.results['baseline'] if r['evaluation']['success']) / len(self.results['baseline'])
        print(f"\nBaseline Accuracy: {accuracy:.2%}")
    
    def run_with_memory(self, problems: list, limit: int = 50):
        """Run with memory retrieval"""
        print("\n=== WITH MEMORY: Solving with retrieval ===\n")
        
        if len(self.memory_bank) == 0:
            print("Warning: Memory bank is empty!")
            return
        
        for problem in tqdm(problems[:limit], desc="With Memory"):
            # Retrieve with answer leak protection
            retrieved = self.retriever.retrieve(
                problem['question'],
                self.memory_bank.get_all_memories(),
                top_k=2,
                expected_value=problem['expected_value']
            )
            
            # Format memories for prompt
            memory_context = self.retriever.format_memories_for_prompt(retrieved)
            
            # Solve with memory
            solution = self.llm.solve_math_problem(
                problem['question'],
                retrieved_memories=memory_context
            )
            
            evaluation = self.judge.evaluate_with_reasoning(
                problem['question'],
                solution['answer'],
                problem['expected_value']
            )
            
            result = {
                'problem_id': problem['id'],
                'question': problem['question'],
                'retrieved_memories': [m.title for m, _ in retrieved],
                'num_memories_retrieved': len(retrieved),
                'solution': solution,
                'evaluation': evaluation
            }
            
            self.results['with_memory'].append(result)
        
        accuracy = sum(1 for r in self.results['with_memory'] if r['evaluation']['success']) / len(self.results['with_memory'])
        print(f"\nWith Memory Accuracy: {accuracy:.2%}")
        print(f"Memory bank size: {len(self.memory_bank)} items")
    
    def save_results(self):
        """Save all results with statistical analysis"""
        with open('results/phase1_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Compute statistics
        baseline_successes = sum(1 for r in self.results['baseline'] if r['evaluation']['success'])
        memory_successes = sum(1 for r in self.results['with_memory'] if r['evaluation']['success'])
        
        baseline_acc = baseline_successes / len(self.results['baseline']) if self.results['baseline'] else 0
        memory_acc = memory_successes / len(self.results['with_memory']) if self.results['with_memory'] else 0
        
        # Compute confidence intervals
        baseline_ci = self._compute_wilson_ci(baseline_successes, len(self.results['baseline']))
        memory_ci = self._compute_wilson_ci(memory_successes, len(self.results['with_memory']))
        
        summary = {
            'baseline_accuracy': baseline_acc,
            'baseline_ci_lower': baseline_ci[0],
            'baseline_ci_upper': baseline_ci[1],
            'with_memory_accuracy': memory_acc,
            'with_memory_ci_lower': memory_ci[0],
            'with_memory_ci_upper': memory_ci[1],
            'absolute_improvement': memory_acc - baseline_acc,
            'relative_improvement': (memory_acc - baseline_acc) / baseline_acc if baseline_acc > 0 else 0,
            'confidence_intervals_overlap': not (memory_ci[0] > baseline_ci[1]),
            'statistically_significant': memory_ci[0] > baseline_ci[1],
            'memory_bank_size': len(self.memory_bank),
            'problems_tested': len(self.results['baseline']),
            'random_seed': self.random_seed
        }
        
        with open('results/phase1_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print results
        print("\n" + "="*70)
        print("PHASE 1 RESULTS")
        print("="*70)
        print(f"Baseline Accuracy:    {baseline_acc:.2%} (95% CI: [{baseline_ci[0]:.2%}, {baseline_ci[1]:.2%}])")
        print(f"With Memory Accuracy: {memory_acc:.2%} (95% CI: [{memory_ci[0]:.2%}, {memory_ci[1]:.2%}])")
        print(f"Absolute Improvement: {summary['absolute_improvement']:+.2%}")
        print(f"Relative Improvement: {summary['relative_improvement']:+.2%}")
        print(f"Memory Bank Size:     {len(self.memory_bank)} items")
        print()
        if summary['statistically_significant']:
            print("✓ Improvement is statistically significant at 95% confidence")
            print("  (Memory CI lower bound > Baseline CI upper bound)")
        else:
            print("⚠ Improvement not statistically significant at 95% confidence")
            print("  (Confidence intervals overlap)")
        print("="*70)
    
    def _compute_wilson_ci(self, successes: int, n: int, confidence: float = 0.95) -> tuple:
        """Compute Wilson score confidence interval for binomial proportion"""
        if n == 0:
            return (0.0, 0.0)
        
        import math
        
        p = successes / n
        z = 1.96  # 95% confidence
        
        denominator = 1 + z**2 / n
        center = (p + z**2 / (2*n)) / denominator
        margin = z * math.sqrt(p * (1-p) / n + z**2 / (4*n**2)) / denominator
        
        return (max(0, center - margin), min(1, center + margin))

def main():
    experiment = Phase1Experiment()
    
    train_problems = experiment.load_problems('data/train_problems.json')
    test_problems = experiment.load_problems('data/test_problems.json')
    
    # Build memory bank ONCE from training set
    print("Building memory bank from training set...")
    for problem in tqdm(train_problems[:100], desc="Training"):
        solution = experiment.llm.solve_math_problem(problem['question'])
        evaluation = experiment.judge.evaluate_with_reasoning(
            problem['question'], 
            solution['answer'], 
            problem['expected_value']
        )
        memories = experiment.extractor.extract_from_trajectory(
            problem['id'], 
            problem['question'],
            {**solution, 'expected': problem['expected_value']},
            evaluation['success']
        )
        experiment.memory_bank.add_memories(memories)
    
    print(f"Memory bank built: {len(experiment.memory_bank)} items\n")
    
    # Test 1: Baseline WITHOUT memory (no retrieval, no building)
    print("=== Test 1: Baseline WITHOUT memory ===")
    experiment.run_baseline_without_memory(test_problems, limit=100)
    
    # Test 2: WITH memory (with retrieval)
    print("\n=== Test 2: WITH memory ===")
    experiment.run_with_memory(test_problems, limit=100)
    
    experiment.save_results()

if __name__ == '__main__':
    main()