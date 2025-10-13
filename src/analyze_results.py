# src/analyze_results.py
import json
import matplotlib.pyplot as plt
import numpy as np

def analyze():
    with open('results/phase1_results.json', 'r') as f:
        results = json.load(f)
    
    with open('results/phase1_summary.json', 'r') as f:
        summary = json.load(f)
    
    baseline = results['baseline']
    with_memory = results['with_memory']
    
    # Accuracy over time
    baseline_acc = [int(r['evaluation']['success']) for r in baseline]
    memory_acc = [int(r['evaluation']['success']) for r in with_memory]
    
    baseline_cumulative = [sum(baseline_acc[:i+1])/(i+1) for i in range(len(baseline_acc))]
    memory_cumulative = [sum(memory_acc[:i+1])/(i+1) for i in range(len(memory_acc))]
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Cumulative accuracy
    ax1.plot(baseline_cumulative, label='Baseline', linewidth=2, color='#e74c3c')
    ax1.plot(memory_cumulative, label='With Memory', linewidth=2, color='#2ecc71')
    ax1.axhline(summary['baseline_accuracy'], color='#e74c3c', linestyle='--', alpha=0.3)
    ax1.axhline(summary['with_memory_accuracy'], color='#2ecc71', linestyle='--', alpha=0.3)
    ax1.fill_between(range(len(baseline_cumulative)), 
                      summary['baseline_ci_lower'], 
                      summary['baseline_ci_upper'], 
                      alpha=0.2, color='#e74c3c')
    ax1.fill_between(range(len(memory_cumulative)), 
                      summary['with_memory_ci_lower'], 
                      summary['with_memory_ci_upper'], 
                      alpha=0.2, color='#2ecc71')
    ax1.set_xlabel('Problem Number')
    ax1.set_ylabel('Cumulative Accuracy')
    ax1.set_title('Phase 1: Accuracy Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Bar chart
    categories = ['Baseline', 'With Memory']
    accuracies = [summary['baseline_accuracy'], summary['with_memory_accuracy']]
    colors = ['#e74c3c', '#2ecc71']
    
    bars = ax2.bar(categories, accuracies, color=colors, alpha=0.7)
    
    ci_ranges = [
        (summary['baseline_ci_upper'] - summary['baseline_ci_lower']) / 2,
        (summary['with_memory_ci_upper'] - summary['with_memory_ci_lower']) / 2
    ]
    ax2.errorbar(categories, accuracies, yerr=ci_ranges, fmt='none', color='black', capsize=10)
    
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.1%}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Phase 1: Overall Comparison')
    ax2.set_ylim(0, max(accuracies) * 1.2)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('results/phase1_accuracy.png', dpi=150, bbox_inches='tight')
    print("✓ Saved plot to results/phase1_accuracy.png")
    
    # Improvements
    print("\n" + "="*70)
    print("PROBLEMS THAT IMPROVED WITH MEMORY")
    print("="*70)
    
    improvements = []
    for b, m in zip(baseline, with_memory):
        if b['problem_id'] == m['problem_id']:
            if not b['evaluation']['success'] and m['evaluation']['success']:
                improvements.append({
                    'problem_id': b['problem_id'],
                    'question': b['question'][:100],
                    'baseline_answer': b['solution']['answer'],
                    'memory_answer': m['solution']['answer'],
                    'expected': b['evaluation']['expected'],
                    'retrieved_memories': m.get('retrieved_memories', [])
                })
    
    print(f"\n{len(improvements)} problems improved:\n")
    for i, imp in enumerate(improvements[:10], 1):
        print(f"{i}. {imp['question']}...")
        print(f"   Baseline: {imp['baseline_answer']} ✗")
        print(f"   Memory: {imp['memory_answer']} ✓")
        print(f"   Expected: {imp['expected']}")
        print(f"   Strategies: {', '.join(imp['retrieved_memories'][:2])}")
        print()
    
    # REGRESSIONS - NEW SECTION
    print("\n" + "="*70)
    print("PROBLEMS THAT REGRESSED WITH MEMORY")
    print("="*70)
    
    regressions = []
    for b, m in zip(baseline, with_memory):
        if b['problem_id'] == m['problem_id']:
            if b['evaluation']['success'] and not m['evaluation']['success']:
                regressions.append({
                    'problem_id': b['problem_id'],
                    'question': b['question'][:100],
                    'baseline_answer': b['solution']['answer'],
                    'memory_answer': m['solution']['answer'],
                    'expected': b['evaluation']['expected'],
                    'retrieved_memories': m.get('retrieved_memories', [])
                })
    
    print(f"\n{len(regressions)} problems regressed:\n")
    for i, reg in enumerate(regressions[:10], 1):
        print(f"{i}. {reg['question']}...")
        print(f"   Baseline: {reg['baseline_answer']} ✓")
        print(f"   Memory: {reg['memory_answer']} ✗")
        print(f"   Expected: {reg['expected']}")
        print(f"   Retrieved: {', '.join(reg['retrieved_memories'][:2])}")
        print()
    
    # Stats
    print("\n" + "="*70)
    print("STATISTICAL ANALYSIS")
    print("="*70)
    print(f"Baseline:    {summary['baseline_accuracy']:.2%} " +
          f"(CI: [{summary['baseline_ci_lower']:.2%}, {summary['baseline_ci_upper']:.2%}])")
    print(f"With Memory: {summary['with_memory_accuracy']:.2%} " +
          f"(CI: [{summary['with_memory_ci_lower']:.2%}, {summary['with_memory_ci_upper']:.2%}])")
    print()
    print(f"Absolute: {summary['absolute_improvement']:+.2%}")
    print(f"Relative: {summary['relative_improvement']:+.2%}")
    print()
    print(f"Improvements: {len(improvements)}")
    print(f"Regressions: {len(regressions)}")
    print(f"Net change: {len(improvements) - len(regressions)}")
    print()
    
    if summary['statistically_significant']:
        print("✓ Statistically significant at 95% confidence")
    else:
        print("⚠ Not statistically significant")

if __name__ == '__main__':
    analyze()