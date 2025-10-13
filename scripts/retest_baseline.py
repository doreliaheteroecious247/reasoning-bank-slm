# scripts/retest_baseline.py
import sys
sys.path.append('src')

from run_phase1 import Phase1Experiment

experiment = Phase1Experiment()
test_problems = experiment.load_problems('data/test_problems.json')

# Clear memory bank temporarily
experiment.memory_bank.clear()

print("Running baseline on TEST set (same problems as with-memory)...")
experiment.run_baseline(test_problems, limit=50)

# Save separate results
import json
with open('results/test_baseline.json', 'w') as f:
    json.dump({
        'baseline_on_test': experiment.results['baseline']
    }, f, indent=2)

baseline_acc = sum(1 for r in experiment.results['baseline'] if r['evaluation']['success']) / 50
print(f"\nTest Set Baseline (no memory): {baseline_acc:.2%}")