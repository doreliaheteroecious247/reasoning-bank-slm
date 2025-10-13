# scripts/diagnose_failure.py
import json
import sys
sys.path.append('src')

# Load results
with open('results/phase1_results.json', 'r') as f:
    results = json.load(f)

baseline = results['baseline']
with_memory = results['with_memory']

print("="*70)
print("DEBUGGING MARCY PENSION PROBLEM")
print("="*70)

# Find the Marcy pension problem specifically
for b, m in zip(baseline, with_memory):
    if 'Marcy' in b['question'] and 'pension' in b['question']:
        print("\nFOUND MARCY PROBLEM\n")
        print(f"Question: {b['question'][:150]}...\n")
        
        print("BASELINE evaluation:")
        print(json.dumps(b['evaluation'], indent=2))
        
        print("\nMEMORY evaluation:")
        print(json.dumps(m['evaluation'], indent=2))
        
        print("\nRaw solution answers:")
        print(f"Baseline solution answer: '{b['solution']['answer']}'")
        print(f"Memory solution answer: '{m['solution']['answer']}'")
        
        print(f"\nExpected value: '{b['evaluation']['expected']}'")
        
        print("\n" + "="*70)
        break

print("\n" + "="*70)
print("FINDING ALL REGRESSIONS")
print("="*70)

# Find problems that REGRESSED (correct -> incorrect)
regressions = []
for b, m in zip(baseline, with_memory):
    if b['problem_id'] == m['problem_id']:
        if b['evaluation']['success'] and not m['evaluation']['success']:
            regressions.append({
                'problem': b['question'][:80],
                'baseline_answer': b['evaluation']['predicted_number'],
                'memory_answer': m['evaluation']['predicted_number'],
                'expected': b['evaluation']['expected_number'],
                'retrieved_memories': m.get('retrieved_memories', [])
            })

print(f"\nFound {len(regressions)} problems that REGRESSED with memory\n")

for i, reg in enumerate(regressions[:5], 1):
    print(f"\n{i}. Problem: {reg['problem']}...")
    print(f"   Baseline: {reg['baseline_answer']} ✓")
    print(f"   With Memory: {reg['memory_answer']} ✗")
    print(f"   Expected: {reg['expected']}")
    print(f"   Retrieved strategies: {reg['retrieved_memories'][:3]}")

print("\n" + "="*70)

# Load and inspect memory bank
try:
    with open('memory_bank/reasoning_bank.json', 'r') as f:
        memories = json.load(f)

    print(f"\nMemory Bank Stats:")
    print(f"  Total memories: {len(memories)}")
    success_count = sum(1 for m in memories if m['success'])
    print(f"  From successes: {success_count}")
    print(f"  From failures: {len(memories) - success_count}")

    print(f"\nSample Memory Items:")
    print("="*70)
    for i, mem in enumerate(memories[:3], 1):
        print(f"\n{i}. {mem['title']}")
        print(f"   Status: {'Success' if mem['success'] else 'Failure'}")
        print(f"   Content preview: {mem['content'][:150]}...")
except FileNotFoundError:
    print("\nMemory bank file not found. Run the experiment first.")