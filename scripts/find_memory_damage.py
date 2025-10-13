# scripts/find_memory_damage.py
import json
import sys
sys.path.append('src')

# Load both results
with open('results/test_baseline.json', 'r') as f:
    test_baseline = json.load(f)['baseline_on_test']

with open('results/phase1_results.json', 'r') as f:
    full_results = json.load(f)
    with_memory = full_results['with_memory']

print("Comparing same test set: baseline vs with-memory\n")
print("="*70)

# Find problems where baseline succeeded but memory failed
memory_damage = []
for b, m in zip(test_baseline, with_memory):
    b_correct = b['evaluation']['success']
    m_correct = m['evaluation']['success']
    
    if b_correct and not m_correct:
        # Load the actual responses to see reasoning
        memory_damage.append({
            'problem': b['question'][:100],
            'baseline_answer': b['solution'].get('answer', 'N/A'),
            'baseline_reasoning': b['solution'].get('reasoning', '')[:200],
            'memory_answer': m['solution'].get('answer', 'N/A'),
            'memory_reasoning': m['solution'].get('reasoning', '')[:200],
            'expected': b['evaluation']['expected'],
            'retrieved': m.get('retrieved_memories', [])
        })

print(f"Found {len(memory_damage)} problems where memory HURT performance\n")

for i, case in enumerate(memory_damage[:3], 1):
    print(f"\n{'='*70}")
    print(f"CASE {i}: {case['problem']}...")
    print(f"\nExpected: {case['expected']}")
    print(f"\n--- BASELINE (CORRECT) ---")
    print(f"Answer: {case['baseline_answer']}")
    print(f"Reasoning: {case['baseline_reasoning']}...")
    print(f"\n--- WITH MEMORY (WRONG) ---")
    print(f"Answer: {case['memory_answer']}")
    print(f"Reasoning: {case['memory_reasoning']}...")
    print(f"\nRetrieved memories:")
    for mem in case['retrieved'][:3]:
        print(f"  - {mem}")

print(f"\n{'='*70}")
print(f"\nHypotheses to test:")
print("1. Are retrieved memories actually relevant?")
print("2. Is the model getting confused by strategy hints?")
print("3. Is the anti-copying warning making it second-guess correct answers?")
print("4. Is prompt too long, causing attention issues?")