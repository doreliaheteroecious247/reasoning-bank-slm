# scripts/test_retrieval.py
import sys
sys.path.append('src')

from memory import ReasoningBank, MemoryItem
from retrieval.retriever import MemoryRetriever
from datetime import datetime

# Create test memories
bank = ReasoningBank('memory_bank/test_bank.json')
bank.clear()

# Memory with answer "42"
bank.add_memory(MemoryItem(
    title="Test Strategy",
    description="A test memory",
    content="When solving this type of problem, the answer is 42.",
    source_problem_id="test_1",
    success=True,
    created_at=datetime.now().isoformat()
))

# Memory without specific answer
bank.add_memory(MemoryItem(
    title="General Strategy",
    description="Good approach",
    content="Break down the problem into steps and solve systematically.",
    source_problem_id="test_2",
    success=True,
    created_at=datetime.now().isoformat()
))

# Test retrieval with answer leak protection
retriever = MemoryRetriever()
query = "What is 20 + 22?"
retrieved = retriever.retrieve(query, bank.get_all_memories(), top_k=2, expected_value="42")

print(f"Query: {query}")
print(f"Expected answer: 42")
print(f"Memories retrieved: {len(retrieved)}")

for mem, score in retrieved:
    print(f"  - {mem.title}")
    if "42" in mem.content:
        print("    ⚠ WARNING: Answer leak detected!")
    else:
        print("    ✓ No answer leak")

# Cleanup
bank.clear()

if len(retrieved) == 1 and "42" not in retrieved[0][0].content:
    print("\n✓ Retrieval filter working correctly!")
else:
    print("\n✗ Retrieval filter may have issues")