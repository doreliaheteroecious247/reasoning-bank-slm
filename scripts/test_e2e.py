# scripts/test_e2e.py
import sys
sys.path.append('src')

from llm_client import LlamaServerClient
from judge.evaluator import MathJudge

# Test LLM connection
print("Testing llama-server connection...")
client = LlamaServerClient()

response = client.solve_math_problem("What is 5 + 7?")
print(f"Question: What is 5 + 7?")
print(f"Response: {response['answer']}")

# Test judge
judge = MathJudge(client)
is_correct = judge.is_correct(response['answer'], "12")

print(f"Judge evaluation: {'✓ Correct' if is_correct else '✗ Wrong'}")

if is_correct:
    print("\n✓ End-to-end flow working!")
else:
    print("\n⚠ Check if model is producing valid answers")