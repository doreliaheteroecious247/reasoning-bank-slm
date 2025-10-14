# src/llm_client.py
from openai import OpenAI
import json
from typing import Optional, Dict

class LlamaServerClient:
    """Client for llama-server using OpenAI-compatible API"""
    
    def __init__(self, base_url="http://localhost:8080/v1", model_name="qwen3-1.7b"):
        self.client = OpenAI(
            base_url=base_url,
            api_key="not-needed"  # llama-server doesn't need key
        )
        self.model_name = model_name
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None, 
                 max_tokens: int = 16384, temperature: float = 0.6) -> str:
        """Generate completion"""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling llama-server: {e}")
            return ""
    
    def solve_math_problem(self, question: str, retrieved_memories: str = "") -> Dict:
        """Solve a math problem with optional memory context"""
        
        # Base system prompt with anti-copying enforcement
        system_prompt = """You are a math problem solver. Solve problems step-by-step using clear reasoning.

Format your response as:
REASONING: <show your step-by-step calculations>
ANSWER: <final numeric answer only>"""
        
        # Add memory context if provided, with explicit warnings
        if retrieved_memories:
            system_prompt += f"\n\n{retrieved_memories}"
            system_prompt += "\n**Remember**: The hints above provide STRATEGIES ONLY. You must calculate the answer yourself using your own arithmetic."
        
        prompt = f"## Problem:\n{question}\n\n## Your Solution:"
        
        response = self.generate(prompt, system_prompt, temperature=0.0)
        
        # Parse response
        reasoning = ""
        answer = ""
        
        if "REASONING:" in response and "ANSWER:" in response:
            parts = response.split("ANSWER:")
            reasoning = parts[0].replace("REASONING:", "").strip()
            answer = parts[1].strip()
        else:
            reasoning = response
            # Try to extract last number
            import re
            numbers = re.findall(r'-?\d+\.?\d*', response)
            answer = numbers[-1] if numbers else ""
        
        return {
            'reasoning': reasoning,
            'answer': answer,
            'full_response': response
        }

class JudgeClient(LlamaServerClient):
    """Separate client for judge model (typically larger/faster model on different port)"""
    
    def __init__(self, base_url="http://localhost:8081/v1", model_name="qwen3-4b"):
        super().__init__(base_url=base_url, model_name=model_name)