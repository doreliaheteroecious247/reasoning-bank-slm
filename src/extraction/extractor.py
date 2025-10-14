# src/extraction/extractor.py
import sys
sys.path.append('..')
from llm_client import LlamaServerClient
from memory import MemoryItem
from datetime import datetime
from typing import List, Dict

class MemoryExtractor:
    """Extract memory items from problem-solving trajectories"""
    
    def __init__(self, llm_client: LlamaServerClient):
        self.llm = llm_client
    
    def extract_from_trajectory(self, problem_id: str, question: str, 
                                 solution: Dict, success: bool) -> List[MemoryItem]:
        """Extract 1-3 memory items from a trajectory"""
        
        if success:
            prompt = self._create_success_prompt(question, solution['reasoning'])
        else:
            prompt = self._create_failure_prompt(question, solution['reasoning'], 
                                                  solution.get('expected', ''))
        
        response = self.llm.generate(prompt, temperature=0.0, max_tokens=2048)
        
        # Parse memory items from response
        memories = self._parse_memory_items(response, problem_id, success)
        
        return memories
    
    def _create_success_prompt(self, question: str, reasoning: str) -> str:
        return f"""You successfully solved this math problem. Extract 1-3 generalizable strategies that led to success.

PROBLEM: {question}

YOUR SOLUTION: {reasoning}

Extract strategies in this format:

MEMORY 1:
TITLE: <concise strategy name>
DESCRIPTION: <one sentence summary>
CONTENT: <detailed transferable strategy>

MEMORY 2:
...

Focus on WHY the approach worked and how it could apply to similar problems."""
    
    def _create_failure_prompt(self, question: str, reasoning: str, expected: str) -> str:
        return f"""You attempted this math problem but got it wrong. Extract 1-3 lessons about what went wrong.

PROBLEM: {question}

YOUR ATTEMPT: {reasoning}

EXPECTED: {expected}

Extract lessons in this format:

MEMORY 1:
TITLE: <what to avoid or check>
DESCRIPTION: <one sentence summary>
CONTENT: <detailed lesson or preventive strategy>

MEMORY 2:
...

Focus on the mistake and how to prevent it in future similar problems."""
    
    def _parse_memory_items(self, response: str, problem_id: str, success: bool) -> List[MemoryItem]:
        """Parse structured memory items from LLM response"""
        memories = []
        
        # Split by MEMORY markers
        parts = response.split('MEMORY ')
        
        for part in parts[1:]:  # Skip first split (before first MEMORY)
            try:
                # Extract fields
                title = self._extract_field(part, 'TITLE:')
                description = self._extract_field(part, 'DESCRIPTION:')
                content = self._extract_field(part, 'CONTENT:')
                
                if title and description and content:
                    memory = MemoryItem(
                        title=title,
                        description=description,
                        content=content,
                        source_problem_id=problem_id,
                        success=success,
                        created_at=datetime.now().isoformat()
                    )
                    memories.append(memory)
            except:
                continue
        
        return memories
    
    def _extract_field(self, text: str, field_name: str) -> str:
        """Extract field value from formatted text"""
        if field_name not in text:
            return ""
        
        start = text.index(field_name) + len(field_name)
        # Find next field or end
        next_field_markers = ['TITLE:', 'DESCRIPTION:', 'CONTENT:', 'MEMORY ']
        
        end = len(text)
        for marker in next_field_markers:
            if marker in text[start:]:
                candidate_end = start + text[start:].index(marker)
                if candidate_end < end:
                    end = candidate_end
        
        value = text[start:end].strip()
        return value