# src/judge/evaluator.py
import sys
import re
from typing import Dict, Optional
sys.path.append('src')
from llm_client import JudgeClient

class MathJudge:
    """Evaluate if math solution is correct with GSM8K and MATH dataset support"""
    
    def __init__(self, llm_client: JudgeClient):
        self.llm = llm_client
    
    def is_correct(self, predicted: str, expected: str) -> bool:
        """Robust numeric comparison with GSM8K and MATH format awareness"""
        try:
            pred_num = self._extract_number(predicted)
            exp_num = self._extract_number(expected)
            
            if pred_num is None or exp_num is None:
                # Try string comparison for non-numeric answers (e.g., algebraic expressions)
                pred_clean = self._normalize_text(str(predicted))
                exp_clean = self._normalize_text(str(expected))
                return pred_clean == exp_clean
            
            # Allow small floating point error for integer answers
            if isinstance(pred_num, int) and isinstance(exp_num, int):
                return pred_num == exp_num
            
            # For floats, use relative tolerance
            return abs(pred_num - exp_num) < 0.01
        except Exception as e:
            print(f"Judge error: {e}")
            return False
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison (for non-numeric answers)"""
        text = text.lower().strip()
        # Remove LaTeX formatting
        text = text.replace('\\', '')
        text = text.replace('$', '')
        text = text.replace('{', '')
        text = text.replace('}', '')
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def _extract_number(self, text: str) -> Optional[float]:
        """Extract number from text with support for GSM8K and MATH formats"""
        if not text:
            return None
        
        text = str(text).strip()
        
        # MATH dataset format: \boxed{42}
        if '\\boxed' in text:
            match = re.search(r'\\boxed\{([^}]+)\}', text)
            if match:
                boxed_content = match.group(1)
                return self._clean_number(boxed_content)
        
        # GSM8K gold format: "Some text #### 42"
        if '####' in text:
            answer_part = text.split('####')[-1].strip()
            return self._clean_number(answer_part)
        
        # Our model format: "ANSWER: 42"
        if 'ANSWER:' in text.upper():
            answer_part = text.upper().split('ANSWER:')[-1].strip()
            return self._clean_number(answer_part)
        
        # Fallback: extract the last number with possible commas and decimals
        # Handle fractions directly (e.g., "3/4")
        if '/' in text and len(text.split('/')) == 2:
            return self._clean_number(text)
        
        # Find all numbers including those with commas
        matches = re.findall(r'[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?', text)
        if matches:
            # Return the last match found in the text
            return self._clean_number(matches[-1])
        
        return None
    
    def _clean_number(self, text: str) -> Optional[float]:
        """Normalize number formatting and handle special cases"""
        if not text:
            return None
        
        # Remove LaTeX and common formatting characters
        cleaned = text.replace('\\', '').replace('$', '').strip()
        cleaned = cleaned.replace(',', '').replace('%', '')
        
        # Handle fractions: "3/4" -> 0.75
        if '/' in cleaned:
            try:
                parts = cleaned.split('/')
                if len(parts) == 2:
                    numerator = float(parts[0].strip())
                    denominator = float(parts[1].strip())
                    if denominator != 0:
                        result = numerator / denominator
                        # Return as int if it's a whole number
                        return int(result) if result.is_integer() else result
            except:
                pass
        
        # Handle decimals and integers
        try:
            num = float(cleaned)
            # Return as int if it's a whole number
            return int(num) if num.is_integer() else num
        except:
            return None
    
    def evaluate_with_reasoning(self, question: str, solution: str, 
                                 expected_answer: str) -> Dict:
        """Full evaluation with detailed feedback"""
        is_correct = self.is_correct(solution, expected_answer)
        
        predicted_num = self._extract_number(solution)
        expected_num = self._extract_number(expected_answer)
        
        return {
            'success': is_correct,
            'predicted': solution,
            'predicted_number': predicted_num,
            'expected': expected_answer,
            'expected_number': expected_num,
            'reasoning': f"Predicted: {predicted_num}, Expected: {expected_num}"
        }


# Unit tests for the judge
def test_judge():
    """Test judge parsing with common formatting quirks"""
    from llm_client import LlamaServerClient
    
    # Mock client (judge doesn't actually use it for numeric comparison)
    judge = MathJudge(None)
    
    print("Testing judge...")
    
    # Test GSM8K format
    assert judge._extract_number("Some reasoning #### 42") == 42, "GSM8K format failed"
    print("✓ GSM8K format")
    
    # Test MATH format
    assert judge._extract_number("Therefore \\boxed{42}") == 42, "MATH boxed format failed"
    print("✓ MATH boxed format")
    
    # Test our format
    assert judge._extract_number("ANSWER: 42") == 42, "ANSWER format failed"
    print("✓ ANSWER format")
    
    # Test with formatting
    assert judge._extract_number("The answer is $1,234.56") == 1234.56, "Currency formatting failed"
    assert judge._extract_number("Total: 1,000") == 1000, "Comma formatting failed"
    print("✓ Currency and comma formatting")
    
    # Test fractions
    assert judge._extract_number("3/4") == 0.75, "Fraction failed"
    assert judge._extract_number("1/2") == 0.5, "Simple fraction failed"
    print("✓ Fractions")
    
    # Test multiple numbers (should get last)
    assert judge._extract_number("First we get 10, then 20, final answer is 30") == 30, "Last number extraction failed"
    print("✓ Multiple numbers (last)")
    
    # Test integer vs float
    assert judge._extract_number("42.0") == 42, "Integer from float failed"
    assert judge._extract_number("42.5") == 42.5, "Actual float failed"
    print("✓ Integer/float handling")
    
    # Test comparison
    assert judge.is_correct("42", "42"), "Simple match failed"
    assert judge.is_correct("ANSWER: 42", "#### 42"), "Format mismatch failed"
    assert judge.is_correct("1/2", "0.5"), "Fraction comparison failed"
    assert not judge.is_correct("42", "43"), "Should not match"
    print("✓ Comparison logic")
    
    # Test critical case: 2500 vs 25000
    assert not judge.is_correct("2500", "25000"), "2500 should NOT equal 25000"
    assert not judge.is_correct("ANSWER: 2500", "#### 25000"), "2500 should NOT equal 25000 (formatted)"
    print("✓ Critical: 2500 ≠ 25000")
    
    print("\n✓ All judge tests passed!")
    return True

if __name__ == '__main__':
    test_judge()