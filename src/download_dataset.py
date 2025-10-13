# src/download_dataset.py
from datasets import load_dataset
import json
import re

def prepare_gsm8k():
    """Download and prepare GSM8K dataset"""
    dataset = load_dataset("gsm8k", "main")
    
    # Take 200 training examples for memory accumulation
    # Take 100 test examples for evaluation
    train_data = list(dataset['train'].select(range(200)))
    test_data = list(dataset['test'].select(range(100)))
    
    # Format for our experiment
    def format_item(item, idx):
        return {
            'id': f'gsm8k_{idx}',
            'question': item['question'],
            'answer': item['answer'],
            'expected_value': extract_answer(item['answer'])
        }
    
    def extract_answer(answer_text):
        """Extract final numeric answer from GSM8K format"""
        # GSM8K answers are like "#### 42"
        if '####' in answer_text:
            return answer_text.split('####')[-1].strip()
        return None
    
    train_formatted = [format_item(item, idx) for idx, item in enumerate(train_data)]
    test_formatted = [format_item(item, idx+1000) for idx, item in enumerate(test_data)]
    
    # Save
    with open('data/train_problems.json', 'w') as f:
        json.dump(train_formatted, f, indent=2)
    
    with open('data/test_problems.json', 'w') as f:
        json.dump(test_formatted, f, indent=2)
    
    print(f"Prepared {len(train_formatted)} training problems")
    print(f"Prepared {len(test_formatted)} test problems")
    print(f"Example problem: {train_formatted[0]['question']}")


def prepare_math_dataset():
    """Download and prepare MATH dataset (harder than GSM8K)"""
    
    print("Loading MATH dataset from competition_math...")
    try:
        dataset = load_dataset("competition_math")
    except Exception as e:
        print(f"Error loading competition_math: {e}")
        print("\nTrying alternative: qwedsacf_competition_math...")
        try:
            dataset = load_dataset("qwedsacf/competition_math")
        except Exception as e2:
            print(f"Error loading qwedsacf_competition_math: {e2}")
            print("\nNeither dataset found. Please install manually or use GSM8K.")
            return
    
    print(f"Available splits: {list(dataset.keys())}")
    
    # Inspect first item to understand structure
    if 'train' in dataset:
        first_item = dataset['train'][0]
        print(f"\nDataset structure (first item keys): {list(first_item.keys())}")
        print(f"Sample item: {first_item}")
    
    # Determine field names
    train_split = dataset['train'] if 'train' in dataset else dataset[list(dataset.keys())[0]]
    
    # Check field names from first item
    sample = train_split[0]
    problem_field = 'problem' if 'problem' in sample else 'question'
    solution_field = 'solution' if 'solution' in sample else 'answer'
    level_field = 'level' if 'level' in sample else 'difficulty'
    type_field = 'type' if 'type' in sample else 'subject'
    
    print(f"\nUsing field mappings:")
    print(f"  Problem: {problem_field}")
    print(f"  Solution: {solution_field}")
    print(f"  Level: {level_field}")
    print(f"  Type: {type_field}")
    
    # Filter to medium difficulty problems
    target_levels = ['Level 3', 'Level 4', '3', '4', 'Medium', 'medium']
    
    train_filtered = []
    for item in train_split:
        level = str(item.get(level_field, ''))
        if any(target in level for target in target_levels):
            train_filtered.append(item)
    
    print(f"\nFiltered to {len(train_filtered)} medium-difficulty problems")
    
    # Handle test split
    if 'test' in dataset:
        test_split = dataset['test']
        test_filtered = []
        for item in test_split:
            level = str(item.get(level_field, ''))
            if any(target in level for target in target_levels):
                test_filtered.append(item)
    else:
        # Split train into train/test
        print("No test split found, using last 20% of train data for testing")
        split_idx = int(len(train_filtered) * 0.8)
        test_filtered = train_filtered[split_idx:]
        train_filtered = train_filtered[:split_idx]
    
    # Take 200 train, 100 test
    train_data = train_filtered[:200]
    test_data = test_filtered[:100]
    
    def extract_boxed_answer(solution):
        """MATH answers are in \\boxed{...} format"""
        # Try to find boxed answer
        match = re.search(r'\\boxed\{([^}]+)\}', solution)
        if match:
            answer = match.group(1)
            # Clean up LaTeX formatting
            answer = answer.replace('\\', '')
            answer = answer.replace('$', '')
            answer = answer.strip()
            return answer
        
        # Fallback: look for final number
        numbers = re.findall(r'-?\d+\.?\d*', solution)
        if numbers:
            return numbers[-1]
        
        return None
    
    def format_item(item, idx):
        problem = item.get(problem_field, '')
        solution = item.get(solution_field, '')
        level = item.get(level_field, 'Unknown')
        problem_type = item.get(type_field, 'Unknown')
        
        expected = extract_boxed_answer(solution)
        return {
            'id': f'math_{idx}',
            'question': problem,
            'answer': solution,
            'expected_value': expected,
            'level': str(level),
            'type': str(problem_type)
        }
    
    train_formatted = [format_item(item, idx) for idx, item in enumerate(train_data)]
    test_formatted = [format_item(item, idx+1000) for idx, item in enumerate(test_data)]
    
    # Filter out problems where we couldn't extract the answer
    train_formatted = [p for p in train_formatted if p['expected_value'] is not None]
    test_formatted = [p for p in test_formatted if p['expected_value'] is not None]
    
    # Save
    with open('data/train_problems.json', 'w') as f:
        json.dump(train_formatted, f, indent=2)
    
    with open('data/test_problems.json', 'w') as f:
        json.dump(test_formatted, f, indent=2)
    
    print(f"\n✓ Prepared {len(train_formatted)} training problems")
    print(f"✓ Prepared {len(test_formatted)} test problems")
    print(f"\nExample problem: {train_formatted[0]['question'][:100]}...")
    print(f"Example answer: {train_formatted[0]['expected_value']}")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'math':
        print("Preparing MATH dataset (harder)...")
        prepare_math_dataset()
    else:
        print("Preparing GSM8K dataset...")
        print("To use MATH dataset instead, run: python src/download_dataset.py math")
        prepare_gsm8k()