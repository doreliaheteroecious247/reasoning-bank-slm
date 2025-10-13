## Project Overview

**ReasoningBank SLM** is an experiment that applies Google Research's `ReasoningBank` framework to small language models (≤3B parameters). This project tests whether the memory-based self-improvement techniques from the original ReasoningBank paper also benefit much smaller, less capable models.

### What is ReasoningBank?

ReasoningBank is a novel memory framework introduced by Google AI (September 2025) that enables LLM agents to learn continuously from their interaction history. Instead of discarding insights after each task, ReasoningBank:

1. **Distills reasoning strategies** from both successful and failed experiences
2. **Stores them as reusable memory items** with semantic embeddings  
3. **Retrieves relevant memories** during new tasks to inform decision-making
4. **Integrates new learnings back** into the memory bank, enabling self-evolution

The framework combines memory awareness with test-time scaling (MaTTS), establishing a symbiotic relationship where better memory guides more effective scaling, and abundant experiences create higher-quality memory.

### Why Small Language Models?

While ReasoningBank was demonstrated on larger models, this experiment investigates whether similar gains apply to small models (1-4B parameters). Small models have advantages but often struggle with:
- Insufficient internal knowledge retention
- Limited reasoning capabilities  
- Difficulty transferring learning across tasks

If ReasoningBank can improve small model performance, it could:
- Enable more capable reasoning with minimal compute
- Create more accessible AI for resource-constrained environments
- Demonstrate that strategic memory retrieval is more valuable than model scale alone

## Technical Implementation

### Core Components

- **`src/memory.py`**: JSON-based memory storage for reasoning strategies
- **`src/retrieval/retriever.py`**: Semantic search with answer leak protection
- **`src/extraction/extractor.py`**: LLM-powered strategy extraction from trajectories  
- **`src/llm_client.py`**: OpenAI-compatible client for llama-server
- **`src/judge/evaluator.py`**: Dataset-aware math solution evaluation
- **`src/run_phase1.py`**: Experiment orchestration comparing baseline vs. memory-augmented performance

### Memory Schema

Each memory item contains:
```json
{
  "title": "Strategy name",
  "description": "One-sentence summary", 
  "content": "Detailed transferable strategy",
  "source_problem_id": "Origin problem",
  "success": true,
  "created_at": "ISO timestamp",
  "embedding": [0.1, 0.2, ...]
}
```

### Answer Leak Protection

Critical for fair evaluation, the retrieval system filters memories containing:
- Numeric values matching the test item's expected answer
- High similarity (>90%) to the full question text

## Experiments

### Phase 1: Does Retrieval Help?

Currently implemented and measures whether memory retrieval improves a 4B model's `competition_math` math performance.

**Methodology:**
1. Build memory bank from training set trajectories 
2. Test baseline accuracy without memory
3. Test memory-augmented accuracy with retrieval
4. Compare with statistical significance testing

**Results materialize as:**
- `results/phase1_results.json`: Complete trial records
- `results/phase1_summary.json`: Statistical analysis
- `results/phase1_accuracy.png`: Performance visualization

### Planned Phases

**Phase 2: Can It Self-Improve?**
- Harvest successful reasoning traces
- Fine-tune model on consolidated strategies
- Test on previously failed problems

**Phase 3: Does It Compound?**  
- Run multiple improvement cycles
- Measure compounding effects
- Analyze memory quality evolution

## Setup & Usage

### Prerequisites

1. llama-server running Qwen3-4B-Thinking-2507
2. Python 3.8+ with dependencies:
   ```bash
   pip install torch sentence-transformers datasets numpy pandas tqdm scikit-learn matplotlib requests
   ```

### Quick Start

1. **Download Data:**
   ```bash
   python src/download_dataset.py math
   ```

2. **Start Model Server:**
   ```bash
   cd models
   llama-server -m qwen3-4b-thinking-2507-q8_0.gguf -c 4096 --port 8080 -ngl 99
   ```

3. **Run Phase 1 Experiment:**
   ```bash
   python src/run_phase1.py
   ```

4. **Analyze Results:**
   ```bash
   python src/analyze_results.py
   ```

### Configuration

- **Model**: Qwen3-4B-Thinking-2507 (or downgrade to 0.5B for testing)
- **Dataset**: qwedsacf/competition_math
- **Memory Model**: Qwen3-0.6B-Embedding embeddings
- **Seed Strategy**: Deterministic seeding from first N training problems

## Key Findings

*[Results to be updated after Phase 1 completion]*

**Phase 1 Performance:**
- Baseline accuracy: TBD
- Memory-augmented accuracy: TBD  
- Absolute improvement: TBD
- Statistical significance: TBD

**Memory Quality Analysis:**
- Total memories accumulated: TBD
- Average retrieval relevance: TBD
- Strategy reusability rate: TBD

## Artifacts & Results

- `memory_bank/reasoning_bank.json`: Complete memory collection
- `results/`: Statistical analysis and visualizations  
- `logs/`: Experimental logs and debugging output
- `data/`: Processed GSM8K datasets

## Success Criteria

**Phase 1 Success:**
- Memory retrieval improves accuracy by >3%
- Improvements are statistically significant (95% CI)
- No evidence of answer leakage artifacts

**Overall Success:**  
- Small models achieve ReasoningBank-reported gains
- Improvements compound across multiple cycles
- Cross-domain strategy transfer demonstrated

## Related Work

- **ReasoningBank paper**: [arXiv:2509.25140](https://arxiv.org/abs/2509.25140) 
- **qwedsacf/competition_math Dataset**: [Hugging Face](https://huggingface.co/datasets/qwedsacf/competition_math)
- **Qwen Models**: [Hugging Face Collection](https://huggingface.co/collections/Qwen)


## License

MIT License - See repository for details.