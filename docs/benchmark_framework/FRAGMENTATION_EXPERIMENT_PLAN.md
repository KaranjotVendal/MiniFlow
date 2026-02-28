# Memory Fragmentation Investigation Plan

## Objective

Investigate GPU memory fragmentation differences between XTTS and VibeVoice TTS models, and create a reusable memory profiling framework for MiniFlow experiments.

This investigation aims to understand why VibeVoice showed 59% faster LLM latency despite no code changes - the hypothesis is that XTTS causes GPU memory fragmentation which slows down subsequent LLM model loading.

---

## Background

### The Problem

During benchmark comparison, VibeVoice showed significantly faster LLM latency (36s vs 89s) despite identical LLM code. This unexpected improvement suggests an external factor - memory fragmentation is the leading hypothesis.

### Hypothesis

XTTS (Coqui TTS) uses different memory allocation patterns than VibeVoice (transformers-native), causing GPU memory fragmentation. When LLM model reloads after XTTS, the allocator struggles to find contiguous memory blocks, causing slowdowns.

### Evidence Needed

1. Quantified fragmentation metrics (waste ratio, inactive blocks)
2. Memory state comparison after TTS operations
3. Correlation between fragmentation and LLM load time

---

## Deliverables

### 1. `src/utils/memory_profiler.py`

Reusable memory profiling utilities for GPU memory analysis.

### 2. `src/scripts/fragmentation_experiment.py`

Controlled experiment comparing fragmentation between XTTS and VibeVoice.

---

## Module 1: `src/utils/memory_profiler.py`

### Core Functions

| Function | Purpose | Returns |
|----------|---------|---------|
| `log_memory_state(tag, device=0)` | Quick snapshot (allocated, reserved, max) | dict with MB values |
| `log_memory_summary(tag, device=0, abbreviated=True)` | Full `torch.cuda.memory_summary()` output | None (prints) |
| `log_memory_stats(tag, device=0)` | Detailed `torch.cuda.memory_stats()` | None (prints) |
| `get_fragmentation_metrics(device=0)` | Calculate waste ratio, inactive blocks | dict |

### Timing Functions

| Function | Purpose |
|----------|---------|
| `time_model_load(model_loader, tag, device=0)` | Measure load time + memory impact |
| `time_inference(model, input_data, tag="Inference")` | Measure inference time |

### Decorators

| Decorator | Purpose |
|-----------|---------|
| `@time_operation` | Log execution time of any function |
| `@track_memory` | Track memory before/after function call |

### Context Managers

| Context Manager | Purpose |
|-----------------|---------|
| `MemoryTracker(tag, device=0)` | Track memory changes within a block |

### Component-Specific Profiling

| Function | Purpose |
|----------|---------|
| `profile_asr(audio_tensor, sampling_rate, model_id, device)` | Profile ASR with memory tracking |
| `profile_llm(prompt, model_id, device, max_new_tokens)` | Profile LLM with memory tracking |
| `profile_tts(text, model_name, model_id, device, speaker)` | Profile TTS with memory tracking |

---

## Module 2: `src/scripts/fragmentation_experiment.py`

### Experiment Design

```python
def run_fragmentation_comparison(num_samples=5):
    """
    Compare memory fragmentation between XTTS and VibeVoice.

    Steps:
    1. Log initial GPU state (fresh start)
    2. Run 5 XTTS samples, log memory after each TTS unload
    3. Run 5 VibeVoice samples, log memory after each TTS unload
    4. Compare fragmentation metrics between runs
    5. Generate summary report
    """
```

### Key Metrics to Track

| Metric | Source | Interpretation |
|--------|--------|----------------|
| Waste ratio (%) | `(reserved - allocated) / reserved * 100` | Higher = more fragmentation |
| Inactive blocks | `memory_stats()['inactive_split.all.alloc_count']` | More = fragmentation |
| Total segments | `memory_stats()['segment.count']` | More = fragmentation |
| LLM load time | `time_model_load()` | Correlates with fragmentation |

### Expected Results

| Scenario | XTTS Fragmentation | VibeVoice Fragmentation | Conclusion |
|----------|-------------------|------------------------|------------|
| Hypothesis TRUE | Increases over samples (15% -> 25%) | Stays stable (5-6%) | XTTS causes fragmentation |
| Hypothesis FALSE | Similar between models | Similar between models | Fragmentation not the cause |

### Output Format

```markdown
# Memory Fragmentation Experiment Report

## XTTS Run (5 samples)

| Sample | Post-TTS Fragmentation | Inactive Blocks |
|--------|------------------------|-----------------|
| 1      | 15.2%                  | 47              |
| 2      | 18.7%                  | 52              |
| 3      | 22.1%                  | 61              |
| 4      | 24.5%                  | 68              |
| 5      | 26.8%                  | 74              |

## VibeVoice Run (5 samples)

| Sample | Post-TTS Fragmentation | Inactive Blocks |
|--------|------------------------|-----------------|
| 1      | 5.3%                   | 12              |
| 2      | 6.1%                   | 14              |
| 3      | 5.8%                   | 13              |
| 4      | 5.5%                   | 12              |
| 5      | 5.9%                   | 14              |

## Analysis

- XTTS fragmentation increases by 77% over 5 samples
- VibeVoice stays stable (5-6% range)
- LLM load time correlates with fragmentation
- CONCLUSION: Memory fragmentation IS the cause of LLM latency differences
```

---

## Usage Examples

### Quick Memory Snapshot

```python
from src.utils.memory_profiler import log_memory_state

log_memory_state("Before TTS")
# Output:
# === Before TTS ===
#   Allocated: 2048.5 MB
#   Reserved:  3072.0 MB
#   Max Alloc: 4096.0 MB
#   Waste:     33.3%
```

### Track a Block of Code

```python
from src.utils.memory_profiler import MemoryTracker

with MemoryTracker("TTS Generation"):
    waveform = tts_model.generate(text)
# Output shows memory delta before/after
```

### Get Fragmentation Metrics

```python
from src.utils.memory_profiler import get_fragmentation_metrics

metrics = get_fragmentation_metrics()
print(f"Fragmentation: {metrics['waste_ratio_pct']:.1f}%")
print(f"Inactive blocks: {metrics['inactive_blocks']}")
```

### Time Model Loading

```python
from src.utils.memory_profiler import time_model_load

timing = time_model_load(
    lambda: AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda"),
    "LLM"
)
print(f"Load time: {timing['load_time']:.2f}s")
print(f"Peak memory: {timing['mem_peak_mb']:.0f} MB")
```

---

## Implementation Steps

### Step 1: Create `src/utils/memory_profiler.py`

1. Implement core logging functions
2. Implement timing functions
3. Implement decorators
4. Implement context managers
5. Implement component-specific profiling functions

### Step 2: Create `src/scripts/fragmentation_experiment.py`

1. Implement experiment runner
2. Add XTTS and VibeVoice sample runners
3. Generate comparison report
4. Add analysis and conclusions

### Step 3: Run Experiment

```bash
uv run python src/scripts/fragmentation_experiment.py
```

### Step 4: Document Results

Update `memory_fragementation_experiment.md` with findings.

---

## Files Modified/Created

| File | Action |
|------|--------|
| `src/utils/memory_profiler.py` | Create |
| `src/scripts/fragmentation_experiment.py` | Create |
| `FRAGMENTATION_EXPERIMENT_PLAN.md` | Create (this file) |

---

## Learning Outcomes

### GPU Memory Basics

- **Allocated memory**: Actively used by tensors
- **Reserved memory**: Cached in memory pool (available for reuse)
- **Fragmentation**: When reserved memory is split into small, unusable blocks

### What Causes Fragmentation

| Cause | Effect |
|-------|--------|
| Many small allocations | Creates scattered used blocks |
| Frequent alloc/free | Leaves holes in memory |
| Different tensor sizes | Poor fit in remaining space |
| CUDA context switches | Fragmentation over time |

### How to Detect Fragmentation

1. **Waste ratio**: `(reserved - allocated) / reserved`
2. **Inactive blocks**: Unused but reserved blocks
3. **Allocation failures**: When allocator can't find contiguous memory

### Optimization Strategies (Post-Experiment)

1. **Aggressive cleanup**: `synchronize()` + `empty_cache()` after each model
2. **CPU caching**: Keep models in system RAM, transfer to GPU as needed
3. **Pipeline reordering**: Run LLM before TTS to avoid fragmentation
4. **Memory pooling**: Pre-allocate memory blocks for reuse

---

## Related Files

- `memory_fragementation_experiment.md` - Existing experiment notes
- `LLM_LATENCY_INVESTIGATION.md` - LLM latency analysis
- `memory_read_me.md` - Memory overview
- `src/sts_pipeline.py` - Main pipeline
- `src/tts/tts_pipelines.py` - TTS implementations

---

## Questions for Future Investigation

1. Is fragmentation the PRIMARY cause or just a contributing factor?
2. Can we reduce XTTS fragmentation without switching models?
3. What about VibeVoice fragmentation over longer runs?
4. Should we implement CPU-based model caching?
5. Does pipeline reordering solve the issue?

---

## References

- [PyTorch CUDA Memory Documentation](https://docs.pytorch.org/docs/stable/cuda.html)
- [CUDA Memory Management Best Practices](https://pytorch.org/docs/stable/notes/cuda.html#memory-management)
- [Memory Fragmentation in Deep Learning](https://pytorch.org/docs/stable/notes/cuda.html#memory-management)

---

*Plan created: January 2026*
*Status: Shelved for future implementation*
