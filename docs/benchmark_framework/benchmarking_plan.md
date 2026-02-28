# MiniFlow Benchmarking Framework Analysis & Enhancement Plan

## Executive Summary

This document provides a comprehensive analysis of MiniFlow's current benchmark framework and proposes enhancements for rigorous research evidence collection during Phase-1 experiments (KV cache, Flash Attention, Small Language Models).

**Current Status:**
- Comprehensive latency tracking (ASR, LLM, TTS, Total)
- Quality metrics (WER, UTMOS)
- Resource monitoring (GPU memory per stage)
- Statistical aggregation (mean, median, p95, p99)

**Gap Analysis:**
- Missing token-level metrics for LLM analysis
- No model loading time tracking
- No configuration flag recording
- No memory efficiency metrics
- No streaming/perceived latency metrics

**Target:** Research-grade benchmarking suitable for technical publication

---

## Current Benchmark Framework Deep Dive

### Architecture Overview

```
Benchmark/
└── {experiment_name}/
    └── {run_id}/
        ├── config.yml           # Full experiment configuration
        ├── metadata.json        # Experiment metadata
        ├── raw_logs.jsonl       # Per-trial detailed metrics
        ├── summary.json         # Statistical aggregates
        ├── warmup{1,2,3}.json   # Warmup trial data
        └── report.md            # Generated report (optional)
```

### Current Metrics Collection

#### 1. Per-Trial Metrics (raw_logs.jsonl)

**Latency Metrics:**
| Metric | Description | Unit | Current Baseline |
|--------|-------------|------|------------------|
| `asr_latency` | Time for ASR transcription | seconds | 0.43s |
| `llm_latency` | Time for LLM text generation | seconds | 36.08s |
| `tts_latency` | Time for TTS synthesis | seconds | 16.69s |
| `total_latency` | Sum of all stages | seconds | 53.19s |
| `timestamp_start` | Wall-clock start time | Unix timestamp | - |
| `timestamp_end` | Wall-clock end time | Unix timestamp | - |

**Quality Metrics:**
| Metric | Description | Range | Current Baseline |
|--------|-------------|-------|------------------|
| `asr_wer` | Word Error Rate (jiwer) | 0-1 (lower is better) | 0.259 |
| `tts_utmos` | UTMOS speech quality score | 1-5 (higher is better) | 3.15 |

**Resource Metrics:**
| Metric | Description | Unit | Current Baseline |
|--------|-------------|------|------------------|
| `asr_gpu_peak_mem` | Peak GPU memory during ASR | MB | ~2,100 |
| `llm_gpu_peak_mem` | Peak GPU memory during LLM | MB | ~2,200 |
| `tts_gpu_peak_mem` | Peak GPU memory during TTS | MB | ~2,200 |

#### 2. Aggregated Metrics (summary.json)

**Statistical Aggregates per Component:**
```json
{
  "asr_latency": {
    "mean": 0.425,
    "median": 0.436,
    "p95": 0.497,
    "p99": 0.497
  }
}
```

**Quality Averages:**
- `asr_wer_mean`: Mean WER across all samples
- `tts_utmos_mean`: Mean UTMOS score

#### 3. Configuration Tracking (config.yml)

**Preserved Configuration:**
```yaml
asr:
  model_id: openai/whisper-small
  model_name: whisper-small

llm:
  model_id: Qwen/Qwen2.5-3B-Instruct
  model_name: Qwen2.5-3B
  kv_cache: false              # Critical flag
  max_new_tokens: 100
  quantization:
    enabled: true
    load_in_4bit: true
    quant_type: nf4

tts:
  model_id: microsoft/VibeVoice-Realtime-0.5B
  model_name: vibevoice
```

#### 4. Metadata (metadata.json)

```json
{
  "experiment": "baseline",
  "run_id": "baseline_a0dd05",
  "timestamp": 1766370832.045818,
  "num_samples": 20,
  "warmup": 3,
  "dataset_split": "test"
}
```

### Current Framework Strengths

✅ **Comprehensive Coverage:** All three pipeline stages tracked
✅ **Statistical Rigor:** Mean, median, p95, p99 percentiles
✅ **Quality Preservation:** WER and UTMOS ensure optimization doesn't degrade quality
✅ **Resource Monitoring:** GPU memory tracking per stage
✅ **Reproducibility:** Full config preserved for each run
✅ **Warmup Protocol:** 3 samples to stabilize GPU before measurement
✅ **Raw Data Access:** JSONL format allows re-analysis

### Current Framework Gaps (Critical for Research)

#### Gap 1: Token-Level LLM Metrics

**What's Missing:**
- Tokens generated per sample
- Time to first token (TTFT)
- Tokens per second (throughput)
- Prompt tokens count

**Why It Matters:**
- KV cache optimization should improve tokens/sec
- Need direct evidence of token generation speedup
- TTFT critical for perceived latency in streaming scenarios

**Research Evidence Needed:**
```
Baseline (kv_cache: false):
- tokens_generated: 75
- ttft: 0.5s
- tokens_per_sec: 2.1

Optimized (kv_cache: true):
- tokens_generated: 75
- ttft: 0.5s
- tokens_per_sec: 8.5 (4x improvement!)
```

#### Gap 2: Model Loading/Initialization Time

**What's Missing:**
- Time spent loading model from disk to GPU
- Model caching status (was it already in memory?)
- Cold start vs warm start distinction

**Why It Matters:**
- With 6GB VRAM, models must be loaded/unloaded
- Model caching (Phase 2) should eliminate this overhead
- Critical for understanding total system latency

**Research Evidence Needed:**
```
Baseline (no caching):
- model_load_time: 4.2s per query

Optimized (with caching):
- model_load_time: 0.05s per query (cached)
```

#### Gap 3: Configuration Flags Recording

**What's Missing:**
- Actual runtime configuration flags
- Attention implementation used (eager/SDPA/Flash Attention)
- KV cache enabled/disabled
- Flash Attention version

**Why It Matters:**
- Must prove which optimization caused which improvement
- Essential for reproducibility
- Critical for controlled experiments

**Research Evidence Needed:**
```json
"runtime_config": {
  "kv_cache_enabled": true,
  "attention_implementation": "flash_attention_2",
  "model_size": "1.5B",
  "quantization_bits": 4,
  "flash_attn_version": "2.5.8"
}
```

#### Gap 4: Memory Efficiency & Fragmentation Metrics

**What's Missing:**

**1. Basic Memory Metrics:**
- Memory allocated vs memory reserved
- Memory efficiency ratios
- Cache utilization rates

**2. Detailed Fragmentation Metrics:**
- Waste ratio: (reserved - allocated) / reserved
- Inactive blocks (unused but reserved memory blocks)
- Segment count (total memory segments in pool)
- Pool fraction (fragmentation indicator from PyTorch)
- Inactive block sizes (MB)

**3. Model-Specific Memory Impact:**
- Memory state after each pipeline stage
- Fragmentation accumulation over multiple samples
- Correlation between fragmentation and subsequent model loading time
- TTS model comparison (XTTS vs VibeVoice fragmentation patterns)

**Why It Matters:**
- **6GB VRAM constraint:** Hard limit requires optimal memory usage
- **Flash Attention validation:** Claims 20-30% memory savings - need evidence
- **Fragmentation investigation:** Explain why VibeVoice showed 59% faster LLM latency than XTTS
- **Optimization proof:** Show that memory optimizations actually reduce pressure
- **Root cause analysis:** Understand WHY certain configurations perform better

**Research Evidence Needed:**

*Basic Memory Efficiency:*
```
Baseline (eager attention):
- memory_allocated_mb: 2048
- memory_reserved_mb: 3072
- efficiency: 66.7%

Optimized (Flash Attention):
- memory_allocated_mb: 1843 (-10%)
- memory_reserved_mb: 2457 (-20%)
- efficiency: 75.0% (+8.3%)
```

*Detailed Fragmentation Analysis:*
```
After XTTS TTS (sample 1):
- memory_allocated_mb: 2048
- memory_reserved_mb: 3072
- efficiency: 66.7%
- fragmentation_waste_ratio: 33.3%
- inactive_blocks: 47
- inactive_blocks_size_mb: 892
- segment_count: 32
- next_llm_load_time: 4.8s

After XTTS TTS (sample 5):
- memory_allocated_mb: 2048
- memory_reserved_mb: 4096
- efficiency: 50.0%
- fragmentation_waste_ratio: 50.0%  ← Worsening!
- inactive_blocks: 74
- inactive_blocks_size_mb: 1843
- segment_count: 58
- next_llm_load_time: 8.2s  ← Slower loading!

After VibeVoice TTS (sample 5):
- memory_allocated_mb: 2048
- memory_reserved_mb: 2457
- efficiency: 83.3%
- fragmentation_waste_ratio: 16.7%  ← Stable!
- inactive_blocks: 14
- inactive_blocks_size_mb: 312
- segment_count: 22
- next_llm_load_time: 2.1s  ← Fast loading!
```

*Key Finding:* Fragmentation increases over XTTS samples, correlating with slower LLM load times. VibeVoice maintains stable, low fragmentation.

#### Gap 5: Streaming/Perceived Latency

**What's Missing:**
- Time to first audio (TTS streaming)
- Time to first token (LLM streaming)
- Chunk-based latency for streaming scenarios

**Why It Matters:**
- Total latency vs perceived latency are different
- User hears first audio while rest generates
- Critical for "real-time" feeling even if total latency is higher

**Research Evidence Needed:**
```
Baseline (batch mode):
- time_to_first_audio: 53.2s (total latency)

Optimized (streaming):
- time_to_first_token: 0.8s
- time_to_first_audio: 2.5s
- total_latency: 15.3s (but user hears audio at 2.5s!)
```

#### Gap 6: Throughput & Scalability

**What's Missing:**
- Queries per second (QPS)
- Samples per second
- Concurrent user handling metrics

**Why It Matters:**
- Single-user vs multi-user performance differs
- Research needs to show scalability
- Important for deployment planning

#### Gap 7: Statistical Variance Analysis

**What's Missing:**
- Standard deviation
- Confidence intervals
- Distribution histograms

**Why It Matters:**
- Mean alone doesn't show variance
- Research requires statistical significance (p < 0.05)
- Need to prove improvements are statistically meaningful

---

## Enhanced Metrics for Phase-1 Experiments

### Priority 1: Essential Metrics (Must Implement)

#### 1. LLM Token Metrics

**New Fields:**
```python
@dataclass
class Metrics:
    # ... existing fields ...

    # Token generation metrics
    llm_tokens_generated: int = 0          # Total output tokens
    llm_prompt_tokens: int = 0             # Input prompt tokens
    llm_ttft: float = 0.0                  # Time to first token (seconds)
    llm_tokens_per_sec: float = 0.0        # Generation throughput
    llm_time_per_token: float = 0.0        # Average time per token
```

**Implementation:**
```python
# In run_llm() function
start_time = time.time()
first_token_time = None
tokens_generated = 0

# Use TextIteratorStreamer for token-level tracking
streamer = TextIteratorStreamer(tokenizer)
generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=config["max_new_tokens"])

# Start generation in background thread
thread = Thread(target=model.generate, kwargs=generation_kwargs)
thread.start()

# Track tokens as they arrive
generated_text = ""
for new_text in streamer:
    if first_token_time is None:
        first_token_time = time.time()
    tokens_generated += 1
    generated_text += new_text

end_time = time.time()

# Calculate metrics
llm_ttft = first_token_time - start_time
llm_latency = end_time - start_time
llm_tokens_per_sec = tokens_generated / (llm_latency - llm_ttft)
llm_time_per_token = (llm_latency - llm_ttft) / tokens_generated if tokens_generated > 0 else 0
```

**Research Value:**
- Prove KV cache effectiveness (should see 4-5x tokens/sec improvement)
- Baseline for token generation speed
- Enables streaming architecture design (Phase 3)

#### 2. Model Loading Metrics

**New Fields:**
```python
@dataclass
class Metrics:
    # ... existing fields ...

    # Model loading metrics
    model_load_time: float = 0.0           # Time to load from disk to GPU
    model_cached: bool = False             # Was model already in GPU memory?
    model_transfer_time: float = 0.0       # Time for CPU→GPU transfer (if cached)
```

**Implementation:**
```python
# Track model loading time
load_start = time.time()

if model_in_cache:
    model_cached = True
    transfer_start = time.time()
    model = cache_manager.load_model("llm", device=DEVICE)
    model_transfer_time = time.time() - transfer_start
    model_load_time = 0.0  # Effectively instant
else:
    model_cached = False
    model = AutoModelForCausalLM.from_pretrained(...)
    model_load_time = time.time() - load_start
    model_transfer_time = 0.0
```

**Research Value:**
- Quantify overhead of sequential loading
- Prove model caching effectiveness (Phase 2)
- Critical for understanding 6GB VRAM constraints

#### 3. Configuration Flags Recording

**Enhanced Summary:**
```json
{
  "experiment": "baseline",
  "run_id": "baseline_a0dd05",
  "runtime_config": {
    "kv_cache_enabled": true,
    "attention_implementation": "flash_attention_2",
    "flash_attn_version": "2.5.8",
    "model_size": "1.5B",
    "quantization_bits": 4,
    "max_new_tokens": 100,
    "compute_dtype": "bfloat16"
  },
  "system_info": {
    "gpu": "NVIDIA GeForce RTX 3060",
    "cuda_version": "13.0",
    "pytorch_version": "2.9.1",
    "transformers_version": "4.57.1",
    "vram_gb": 6
  }
}
```

**Implementation:**
```python
import transformers
import torch

# In benchmark runner
system_info = {
    "gpu": torch.cuda.get_device_name(0),
    "cuda_version": torch.version.cuda,
    "pytorch_version": torch.__version__,
    "transformers_version": transformers.__version__,
    "vram_gb": torch.cuda.get_device_properties(0).total_memory // (1024**3)
}

runtime_config = {
    "kv_cache_enabled": config["llm"].get("use_cache", False),
    "attention_implementation": config["llm"].get("attn_implementation", "eager"),
    "model_size": config["llm"]["model_name"],
    "quantization_bits": 4 if config["llm"]["quantization"]["load_in_4bit"] else 16,
    "max_new_tokens": config["llm"]["max_new_tokens"]
}
```

**Research Value:**
- Prove controlled experiments (change one variable at a time)
- Essential for reproducibility
- Critical for correlating config changes with performance changes

#### 4. Memory Efficiency & Fragmentation Metrics

**New Fields:**
```python
@dataclass
class Metrics:
    # ... existing fields ...

    # Basic Memory Metrics
    memory_allocated_mb: float = 0.0           # Actually used memory
    memory_reserved_mb: float = 0.0            # Reserved by PyTorch
    memory_efficiency: float = 0.0             # allocated / reserved ratio

    # Fragmentation Metrics
    fragmentation_waste_ratio: float = 0.0     # (reserved - allocated) / reserved
    inactive_blocks: int = 0                    # Number of unused reserved blocks
    inactive_blocks_size_mb: float = 0.0       # Total size of inactive blocks
    segment_count: int = 0                      # Total memory segments
    pool_fraction: float = 0.0                 # PyTorch fragmentation indicator

    # Model Loading Correlation
    memory_state_before_load: str = ""         # "clean" or "fragmented"
```

**Implementation:**
```python
import torch

def get_detailed_memory_stats(device=0):
    """Get comprehensive memory statistics including fragmentation."""
    # Basic PyTorch memory stats
    memory_allocated = torch.cuda.memory_allocated(device) / (1024**2)
    memory_reserved = torch.cuda.memory_reserved(device) / (1024**2)

    # Detailed memory statistics from PyTorch
    stats = torch.cuda.memory_stats(device)

    # Fragmentation metrics
    fragmentation_waste_ratio = (
        (memory_reserved - memory_allocated) / memory_reserved
        if memory_reserved > 0 else 0
    )

    inactive_blocks = stats.get('inactive_split.all.alloc_count', 0)
    inactive_blocks_size_mb = (
        stats.get('inactive_split.all.allocated_bytes.all.peak', 0) / (1024**2)
    )
    segment_count = stats.get('segment.count', 0)
    pool_fraction = stats.get('pool_fraction', 0.0)

    return {
        "memory_allocated_mb": memory_allocated,
        "memory_reserved_mb": memory_reserved,
        "memory_efficiency": memory_allocated / memory_reserved if memory_reserved > 0 else 0,
        "fragmentation_waste_ratio": fragmentation_waste_ratio,
        "inactive_blocks": inactive_blocks,
        "inactive_blocks_size_mb": inactive_blocks_size_mb,
        "segment_count": segment_count,
        "pool_fraction": pool_fraction
    }

# Usage after each pipeline stage
memory_stats = get_detailed_memory_stats()

# Classify memory state for correlation analysis
if memory_stats["fragmentation_waste_ratio"] > 0.3 or memory_stats["inactive_blocks"] > 50:
    memory_state_before_load = "fragmented"
else:
    memory_state_before_load = "clean"
```

**Research Value:**
- **Prove Flash Attention memory savings:** Basic metrics (efficiency, reserved memory)
- **Explain performance differences:** Fragmentation metrics reveal WHY certain configs work better
- **6GB VRAM optimization:** Track how optimizations affect memory pressure
- **Root cause analysis:** Correlate fragmentation with model loading times
- **Publication evidence:** Comprehensive memory behavior data for technical report

**Example Analysis:**
```python
# Compare fragmentation between TTS models
def analyze_tts_fragmentation_impact(results):
    """Analyze how TTS model choice affects subsequent LLM performance."""
    xtts_results = [r for r in results if r["tts_model"] == "xtts"]
    vibevoice_results = [r for r in results if r["tts_model"] == "vibevoice"]

    print(f"XTTS - Avg fragmentation: {mean([r['fragmentation_waste_ratio'] for r in xtts_results]):.1%}")
    print(f"XTTS - Avg inactive blocks: {mean([r['inactive_blocks'] for r in xtts_results]):.0f}")
    print(f"XTTS - Avg next LLM load time: {mean([r['next_model_load_time'] for r in xtts_results]):.2f}s")

    print(f"VibeVoice - Avg fragmentation: {mean([r['fragmentation_waste_ratio'] for r in vibevoice_results]):.1%}")
    print(f"VibeVoice - Avg inactive blocks: {mean([r['inactive_blocks'] for r in vibevoice_results]):.0f}")
    print(f"VibeVoice - Avg next LLM load time: {mean([r['next_model_load_time'] for r in vibevoice_results]):.2f}s")
```

### Priority 2: Important Metrics (Should Implement)

#### 5. Streaming Latency Metrics

**New Fields:**
```python
@dataclass
class Metrics:
    # ... existing fields ...

    # Streaming metrics
    time_to_first_token: float = 0.0       # First LLM token generated
    time_to_first_audio: float = 0.0       # First TTS audio chunk ready
    streaming_chunks: int = 0              # Number of audio chunks streamed
    streaming_latency: float = 0.0         # Total streaming time
```

**Research Value:**
- Differentiate total vs perceived latency
- Critical for "real-time" claims
- Enables streaming architecture (Phase 3)

#### 6. GPU Utilization Metrics

**New Fields:**
```python
@dataclass
class Metrics:
    # ... existing fields ...

    # GPU utilization
    gpu_utilization_avg: float = 0.0       # Average GPU utilization %
    gpu_utilization_peak: float = 0.0      # Peak GPU utilization %
    gpu_power_draw_watts: float = 0.0      # Average power consumption
```

**Implementation:**
```python
# Use nvidia-ml-py or pynvml
import pynvml

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)

# Sample GPU utilization during inference
utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
gpu_utilization_avg = utilization.gpu

# Get power draw
power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
```

**Research Value:**
- Identify underutilization (can we batch more?)
- Energy efficiency metrics
- Hardware optimization evidence

#### 7. Quality Regression Analysis

**New Fields:**
```python
@dataclass
class Metrics:
    # ... existing fields ...

    # Quality metrics
    response_length_chars: int = 0         # Response character count
    response_coherence_score: float = 0.0  # Manual or automated score
    prompt_transcript_match: float = 0.0   # Did LLM address the input?
```

**Research Value:**
- Ensure optimizations don't degrade quality
- Track correlation between speed and quality
- Essential for publication-ready results

---

## Phase-1 Experiment Design

### Experiment 1: KV Cache Impact Assessment

**Objective:** Quantify the impact of enabling KV cache on LLM performance

**Hypothesis:** Enabling KV cache will reduce LLM latency by 50-80% and increase tokens/sec by 4-5x

**Experimental Design:**
```yaml
# Baseline configuration
experiment_1a:
  llm:
    use_cache: false  # Baseline
    max_new_tokens: 100

# Test configuration
experiment_1b:
  llm:
    use_cache: true   # Optimized
    max_new_tokens: 100
```

**Metrics to Compare:**
| Metric | Baseline | Test | Expected Change | Evidence Type |
|--------|----------|------|-----------------|---------------|
| llm_latency | 36s | 7-15s | -60% to -80% | Primary |
| llm_tokens_per_sec | 2.1 | 8-10 | +300% to +400% | Primary |
| llm_ttft | 0.5s | 0.5s | No change | Secondary |
| llm_gpu_peak_mem | 2200MB | 2500MB | +10-15% | Acceptable trade-off |
| asr_wer_mean | 0.259 | 0.259 | No change | Quality preserved |
| tts_utmos_mean | 3.15 | 3.15 | No change | Quality preserved |

**Sample Size:** 30 samples per configuration (60 total)

**Statistical Analysis:**
- Paired t-test to compare means (p < 0.05)
- Effect size calculation (Cohen's d)
- Confidence intervals for latency reduction

### Experiment 2: Attention Implementation Comparison

**Objective:** Compare eager, SDPA, and Flash Attention 2 implementations

**Hypothesis:** Flash Attention 2 will provide 10-20% speedup and 20-30% memory reduction over eager attention

**Experimental Design:**
```yaml
# Test 1: Eager attention (baseline)
experiment_2a:
  llm:
    attn_implementation: "eager"

# Test 2: SDPA
experiment_2b:
  llm:
    attn_implementation: "sdpa"

# Test 3: Flash Attention 2
experiment_2c:
  llm:
    attn_implementation: "flash_attention_2"
```

**Metrics to Compare:**
| Metric | Eager | SDPA | Flash Attn 2 | Evidence |
|--------|-------|------|--------------|----------|
| llm_latency | 36s | 32-34s | 28-32s | Speedup |
| memory_reserved_mb | 3072 | 2765 | 2457 | Memory savings |
| memory_efficiency | 66.7% | 72% | 75% | Efficiency gain |
| fragmentation_waste_ratio | 33.3% | 27.5% | 16.7% | Less fragmentation |
| inactive_blocks | 68 | 45 | 22 | Fewer unused blocks |
| llm_tokens_per_sec | 2.1 | 2.3 | 2.6 | Throughput |

**Sample Size:** 20 samples per configuration (60 total)

**Analysis:**
- ANOVA to compare three groups
- Post-hoc Tukey HSD for pairwise comparisons

### Experiment 3: Model Size Impact

**Objective:** Evaluate trade-off between model size and performance

**Hypothesis:** Qwen2.5-1.5B will provide 2x speedup with minimal quality degradation

**Experimental Design:**
```yaml
# Baseline: 3B model
experiment_3a:
  llm:
    model_id: Qwen/Qwen2.5-3B-Instruct
    model_name: Qwen2.5-3B

# Test: 1.5B model
experiment_3b:
  llm:
    model_id: Qwen/Qwen2.5-1.5B-Instruct
    model_name: Qwen2.5-1.5B
```

**Metrics to Compare:**
| Metric | 3B Model | 1.5B Model | Change |
|--------|----------|------------|---------|
| llm_latency | 36s | 18s | -50% |
| llm_gpu_peak_mem | 2200MB | 1500MB | -32% |
| model_load_time | 4.5s | 2.5s | -44% |
| asr_wer_mean | 0.259 | 0.259 | No change (ASR same) |
| response_coherence | Manual eval | Manual eval | Track regression |

**Sample Size:** 30 samples per configuration + manual quality evaluation of 10 random samples

### Experiment 4: Combined Optimizations

**Objective:** Measure cumulative impact of all Phase-1 optimizations

**Experimental Design:**
```yaml
# Ultimate optimized configuration
experiment_4:
  llm:
    model_id: Qwen/Qwen2.5-1.5B-Instruct
    use_cache: true
    attn_implementation: "flash_attention_2"
    max_new_tokens: 75  # Optimized based on response length analysis
```

**Target Results:**
| Metric | Baseline | Optimized | Improvement |
|--------|----------|-----------|-------------|
| total_latency | 53.2s | 8-12s | -77% to -85% |
| llm_latency | 36.1s | 4-6s | -83% to -89% |
| asr_wer_mean | 0.259 | 0.259 | Preserved |
| tts_utmos_mean | 3.15 | 3.10-3.20 | Within variance |

---

## Statistical Rigor Requirements

### Sample Size Calculation

For research publication quality:
- **Minimum:** 30 samples per configuration (statistical significance)
- **Recommended:** 50 samples per configuration (robust results)
- **Conservative:** 100 samples per configuration (high confidence)

**Current:** 20 samples - adequate for preliminary results, increase to 30 for publication

### Statistical Tests

**For Two-Group Comparisons (e.g., KV cache on/off):**
```python
from scipy import stats

# Paired t-test (if same samples)
t_stat, p_value = stats.ttest_rel(baseline_latencies, optimized_latencies)

# Or independent t-test (if different samples)
t_stat, p_value = stats.ttest_ind(baseline_latencies, optimized_latencies)

# Effect size (Cohen's d)
def cohens_d(x, y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)
```

**For Multiple Groups (e.g., attention implementations):**
```python
# One-way ANOVA
f_stat, p_value = stats.f_oneway(eager_times, sdpa_times, flash_times)

# Post-hoc Tukey HSD
from statsmodels.stats.multicomp import pairwise_tukeyhsd
tukey = pairwise_tukeyhsd(endog=all_times, groups=groups, alpha=0.05)
```

### Confidence Intervals

Report 95% confidence intervals for all latency metrics:
```python
def confidence_interval(data, confidence=0.95):
    n = len(data)
    mean = np.mean(data)
    std_err = stats.sem(data)
    h = std_err * stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean - h, mean + h
```

### Variance Reporting

Include in summary.json:
```json
{
  "llm_latency": {
    "mean": 36.08,
    "median": 33.98,
    "std": 8.45,
    "p95": 55.00,
    "p99": 55.00,
    "ci_95": [32.15, 40.01]
  }
}
```

---

## Evidence Collection Strategy for Technical Report

### Phase-1 Evidence Matrix

| Experiment | Research Question | Key Metrics | Evidence Type | Success Criteria |
|------------|-------------------|-------------|---------------|------------------|
| **1. KV Cache** | Does KV cache reduce LLM latency? | llm_latency, llm_tokens_per_sec | Quantitative | >50% latency reduction, p<0.05 |
| **2. Flash Attention** | Does Flash Attention improve speed/memory? | llm_latency, memory_reserved_mb, memory_efficiency, fragmentation_waste_ratio, inactive_blocks | Quantitative | >10% speedup, >15% memory reduction, >30% fragmentation reduction |
| **3. Model Size** | Can we use smaller LLM without quality loss? | llm_latency, asr_wer, manual evaluation | Mixed | 2x speedup, WER unchanged, quality acceptable |
| **4. Combined** | What's cumulative impact? | total_latency, component breakdown | Quantitative | >75% total latency reduction |

### Controlled Experiment Protocol

**Principle:** Change only ONE variable at a time

**Protocol:**
1. **Baseline Run:** Document all current metrics
2. **Single Change:** Modify only target parameter (e.g., enable KV cache)
3. **Test Run:** Measure with same dataset, same hardware
4. **Comparison:** Statistical test for significance
5. **Rollback:** Return to baseline before next experiment

**Example Sequence:**
```
Run 1: Baseline (kv_cache=false, eager attention, 3B model)
Run 2: Test KV cache (kv_cache=true, eager attention, 3B model) ← Only change
Run 3: Test Flash Attention (kv_cache=false, flash_attention, 3B model) ← Only change
Run 4: Test 1.5B model (kv_cache=false, eager attention, 1.5B model) ← Only change
Run 5: Combined (kv_cache=true, flash_attention, 1.5B model) ← All changes
```

### Data Collection Checklist

For each experiment run:

**Before Starting:**
- [ ] Document hardware specifications (GPU, VRAM, CUDA version)
- [ ] Record software versions (PyTorch, Transformers, Flash Attention)
- [ ] Verify configuration matches experimental design
- [ ] Clear GPU cache and reset peak memory stats
- [ ] Run warmup samples (3-5) to stabilize GPU

**During Run:**
- [ ] Monitor GPU memory usage (watch for OOM)
- [ ] Log any errors or anomalies
- [ ] Record wall-clock time for total experiment duration

**After Run:**
- [ ] Verify all expected output files created
- [ ] Check summary.json has all required fields
- [ ] Validate raw_logs.jsonl has correct number of entries
- [ ] Run comparison script to generate report
- [ ] Archive results with descriptive folder name

### Report Generation Workflow

**After Each Experiment:**
```bash
# 1. Run experiment
uv run python src/scripts/run_experiment.py --config configs/experiment_1b.yml

# 2. Compare to baseline
uv run python src/scripts/compare_benchmarks.py \
    Benchmark/baseline/baseline_a0dd05/summary.json \
    Benchmark/baseline/baseline_newrun/summary.json \
    --title "KV Cache Impact Analysis"

# 3. Generate detailed report
uv run python src/scripts/generate_experiment_report.py \
    --exp_dir Benchmark/baseline/baseline_newrun \
    --output reports/experiment_1b_report.md

# 4. Archive results
cp -r Benchmark/baseline/baseline_newrun results_archive/exp1_kv_cache/
```

---

## Implementation Roadmap

### Week 1: Enhanced Metrics Implementation

**Task 1.1: Extend Metrics Dataclass**
- Add token metrics (llm_tokens_generated, llm_ttft, llm_tokens_per_sec)
- Add model loading metrics (model_load_time, model_cached)
- Add memory efficiency metrics (memory_allocated_mb, memory_reserved_mb, memory_efficiency)

**Task 1.2: Update process_sample() Function**
- Implement token-level tracking for LLM
- Add model loading time measurement
- Capture detailed memory metrics

**Task 1.3: Enhance Summary Generation**
- Add runtime_config section
- Add system_info section
- Include variance metrics (std, confidence intervals)

**Task 1.4: Testing & Validation**
- Run test experiment with new metrics
- Verify all metrics populated correctly
- Compare output format to expected research format

### Week 2: Experiment Execution

**Task 2.1: Baseline Establishment**
- Run 30-sample baseline with current config
- Document all metrics
- Create "ground truth" for comparisons

**Task 2.2: Experiment 1 (KV Cache)**
- Enable KV cache, run 30 samples
- Statistical comparison to baseline
- Document results

**Task 2.3: Experiment 2 (Flash Attention)**
- Install Flash Attention 2
- Run comparison (eager vs SDPA vs Flash Attention)
- Document memory and speed improvements

**Task 2.4: Experiment 3 (Model Size)**
- Test Qwen2.5-1.5B
- Evaluate quality trade-off
- Document findings

**Task 2.5: Experiment 4 (Combined)**
- Run ultimate optimized configuration
- Measure cumulative improvements
- Validate quality preservation

### Week 3: Analysis & Documentation

**Task 3.1: Statistical Analysis**
- Calculate effect sizes for all experiments
- Generate confidence intervals
- Perform significance testing

**Task 3.2: Visualization**
- Create latency comparison charts
- Generate memory usage plots
- Plot token throughput improvements

**Task 3.3: Report Writing**
- Document methodology
- Present results with statistical rigor
- Discuss implications for real-time systems

---

## Research Paper Structure (Future)

### Abstract
- 250 words summarizing problem, methods, results, implications

### 1. Introduction
- Problem: High latency in speech-to-speech systems
- Constraint: Consumer hardware (6GB VRAM)
- Objective: Achieve near real-time performance

### 2. Related Work
- Existing S2S systems and their latencies
- Optimization techniques (KV cache, Flash Attention, quantization)
- Gap: Limited research on resource-constrained optimization

### 3. Methodology
- System architecture (ASR→LLM→TTS pipeline)
- Benchmark framework (metrics, statistical methods)
- Experimental design (controlled experiments)

### 4. Results
#### 4.1 KV Cache Impact
- 50-80% LLM latency reduction
- 4-5x token throughput improvement

#### 4.2 Flash Attention Impact
- 10-20% speedup
- 20-30% memory reduction

#### 4.3 Model Size Trade-off
- 2x speedup with 1.5B model
- Quality preservation analysis

#### 4.4 Combined Optimizations
- 77-85% total latency reduction
- Perceived latency analysis

### 5. Discussion
- Implications for real-time S2S systems
- Limitations (6GB VRAM constraint)
- Future work (streaming, hardware upgrades)

### 6. Conclusion
- Summary of key findings
- Practical recommendations

---

## Conclusion

This enhanced benchmark framework provides:

✅ **Research-grade metrics** for rigorous evaluation
✅ **Controlled experimental design** for causal inference
✅ **Statistical rigor** suitable for publication
✅ **Comprehensive evidence collection** for technical report
✅ **Clear success criteria** for each optimization

**Next Steps:**
1. Implement enhanced Metrics dataclass
2. Execute Phase-1 experiments
3. Collect evidence for technical report
4. Iterate based on findings

**Success Criteria for Phase-1:**
- Reduce total latency from 53s to 10-15s
- Prove each optimization contributes measurably
- Maintain quality metrics (WER, UTMOS)
- Generate publication-ready statistical analysis

---

*Document Version: 1.0*
*Created: January 2026*
*Status: Ready for Implementation*
*Target: Research-grade benchmarking for MiniFlow optimization*
