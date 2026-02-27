**Title: MiniFlow: Low-Latency Speech-to-Speech Agent with Accent-Robust Optimizations**

**Vision:** Develop an end-to-end, real-time speech-to-speech conversational agent that handles diverse accents with low latency, showcasing advanced transformer optimizations for production-grade Voice AI. The biggest thing I am trying to get done through this project is get an end-to-end working on a GPU with 6gb vram.

---

## Benchmark Results

### Executive Summary

MiniFlow's speech-to-speech pipeline has been benchmarked using two different TTS backends: **XTTS** (Coqui TTS) and **VibeVoice-Realtime-0.5B** (Microsoft). The VibeVoice configuration demonstrated **55% reduction in total latency** (118s → 53s) across 20 test samples from the GLOBE dataset.

### Performance Comparison

| Component | XTTS Run (baseline_67ecd4) | VibeVoice Run (baseline_a0dd05) | Improvement |
|-----------|---------------------------|----------------------------------|-------------|
| **ASR Latency** | 1.255s (mean) | 0.425s (mean) | **66% faster** |
| **LLM Latency** | 88.99s (mean) | 36.08s (mean) | **59% faster** |
| **TTS Latency** | 27.82s (mean) | 16.69s (mean) | **40% faster** |
| **Total Latency** | 118.06s (mean) | 53.19s (mean) | **55% faster** |

### Quality Metrics

| Metric | XTTS | VibeVoice | Change |
|--------|------|-----------|--------|
| **ASR WER** | 0.259 | 0.259 | Same |
| **TTS UTMOS** | 3.30 | 3.15 | -4.5% |


### Key Finding: Memory Fragmentation Issue

**Investigation Summary:** The dramatic LLM latency improvement (59% faster) was traced to GPU memory fragmentation, not code or input differences.

**Root Cause:**
- XTTS uses Coqui TTS library with different memory allocation patterns
- Leaves GPU in highly fragmented state after each TTS operation
- LLM model reloads every sample, suffering from allocator inefficiency
- VibeVoice uses transformers-native architecture with predictable memory patterns

**Evidence:**
- Identical samples processed in same order (WER values match exactly)
- No changes to LLM code between runs
- Warmup samples show consistent pattern (VibeVoice faster even without prior TTS)
- VibeVoice has higher average memory usage but better performance

**Impact:** This finding explains why model loading overhead varies significantly based on preceding operations. For consistent pipeline performance, model caching strategies must account for memory fragmentation.

---

## Roadmap

### done - Pipeline Stages

| Component | Status | Details |
|-----------|--------|---------|
| **ASR (Automatic Speech Recognition)** | done | Whisper-small model with forced decoder IDs for English transcription |
| **LLM ** | done | Qwen2.5-3B-Instruct with 4-bit quantization (nf4) |
| **TTS - XTTS** | done | Coqui TTS multi-lingual model with voice cloning |
| **TTS - VibeVoice** | done | Microsoft VibeVoice-Realtime-0.5B with streaming support |
| **Transformers Compatibility** | done | Bridging VibeVoice (transformers 4.51.3) with MiniFlow (4.57.3) via compatibility shims |
| **Benchmark Framework** | done | Structured experiment runner with metrics tracking |
| **Configuration System** | done | YAML-based config files for different pipeline modes |

### in progress - Optimizations

| Item | Status | Description |
|------|--------|-------------|
| **Memory Fragmentation Investigation** | in progress | Analyzing GPU memory patterns between TTS models to understand performance variations |


### planned - Future Improvements

| Item | Priority | Description |
|------|----------|-------------|
| **CPU-Based Model Caching** | High | Implement model caching in system RAM (not VRAM) to avoid 6GB VRAM limitation while reducing disk I/O overhead |
| **LLM Model Caching** | High | Cache Qwen2.5-3B model across samples to eliminate 40-90s reload overhead per sample |
| **Pipeline Reordering Experiments** | Medium | Test ASR→LLM→TTS order to verify memory fragmentation hypothesis |
| **Aggressive Memory Clearing** | Medium | Add torch.cuda.empty_cache() and synchronization before model loads |
| **ASR Model Caching** | Medium | Cache Whisper model similar to LLM caching strategy |
| **Additional TTS Backend** | Low | Implement CosyVoice as alternative TTS option |
| **Streaming TTS Support** | Low | Full streaming audio output for real-time voice response |

---

## Running Benchmarks

```bash
# Run full benchmark suite
uv run python src/benchmark/runner.py

# Run single sample for testing
uv run python -m src.debug_scripts.debug_tts
uv run python -m src.debug_scripts.debug_llm
uv run python -m src.debug_scripts.debug_asr

# Run memory experiments
uv run python src/scripts/run_memory_experiments.py
```

## Configuration

Pipeline configurations are defined in `configs/`:
- `baseline.yml` - Original XTTS pipeline config
- `2_TTS-to-vibevoice.yml` - VibeVoice TTS config

## Docker Profiles

Use separate compose modes for prod-like and dev workflows:

```bash
# Prod-like container runtime (no bind mounts, no reload)
docker compose up --build

# Dev runtime (bind mounts + autoreload)
docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build
```
