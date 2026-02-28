# MiniFlow Real-Time Optimization Roadmap

## Executive Summary

This document provides a comprehensive roadmap for optimizing MiniFlow's end-to-end speech-to-speech pipeline to achieve near real-time or real-time performance.

**Current Status (Verified by Benchmarks):**
- Pipeline: ASR → LLM → TTS
- Total Latency: 53s mean / 81s p95 (VibeVoice) | 118s (XTTS)
- Primary Bottleneck: LLM (68% of total time: 36s out of 53s)
- Secondary Bottleneck: TTS (31% of total time: 17s)
- ASR: Already optimized (0.4s - not a bottleneck)
- VRAM Constraint: 6GB (limits simultaneous model loading)
- KV Cache: Currently DISABLED (major opportunity)

**Target Latencies:**
- **Near Real-time:** < 2s total latency (achievable with optimization)
- **Streaming Perception:** < 500ms time-to-first-audio (achievable with streaming)
- **True Real-time:** < 500ms total latency (requires hardware upgrade or major architectural changes)

---

## Benchmark Analysis

### Current Performance (VibeVoice Pipeline)

| Component | Mean Latency | % of Total | p95 Latency | Status |
|-----------|--------------|------------|-------------|---------|
| **ASR** | 0.43s | 0.8% | 0.50s | ✅ Optimized |
| **LLM** | **36.08s** | **67.8%** | **55.00s** | 🔴 Critical Bottleneck |
| **TTS** | **16.69s** | **31.4%** | **25.92s** | 🟡 Secondary Bottleneck |
| **Total** | **53.19s** | 100% | **81.26s** | - |

### Comparison: XTTS vs VibeVoice

| Metric | XTTS | VibeVoice | Improvement |
|--------|------|-----------|-------------|
| Total Latency | 118.06s | 53.19s | **-54.9%** |
| LLM Latency | 88.99s | 36.08s | -59.5% |
| TTS Latency | 27.82s | 16.69s | -40.0% |
| ASR Latency | 1.26s | 0.43s | -66.1% |
| ASR WER | 0.259 | 0.259 | No change |
| TTS UTMOS | 3.304 | 3.150 | -4.7% |

**Key Insight:** Switching from XTTS to VibeVoice achieved a 55% latency reduction with minimal quality loss. Similar optimizations on the LLM side could yield comparable improvements.

---

## Optimization Strategy Overview

### The Real-Time Challenge

With 6GB VRAM, we cannot fit all models in GPU memory simultaneously. The current sequential approach (load ASR → unload → load LLM → unload → load TTS) adds significant overhead.

**Three-Pronged Approach:**
1. **Speed up individual components** (especially LLM)
2. **Keep models cached** (reduce load/unload overhead)
3. **Stream and overlap** (hide latency through architecture)

### Critical Finding: KV Cache is Disabled

Your current config has `use_cache: false` for the LLM. This is costing you **50-80% performance penalty**. Enabling KV cache should be the **first priority**.

**Why KV Cache Matters:**
- Without KV cache: O(n²) complexity, recomputes all previous tokens
- With KV cache: O(n) complexity, reuses cached key-value pairs
- **Expected speedup: 2-5x for token generation**

---

## Phase-by-Phase Roadmap

### Phase 1: Quick Wins (Week 1-2)
**Goal: 53s → 10-15s total latency**
**Effort: Low | Risk: Low | Impact: High**

#### Task 1.1: Enable KV Cache (CRITICAL - Do First)
- **File:** `configs/2_TTS-to-vibevoice.yml`
- **Change:** `use_cache: false` → `use_cache: true`
- **Expected Impact:** LLM latency 36s → 7-15s (50-80% reduction)
- **Implementation Time:** 5 minutes
- **Validation:** Re-run benchmark and compare

#### Task 1.2: Enable Flash Attention 2
- **Prerequisite:** `pip install flash-attn --no-build-isolation`
- **Change:** Add `attn_implementation="flash_attention_2"` to model config
- **Expected Impact:** 10-20% additional LLM speedup
- **Implementation Time:** 30 minutes
- **Note:** Requires compatible GPU (most modern NVIDIA GPUs supported)

#### Task 1.3: Optimize max_new_tokens
- **Current:** Likely generating 100 tokens
- **Action:** Analyze average response length from benchmarks
- **Change:** Reduce to 50-75 tokens if responses are shorter
- **Expected Impact:** Proportional reduction in LLM latency
- **Implementation Time:** 15 minutes

#### Task 1.4: Test Smaller LLM (Qwen2.5-1.5B)
- **Current:** Qwen2.5-3B-Instruct
- **Test:** Qwen2.5-1.5B-Instruct
- **Expected Impact:** 2x speedup (36s → 18s)
- **Trade-off:** Slightly reduced capability but still good quality
- **Implementation Time:** 30 minutes
- **Decision Point:** Compare quality vs speed trade-off

#### Task 1.5: Enable Streaming TTS
- **VibeVoice Support:** Already supports streaming mode
- **Change:** Use streaming inference instead of batch
- **Expected Impact:** Time-to-first-audio: 17s → 300ms
- **Implementation Time:** 2-4 hours
- **Benefit:** Dramatically improves perceived latency

**Phase 1 Expected Outcome:**
- Best case: 0.4s (ASR) + 5s (LLM with KV cache + smaller model) + 2s (TTS) = **~7-8s total**
- Conservative: **10-15s total** (still 70% improvement)
- Perceived latency with streaming: **< 1s** (time-to-first-audio)

---

### Phase 2: Model Caching & Memory Optimization (Week 3-4)
**Goal: 10-15s → 5-8s total latency**
**Effort: Medium | Risk: Medium | Impact: High**

#### Task 2.1: Implement CPU-Based Model Caching
**Problem:** Current pipeline loads/unloads models for each sample
**Solution:** Keep models in CPU RAM, transfer to GPU when needed

**Implementation Steps:**
1. Create model cache manager in `src/model_cache.py`
2. Load models to CPU memory on startup
3. Move to GPU only during inference
4. Return to CPU after inference

**Expected Impact:** Eliminates 2-5s load time per query
**VRAM Usage:** Can keep 2-3 models "warm" in CPU RAM

#### Task 2.2: Keep LLM in GPU Memory (If Possible)
With 4-bit quantization:
- Qwen2.5-3B: ~3GB VRAM
- Qwen2.5-1.5B: ~1.5GB VRAM

**Strategy:** Keep LLM always in GPU, cache ASR/TTS in CPU
**Expected Impact:** LLM ready instantly (0s load time)
**Trade-off:** Less VRAM for ASR/TTS, requires offloading

#### Task 2.3: Implement Conversation State Management
**Purpose:** Pass KV cache between conversation turns

**Implementation:**
1. Store KV cache in Redis/memory after each turn
2. Load KV cache at start of next turn
3. Implement context window management (sliding window for long conversations)

**Expected Impact:** Subsequent queries much faster (5s → 2s for cached context)

**Phase 2 Expected Outcome:**
- Eliminate model loading overhead
- Subsequent queries: 0.4s (ASR) + 3s (cached LLM) + 2s (TTS) = **~5-6s**
- First query: Similar to Phase 1 (~10s)

---

### Phase 3: Streaming Architecture (Week 5-6)
**Goal: Perceived latency 5s → < 1s (time-to-first-audio)**
**Effort: High | Risk: Medium | Impact: Transformative**

This phase shifts from reducing total latency to **hiding latency** through streaming.

#### Task 3.1: Streaming LLM Output
**Current:** Wait for full LLM response before TTS
**Optimized:** Stream tokens as they're generated

**Implementation:**
1. Use `TextIteratorStreamer` from transformers
2. Buffer first N tokens (e.g., 10-20)
3. Start TTS as soon as buffer has enough context
4. Continue streaming tokens while TTS works

**Expected Impact:** Time-to-first-audio: 36s → 500ms-1s
**Architecture Change:** Requires async/await throughout pipeline

#### Task 3.2: Streaming ASR
**Current:** Wait for full transcription
**Optimized:** Start processing partial transcriptions

**Implementation:**
1. Use Whisper with `return_timestamps=True`
2. Stream partial results every 100-200ms
3. Start LLM with partial transcription, refine as more audio arrives

**Expected Impact:** Start LLM 200-500ms earlier

#### Task 3.3: Pipeline Overlapping
**Current:** ASR → wait → LLM → wait → TTS
**Optimized:** Concurrent execution where possible

```
Timeline:
t=0ms:      ASR starts on audio chunk 1
t=100ms:    ASR chunk 1 done, LLM starts with partial text
t=200ms:    ASR chunk 2 done, LLM updates context, TTS starts with first tokens
t=300ms:    TTS first audio chunk ready (USER HEARS AUDIO!)
t=500ms:    Full response streaming continues
```

**Expected Impact:** Perceived latency < 500ms (time-to-first-audio)
**Full Response:** Still 3-5s, but user doesn't care

#### Task 3.4: WebSocket Streaming API
**Implementation:**
1. Extend FastAPI with WebSocket endpoint (`/ws/stream`)
2. Stream audio chunks as they're generated
3. Client plays audio immediately (no waiting for full file)

**Phase 3 Expected Outcome:**
- Time-to-first-audio: **< 500ms-1s** (real-time perception)
- Total latency: Still 3-5s, but streaming masks it
- User experience: **Conversational real-time**

---

### Phase 4: Advanced Optimizations (Week 7+)
**Goal: True real-time < 2s total latency**
**Effort: High | Risk: High | Impact: Incremental**

#### Task 4.1: Speculative Decoding
**Concept:** Use small "draft" model to predict tokens, verify with main model

**Implementation:**
1. Load Qwen2.5-0.5B as draft model (small, fast)
2. Draft model generates 5-10 candidate tokens
3. Main model verifies all in parallel
4. Accept matching tokens, regenerate mismatches

**Expected Impact:** 2-3x LLM speedup (5s → 2-3s)
**Memory Cost:** +~1GB for draft model
**Complexity:** High - requires custom generation loop

#### Task 4.2: Aggressive Quantization (3-bit)
**Current:** 4-bit NF4 quantization
**Test:** 3-bit quantization with bitsandbytes

**Expected Impact:** 30% speedup, slightly lower quality
**Decision Point:** Only if quality is still acceptable after Phase 1-3

#### Task 4.3: Continuous Batching
**Use Case:** Multiple concurrent users
**Implementation:** Batch multiple requests together for higher throughput

**Expected Impact:** Better throughput, not latency
**When to Implement:** Only if supporting multiple users

#### Task 4.4: Hardware Upgrade Evaluation
**Option A:** 8GB VRAM ($200-400 GPU)
- Can cache ASR + LLM simultaneously
- Reduces latency by 2-3s

**Option B:** 12GB VRAM ($400-600 GPU)
- Can cache all models simultaneously
- Near-instant response (< 2s total)

**Option C:** 24GB VRAM (A10, A5000)
- Everything in memory + speculative decoding
- True real-time (< 1s) achievable

**Recommendation:** Evaluate cost vs engineering time trade-off

---

## Decision Matrix

### Which Path Should You Take?

| Your Situation | Recommended Path | Expected Outcome | Timeline |
|----------------|-----------------|------------------|----------|
| Need quick wins, limited time | Phase 1 only | 10-15s latency | 1-2 weeks |
| Need good performance, have time | Phases 1-2 | 5-8s latency | 3-4 weeks |
| Need best UX, can invest | Phases 1-3 | <1s perceived | 5-6 weeks |
| Need true real-time | Phases 1-4 + Hardware | <2s total | 8+ weeks |
| Have budget, need fast results | Phase 1 + Hardware upgrade | <5s immediately | 1 week |

### Quality vs Speed Trade-offs

| Component | Current | Phase 1 Options | Quality Impact |
|-----------|---------|-----------------|----------------|
| LLM Model | Qwen2.5-3B | Qwen2.5-1.5B | Slight reduction |
| KV Cache | Disabled | Enabled | None (pure speedup) |
| Quantization | 4-bit | 3-bit | Slight reduction |
| TTS | VibeVoice | Streaming mode | None (same model) |
| ASR | Whisper-small | Keep as-is | None |

**Recommendation:** Start with KV cache (no quality loss) and test 1.5B model (evaluate quality yourself).

---

## Implementation Checklist

### Phase 1: Quick Wins

- [ ] **1.1 Enable KV Cache**
  - [ ] Modify `configs/2_TTS-to-vibevoice.yml`
  - [ ] Set `use_cache: true`
  - [ ] Run benchmark to verify improvement

- [ ] **1.2 Add Flash Attention 2**
  - [ ] Install: `pip install flash-attn --no-build-isolation`
  - [ ] Update model loading code
  - [ ] Test compatibility

- [ ] **1.3 Optimize max_new_tokens**
  - [ ] Analyze current response lengths from logs
  - [ ] Adjust to 50-75 if appropriate
  - [ ] Test for quality

- [ ] **1.4 Test Qwen2.5-1.5B**
  - [ ] Create new config file
  - [ ] Run benchmark
  - [ ] Evaluate quality vs speed
  - [ ] Decision: keep or revert

- [ ] **1.5 Enable Streaming TTS**
  - [ ] Research VibeVoice streaming API
  - [ ] Implement streaming inference
  - [ ] Measure time-to-first-audio

### Phase 2: Model Caching

- [ ] **2.1 Create Model Cache Manager**
  - [ ] Implement `src/model_cache.py`
  - [ ] CPU→GPU transfer logic
  - [ ] Cache eviction strategy

- [ ] **2.2 Keep LLM in GPU**
  - [ ] Modify pipeline to not unload LLM
  - [ ] Monitor VRAM usage
  - [ ] Handle OOM scenarios

- [ ] **2.3 Conversation State**
  - [ ] Store KV cache between turns
  - [ ] Implement sliding window
  - [ ] Test multi-turn conversations

### Phase 3: Streaming

- [ ] **3.1 Streaming LLM**
  - [ ] Implement `TextIteratorStreamer`
  - [ ] Token buffering logic
  - [ ] Async pipeline

- [ ] **3.2 Streaming ASR**
  - [ ] Partial transcription support
  - [ ] Early LLM triggering

- [ ] **3.3 Pipeline Overlap**
  - [ ] Async/await throughout
  - [ ] Concurrent stage execution
  - [ ] Buffer management

- [ ] **3.4 WebSocket API**
  - [ ] `/ws/stream` endpoint
  - [ ] Audio chunk streaming
  - [ ] Client-side playback

### Phase 4: Advanced

- [ ] **4.1 Speculative Decoding**
  - [ ] Draft model integration
  - [ ] Verification logic
  - [ ] Benchmark

- [ ] **4.2 Hardware Upgrade**
  - [ ] Evaluate GPU options
  - [ ] Cost-benefit analysis
  - [ ] Implement if approved

---

## Expected Outcomes by Phase

| Phase | Total Latency | Time-to-First-Audio | Implementation Time | Risk |
|-------|---------------|---------------------|---------------------|------|
| Baseline | 53s | 53s | - | - |
| Phase 1 | 10-15s | 10-15s | 1-2 weeks | Low |
| Phase 2 | 5-8s | 5-8s | +2 weeks | Medium |
| Phase 3 | 3-5s | **< 1s** | +2 weeks | Medium |
| Phase 4 | < 2s | **< 500ms** | +4 weeks | High |

**Key Insight:** Phase 3 (streaming) gives the best user experience even if total latency is only moderately improved. For conversational AI, **perceived latency matters more than total latency**.

---

## Risk Mitigation

### Technical Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| KV cache breaks with 4-bit quantization | High | Test thoroughly, have rollback plan |
| Streaming increases memory usage | Medium | Monitor VRAM, implement backpressure |
| Smaller LLM quality unacceptable | Medium | A/B test before committing |
| Flash Attention incompatible | Low | Fall back to eager attention |

### Implementation Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Scope creep | High | Stick to phase boundaries |
| Underestimating effort | Medium | Add 50% buffer to estimates |
| Breaking existing functionality | High | Comprehensive testing after each phase |

---

## Monitoring & Validation

### Key Metrics to Track

1. **Latency Metrics**
   - Mean/p95/p99 for each component
   - Time-to-first-audio (Phase 3+)
   - End-to-end total latency

2. **Quality Metrics**
   - ASR WER (should stay < 0.30)
   - TTS UTMOS (should stay > 3.0)
   - LLM coherence (manual evaluation)

3. **Resource Metrics**
   - GPU VRAM usage
   - CPU RAM usage
   - Model load/unload frequency

### Validation Strategy

**After Each Phase:**
1. Run full benchmark suite
2. Compare to baseline
3. Evaluate quality impact
4. Decision: proceed to next phase or iterate

**Success Criteria:**
- Phase 1: 50%+ latency reduction without quality loss
- Phase 2: Eliminate model loading overhead
- Phase 3: < 1s time-to-first-audio
- Phase 4: < 2s total latency

---

## Alternative: Hardware-First Approach

If you have budget and need results quickly:

**Week 1:**
1. Upgrade to 12GB VRAM GPU ($400-600)
2. Implement Phase 1 (KV cache)
3. Keep all models in GPU memory

**Result:** 53s → 5-10s immediately with minimal code changes

**Cost-Benefit:**
- Engineering time saved: 4-6 weeks
- Hardware cost: $400-600
- ROI: High if timeline is critical

---

## Conclusion

### Recommended Approach

**Start with Phase 1 (Quick Wins) immediately:**
1. Enable KV cache (5 minutes, 50-80% improvement)
2. Test Flash Attention 2 (30 minutes, 10-20% more)
3. Evaluate Qwen2.5-1.5B (30 minutes, 2x speedup)

**Expected result:** 53s → 10-15s in one day

**Then decide:**
- **If 10-15s is acceptable:** Stop here, enjoy the gains
- **If need better:** Continue to Phase 2 (caching)
- **If need best UX:** Continue to Phase 3 (streaming)

### Critical Success Factors

1. **KV Cache is non-negotiable** - Do this first
2. **Measure everything** - Benchmark after each change
3. **Quality gates** - Don't sacrifice too much quality for speed
4. **Streaming changes everything** - Consider perceived vs actual latency

### Questions to Answer Before Starting

1. What is your acceptable latency target? (10s? 5s? 2s?)
2. Is streaming acceptable for your use case?
3. Can you accept a smaller LLM if quality is still good?
4. Is hardware upgrade on the table?
5. How many concurrent users do you need to support?

---

## Related Documents

- `AGENTS.md` - Development guidelines and code patterns
- `VIBEVOICE_FIXES_README.md` - Transformers compatibility fixes
- `configs/baseline.yml` - Current XTTS configuration
- `configs/2_TTS-to-vibevoice.yml` - VibeVoice configuration
- `src/scripts/compare_benchmarks.py` - Benchmark comparison tool

---

## Appendix: Configuration Examples

### Optimized LLM Config (Phase 1)

```yaml
llm:
  model_id: "Qwen/Qwen2.5-1.5B-Instruct"  # Or keep 3B if quality acceptable
  quantization:
    enabled: true
    load_in_4bit: true
    quant_type: "nf4"
    use_double_quant: true
    compute_dtype: "bfloat16"
  use_cache: true  # CRITICAL: Enable KV cache
  max_new_tokens: 75  # Adjust based on your needs
  attn_implementation: "flash_attention_2"  # If available
```

### Model Cache Config (Phase 2)

```python
# src/model_cache.py
MODEL_CACHE_CONFIG = {
    "llm": {
        "device": "cuda",  # Keep in GPU
        "quantization": "4-bit",
    },
    "asr": {
        "device": "cpu",  # Cache in CPU
        "load_on_demand": True,
    },
    "tts": {
        "device": "cpu",  # Cache in CPU
        "load_on_demand": True,
    }
}
```

---

*Document Version: 2.0*
*Last Updated: January 2026*
*Status: Ready for Implementation*

**Next Step:** Start with Task 1.1 (Enable KV Cache) and measure the impact!
