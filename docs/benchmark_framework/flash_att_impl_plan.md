# Flash Attention Implementation Plan for MiniFlow

## GPU Specifications

**Current Hardware:**
- GPU: NVIDIA GeForce RTX 3060
- Architecture: Ampere (SM 8.6)
- CUDA Version: 13.1
- VRAM: 6GB
- Driver: 590.48.01

**Compatibility Status:** ✅ **FULLY SUPPORTED**
- Compute Capability: 8.6 (requires 8.0+)
- CUDA Version: 13.1 (requires 11.6+)
- PyTorch: 2.9.0 (requires 2.0+)

---

## Option 1: Flash Attention 2 (Recommended for Maximum Performance)

### Installation

```bash
# Option A: Pre-built wheels (faster, if available for your CUDA version)
pip install flash-attn --no-build-isolation

# Option B: Build from source (takes 10-20 minutes)
pip install flash-attn --no-build-isolation --verbose

# Verify installation
python -c "import flash_attn; print(flash_attn.__version__)"
```

### Configuration

Update your model loading code in `src/sts_pipeline.py`:

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# Add attn_implementation parameter
model = AutoModelForCausalLM.from_pretrained(
    config["model_id"],
    quantization_config=quant_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",  # Enable Flash Attention 2
)
```

Or update your YAML config:
```yaml
llm:
  model_id: "Qwen/Qwen2.5-3B-Instruct"
  attn_implementation: "flash_attention_2"
  quantization:
    enabled: true
    load_in_4bit: true
    quant_type: "nf4"
```

### Expected Benefits

| Metric | Improvement |
|--------|-------------|
| **Speed** | 10-20% faster inference |
| **Memory** | 20-30% reduction in VRAM usage |
| **Sequence Length** | Can handle longer contexts efficiently |

### Prerequisites Check

Run this to verify compatibility before installing:

```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")
```

**Required outputs:**
- CUDA available: True
- Compute Capability: (8, 6) or higher

---

## Option 2: SDPA (Scaled Dot Product Attention) - Quick Win

### Installation

**No installation required!** SDPA is built into PyTorch 2.0+.

### Configuration

```python
model = AutoModelForCausalLM.from_pretrained(
    config["model_id"],
    quantization_config=quant_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",  # Use PyTorch native optimized attention
)
```

### Expected Benefits

| Metric | Improvement |
|--------|-------------|
| **Speed** | 5-10% faster inference |
| **Memory** | 15-20% reduction in VRAM usage |
| **Setup** | Zero additional dependencies |

### When to Use SDPA

✅ **Use SDPA if:**
- You want to test attention optimization quickly
- You don't want to install additional dependencies
- Flash Attention 2 installation fails
- You need maximum compatibility

❌ **Don't use SDPA if:**
- You need maximum performance
- You want memory-efficient attention for long sequences

---

## Option 3: xFormers

### Installation

```bash
pip install xformers
```

### Configuration

xFormers integrates with transformers through SDPA backend:

```python
model = AutoModelForCausalLM.from_pretrained(
    config["model_id"],
    attn_implementation="sdpa",  # xFormers uses SDP as backend
    # ... other config
)
```

### Expected Benefits

| Metric | Improvement |
|--------|-------------|
| **Speed** | Similar to SDPA |
| **Memory** | 15-25% reduction |
| **Features** | Additional optimizations (beyond attention) |

### When to Use xFormers

- You need additional memory optimizations beyond attention
- You're already using xFormers for other components
- You want a middle ground between SDPA and Flash Attention

---

## Comparison Table

| Implementation | Speed | Memory | Setup Time | Reliability | Recommendation |
|----------------|-------|--------|------------|-------------|----------------|
| **Eager (default)** | 1.0x | 1.0x | None | ⭐⭐⭐⭐⭐ | Current baseline |
| **SDPA** | 1.05-1.10x | 0.80-0.85x | 0 min | ⭐⭐⭐⭐⭐ | Quick win, no install |
| **Flash Attention 2** | 1.10-1.20x | 0.70-0.80x | 10-20 min | ⭐⭐⭐⭐ | Maximum performance |
| **xFormers** | 1.05-1.10x | 0.75-0.85x | 2-5 min | ⭐⭐⭐⭐ | Additional features |

---

## Implementation Strategy

### Phase 1: Quick Test with SDPA (5 minutes)

1. **Modify model loading code** (no installation needed):
   ```python
   attn_implementation="sdpa"
   ```

2. **Run benchmark**:
   ```bash
   uv run python src/scripts/run_experiment.py --config configs/2_TTS-to-vibevoice.yml
   ```

3. **Compare results** to baseline

4. **Decision point:**
   - If SDPA gives noticeable improvement → proceed to Flash Attention 2
   - If minimal improvement → focus on other optimizations (KV cache, smaller model)

### Phase 2: Install Flash Attention 2 (15-30 minutes)

If Phase 1 shows promise:

1. **Install Flash Attention 2**:
   ```bash
   pip install flash-attn --no-build-isolation
   ```

2. **Update configuration**:
   ```python
   attn_implementation="flash_attention_2"
   ```

3. **Run benchmark** and compare

4. **Verify memory savings** with `nvidia-smi`

### Phase 3: Integration with Other Optimizations

Combine attention optimization with other strategies:

```python
# Optimized model loading
model = AutoModelForCausalLM.from_pretrained(
    config["model_id"],
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    ),
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",  # Flash Attention
    use_cache=True,  # KV cache
)
```

---

## Additional VRAM-Saving Options for RTX 3060

Since you have 6GB VRAM, combine attention optimization with these:

### 1. Enable KV Cache (CRITICAL)
```yaml
llm:
  use_cache: true  # This alone saves 50-80% time
```

### 2. Reduce max_new_tokens
```yaml
llm:
  max_new_tokens: 75  # Instead of 100
```

### 3. Use BFloat16
Already in your config, but verify:
```yaml
llm:
  quantization:
    compute_dtype: "bfloat16"
```

### 4. Gradient Checkpointing (for inference)
```python
model.gradient_checkpointing_enable()
```

### 5. Attention Slicing (if still having memory issues)
```python
model.enable_attention_slicing(1)
```

### 6. CPU Offloading
```python
from accelerate import Accelerator

accelerator = Accelerator(cpu_offload=True)
model = accelerator.prepare(model)
```

---

## Testing & Validation

### Before Implementation

1. **Baseline benchmark** (without attention optimization):
   ```bash
   uv run python src/scripts/run_experiment.py --config configs/2_TTS-to-vibevoice.yml
   ```

2. **Record metrics**:
   - LLM latency (currently 36s)
   - GPU memory usage (currently ~2.3GB)
   - Total pipeline latency (currently 53s)

### After Implementation

1. **With SDPA**:
   ```bash
   # Modify code to use attn_implementation="sdpa"
   uv run python src/scripts/run_experiment.py --config configs/2_TTS-to-vibevoice.yml
   ```

2. **With Flash Attention 2**:
   ```bash
   # Install and modify code to use attn_implementation="flash_attention_2"
   uv run python src/scripts/run_experiment.py --config configs/2_TTS-to-vibevoice.yml
   ```

3. **Compare results** using:
   ```bash
   uv run python src/scripts/compare_benchmarks.py \
       Benchmark/baseline/[baseline_run]/summary.json \
       Benchmark/baseline/[optimized_run]/summary.json
   ```

### Success Criteria

✅ **SDPA is worth it if:**
- 5%+ speed improvement
- No quality degradation
- No installation issues

✅ **Flash Attention 2 is worth it if:**
- 10%+ speed improvement over SDPA
- 15%+ memory reduction
- Installation succeeds without errors

❌ **Revert to eager if:**
- Numerical instability or quality issues
- Installation problems
- Minimal improvement

---

## Troubleshooting

### Flash Attention Installation Issues

**Problem:** `RuntimeError: CUDA error: invalid device function`

**Solution:** Your GPU might not be supported. Check compute capability:
```python
import torch
print(torch.cuda.get_device_capability(0))  # Should be (8, 0) or higher
```

**Problem:** Installation takes too long / fails

**Solution:** Use pre-built wheels:
```bash
# Try specific CUDA version
pip install flash-attn --no-build-isolation --find-links https://github.com/Dao-AILab/flash-attention/releases
```

**Problem:** Out of memory during installation

**Solution:** Install with limited parallelism:
```bash
MAX_JOBS=4 pip install flash-attn --no-build-isolation
```

### Runtime Issues

**Problem:** `ValueError: FlashAttention only supports Ampere GPUs or newer`

**Solution:** Your RTX 3060 IS supported. Check PyTorch CUDA version:
```python
import torch
print(torch.version.cuda)  # Should be 11.6 or higher
```

**Problem:** Numerical instability (NaN/Inf in outputs)

**Solution:** Fall back to SDPA or eager:
```python
attn_implementation="sdpa"  # or remove the parameter entirely
```

---

## Decision Tree

```
Start
  │
  ▼
Do you want maximum performance?
  │
  ├─ Yes ──► Install Flash Attention 2
  │            │
  │            ▼
  │         Installation successful?
  │            │
  │            ├─ Yes ──► Use Flash Attention 2
  │            │
  │            └─ No ──► Use SDPA
  │
  └─ No ──► Use SDPA (quick win, no install)
```

---

## Implementation Checklist

- [ ] **Phase 1: Baseline Measurement**
  - [ ] Run benchmark without attention optimization
  - [ ] Record LLM latency, memory usage, total latency
  - [ ] Save results for comparison

- [ ] **Phase 2: SDPA Test**
  - [ ] Modify model loading code to use `attn_implementation="sdpa"`
  - [ ] Run benchmark
  - [ ] Compare to baseline
  - [ ] Decision: Continue to Flash Attention?

- [ ] **Phase 3: Flash Attention 2 (if Phase 2 promising)**
  - [ ] Install Flash Attention 2
  - [ ] Modify model loading code to use `attn_implementation="flash_attention_2"`
  - [ ] Run benchmark
  - [ ] Compare to SDPA and baseline
  - [ ] Decision: Keep or revert?

- [ ] **Phase 4: Integration**
  - [ ] Combine with KV cache
  - [ ] Combine with smaller model (Qwen2.5-1.5B)
  - [ ] Run full benchmark suite
  - [ ] Measure combined impact

---

## Expected Timeline

| Task | Time | Dependencies |
|------|------|--------------|
| Baseline benchmark | 10-15 min | None |
| SDPA implementation | 5 min | None |
| SDPA benchmark | 10-15 min | SDPA implementation |
| Flash Attention install | 10-20 min | None |
| Flash Attention implementation | 5 min | Installation |
| Flash Attention benchmark | 10-15 min | Implementation |
| Compare results | 5 min | All benchmarks |
| **Total (if doing both)** | **~1 hour** | - |

---

## Next Steps

1. **Immediate (Today):**
   - Run baseline benchmark to confirm current performance
   - Implement SDPA (5-minute change)
   - Test SDPA performance

2. **If SDPA shows improvement:**
   - Install Flash Attention 2
   - Test Flash Attention 2
   - Compare all three versions

3. **Decision point:**
   - Choose best attention implementation
   - Integrate with KV cache and other optimizations
   - Update `minimax_recommendation.md` with results

---

## References

- **Flash Attention 2 GitHub:** https://github.com/Dao-AILab/flash-attention
- **Transformers Documentation:** https://huggingface.co/docs/transformers/perf_infer_gpu_one
- **PyTorch SDPA:** https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
- **NVIDIA Ampere Architecture:** https://www.nvidia.com/en-us/data-center/ampere-architecture/

---

*Document Version: 1.0*
*Created: January 2026*
*Hardware Target: NVIDIA RTX 3060 (Ampere, 6GB VRAM)*
*Status: Ready for Implementation*

**Next Action:** Start with Phase 1 (baseline measurement) and Phase 2 (SDPA test)!
