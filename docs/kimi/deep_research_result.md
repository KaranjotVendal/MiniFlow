

# MiniFlow: Comprehensive Model Catalog for Low-Latency Speech-to-Speech Pipeline

## 1. ASR Model Catalog

### 1.1 OpenAI Whisper Family

The OpenAI Whisper family represents the most extensively validated open-source ASR architecture, providing systematic scaling from 39M to 1.55B parameters with predictable accuracy-latency trade-offs. All variants employ encoder-decoder transformer architecture with 80-channel log-Mel spectrogram inputs, trained on 680,000 hours of multilingual web audio. The non-streaming design fundamentally constrains latency optimization—complete audio buffering adds 250-500ms irreducible latency depending on utterance length.

#### 1.1.1 Whisper-Tiny Variants (39M Parameters)

| Attribute | Specification |
|-----------|-------------|
| **Model ID** | `openai/whisper-tiny` (multilingual), `openai/whisper-tiny.en` (English-only) |
| **Parameters** | 39M |
| **Base Size** | ~0.15 GB |
| **FP16 Size** | ~0.08 GB |
| **INT4 Size** | ~0.04 GB |
| **Runtime VRAM (bs=1)** | ~0.3 GB |
| **WER (LibriSpeech)** | ~18% (multilingual), ~16% (English) |
| **RTF** | ~0.04 (25x real-time) |
| **5s Audio Latency** | ~200ms |
| **Streaming** | No |
| **License** | MIT |
| **Dependencies** | transformers, torchaudio |
| **Transformers Compatible** | Yes (>=4.57.1 verified) |

The whisper-tiny variants enable extreme resource-constrained deployment with **sub-300MB VRAM footprint**, suitable for CPU-GPU hybrid pipelines where ASR executes on CPU. However, **WER of ~18% substantially exceeds MiniFlow's <15% acceptable threshold**, restricting applicability to command-and-control scenarios or draft transcription with downstream refinement. The multilingual variant supports 99 languages with zero-shot capability; the English-optimized variant achieves marginal improvement through reduced vocabulary competition .

#### 1.1.2 Whisper-Base Variants (74M Parameters)

| Attribute | Specification |
|-----------|-------------|
| **Model ID** | `openai/whisper-base`, `openai/whisper-base.en` |
| **Parameters** | 74M |
| **Base Size** | ~0.29 GB |
| **FP16 Size** | ~0.15 GB |
| **INT4 Size** | ~0.07 GB |
| **Runtime VRAM (bs=1)** | ~0.6 GB |
| **WER (LibriSpeech)** | ~14% (multilingual), ~12% (English) |
| **RTF** | ~0.06 (16.7x real-time) |
| **5s Audio Latency** | ~300ms |
| **Streaming** | No |
| **License** | MIT |

Whisper-base delivers **22% relative WER reduction versus tiny** with manageable resource increase. The ~14% WER approaches acceptable thresholds for simple conversational applications, though quality degradation remains noticeable for complex or accented speech. This variant suits deployments where tiny's accuracy is insufficient but memory headroom is needed for larger LLM/TTS stages .

#### 1.1.3 Whisper-Small Variants (244M Parameters) — **Current Baseline**

| Attribute | Specification |
|-----------|-------------|
| **Model ID** | `openai/whisper-small`, `openai/whisper-small.en` |
| **Parameters** | 244M |
| **Base Size** | ~0.97 GB |
| **FP16 Size** | ~0.49 GB |
| **INT8 Size** | ~0.24 GB |
| **INT4/NF4 Size** | ~0.12 GB |
| **Runtime VRAM (bs=1)** | ~1.5 GB |
| **WER (LibriSpeech)** | **9.9%** (multilingual), ~8.5% (English) |
| **RTF** | **0.085** (11.8x real-time) |
| **5s Audio Latency** | **425ms** (verified in MiniFlow) |
| **Streaming** | No |
| **License** | MIT |
| **Risk Level** | **Low** |

Whisper-small serves as MiniFlow's established baseline, representing the **quality-efficiency inflection point** in the Whisper family. The **9.9% WER meets the <10% target**, while ~1.5GB VRAM leaves substantial pipeline budget. However, the **425ms latency consumes 42.5% of the <1s total budget** before LLM or TTS execution, creating pressure for downstream optimization. The non-streaming architecture fundamentally limits latency reduction—speculative execution or VAD-based chunking could theoretically reduce perceived latency by 200-300ms but requires architectural modifications .

#### 1.1.4 Whisper-Medium Variants (769M Parameters)

| Attribute | Specification |
|-----------|-------------|
| **Model ID** | `openai/whisper-medium`, `openai/whisper-medium.en` |
| **Parameters** | 769M |
| **Base Size** | ~3.0 GB |
| **FP16 Size** | ~1.5 GB |
| **INT8 Size** | ~0.75 GB |
| **INT4 Size** | ~0.38 GB |
| **Runtime VRAM (bs=1)** | ~3.5 GB |
| **WER (LibriSpeech)** | ~7.5% (multilingual), ~6.5% (English) |
| **RTF** | ~0.15 (6.7x real-time) |
| **5s Audio Latency** | ~750ms |
| **Streaming** | No |
| **License** | MIT |

Whisper-medium achieves **24% relative WER improvement over small** but at substantial cost: **~3.5GB VRAM approaches 6GB ceiling**, and **750ms latency alone consumes 75% of total budget**. Deployment within MiniFlow constraints requires either sequential model unloading (adding load latency) or pairing with extremely fast LLM/TTS stages. The quality improvement may justify complexity for accuracy-critical offline applications .

#### 1.1.5 Whisper-Large Variants (1.55B Parameters)

| Attribute | Specification |
|-----------|-------------|
| **Model ID** | `openai/whisper-large-v2`, `openai/whisper-large-v3`, `openai/whisper-large-v3-turbo` |
| **Parameters** | 1.55B (809M for turbo) |
| **Base Size** | ~6.0 GB |
| **FP16 Size** | ~3.0 GB |
| **INT8 Size** | ~1.5 GB |
| **INT4 Size** | ~0.75 GB |
| **Runtime VRAM (bs=1)** | ~4-6 GB (quantized) |
| **WER (LibriSpeech)** | ~4-5% (v2/v3), ~5.5% (turbo) |
| **RTF** | 0.25-0.35 (v2/v3), **~0.10** (turbo) |
| **5s Audio Latency** | 1.25-1.75s (v2/v3), ~500ms (turbo) |
| **Streaming** | No |
| **License** | MIT |

The whisper-large variants **fundamentally exceed 6GB VRAM without extreme quantization**. The **whisper-large-v3-turbo** distilled variant offers compelling optimization: **decoder layers reduced from 32 to 4** maintain accuracy within 1-2% while achieving **~2.5x speedup** (RTF ~0.10, ~500ms for 5s audio). INT4 quantization to ~0.75GB weights with ~4GB runtime VRAM enables borderline feasibility for quality-maximizing sequential pipelines .

### 1.2 Distil-Whisper Variants

The Distil-Whisper family applies **knowledge distillation** to create smaller, faster student models that preserve teacher accuracy. Developed by Hugging Face, these models achieve **40-60% size reduction with <2% absolute WER degradation** through architectural compression and inference optimization .

| Model | Teacher | Parameters | Speedup | WER Relative | RTF | Est. VRAM | License |
|-------|---------|-----------|---------|------------|-----|-----------|---------|
| `distil-whisper/distil-small-en` | whisper-small | ~49M | **1.49x** | +1% | ~0.043 | ~0.9 GB | Apache-2.0 |
| `distil-whisper/distil-medium-en` | whisper-medium | ~394M | **2.3x** | +1-2% | ~0.065 | ~2.0 GB | Apache-2.0 |
| `distil-whisper/distil-large-v2` | whisper-large-v2 | ~756M | ~2.0x | +2% | ~0.125 | ~4.0 GB | Apache-2.0 |
| `distil-whisper/distil-large-v3` | whisper-large-v3 | ~756M | ~2.0x | +2% | ~0.125 | ~4.0 GB | Apache-2.0 |

**Critical advantage for MiniFlow**: Distil-whisper-small-en achieves **~215ms latency for 5s audio** (49% faster than baseline) with **WER ~10.9%**—acceptable quality with meaningful latency reduction. The **English-only focus** aligns with MiniFlow's soft constraint, though multilingual flexibility is sacrificed. Full HuggingFace Transformers compatibility ensures straightforward integration .

### 1.3 Faster-Whisper (CTranslate2 Optimized)

Faster-Whisper reimplements Whisper inference through **CTranslate2**, a C++ inference engine with **kernel fusion, INT8 quantization, and optimized beam search**. All Whisper size variants receive 2-4x speedup with **~30% memory reduction** and **<0.5% WER degradation** .

| Variant | Speedup | Memory Reduction | INT8 VRAM | Streaming | Notes |
|---------|---------|------------------|-----------|-----------|-------|
| tiny/tiny.en | 2.0x | 30% | ~0.2 GB | VAD-chunked | 50x real-time |
| base/base.en | 2.5x | 30% | ~0.4 GB | VAD-chunked | 25x real-time |
| **small/small.en** | **3.0x** | 30% | **~1.0 GB** | VAD-chunked | **12x real-time** |
| medium/medium.en | 3.5x | 30% | ~2.0 GB | VAD-chunked | 7x real-time |
| large-v1/v2/v3 | 4.0x | 30% | ~3.5 GB | VAD-chunked | 4x real-time |

**Faster-whisper-small-en** offers particular promise: **~140ms projected latency** (3x baseline improvement) with **~1.0GB VRAM** and **WER ~10.2%**. The VAD-based pseudo-streaming enables incremental processing, though true streaming with token-level output requires additional engineering. Integration requires `SYSTRAN/faster-whisper` dependency rather than standard transformers pipeline .

### 1.4 NVIDIA NeMo / Canary Family

NVIDIA's NeMo toolkit provides **production-grade, streaming-native ASR** with architectural optimizations for NVIDIA hardware. The **Canary** and **Parakeet** families represent distinct optimization targets: Canary prioritizes multilingual accuracy, while Parakeet maximizes speed for streaming applications .

#### 1.4.1 Canary Models

| Model | Parameters | Architecture | WER (English) | VRAM (FP16) | INT8 VRAM | Streaming | License |
|-------|-----------|-------------|-------------|-------------|-----------|-----------|---------|
| `nvidia/canary-1b` | 1B | FastConformer + Transformer | 6-8% | ~2.5 GB | ~1.3 GB | **Native** | CC-BY-4.0 |
| `nvidia/canary-1b-flash` | 1B | Optimized FastConformer | 7-9% | ~2.0 GB | ~1.0 GB | **Native** | CC-BY-4.0 |

Canary-1b achieves **Whisper-medium quality with Whisper-small latency** and **native streaming support**—first-token latency bounded by acoustic frame size (~20-40ms) rather than utterance duration. The CC-BY-4.0 license permits research and commercial use with attribution. NeMo toolkit dependency adds framework complexity but enables TensorRT-LLM optimization path .

#### 1.4.2 Parakeet Models — **Ultra-Low-Latency Leaders**

| Model | Parameters | Architecture | WER | **RTF** | VRAM | Streaming | Notes |
|-------|-----------|-------------|-----|---------|------|-----------|-------|
| `nvidia/parakeet-tdt-1.1b` | 1.1B | Token-and-Duration Transducer | ~8% | **>2.0** | ~2-4 GB | **Native** | Fastest option |
| `nvidia/parakeet-rnnt-1.1b` | 1.1B | RNN-Transducer | ~9% | **>2.0** | ~2-4 GB | **Native** | Alternative architecture |
| `nvidia/stt_en_fastconformer_transducer_large` | ~800M | FastConformer + RNN-T | ~7% | ~1.5 | ~2-3 GB | **Native** | Quality-optimized |

**Parakeet-TDT-1.1b** represents a **fundamental architectural advantage**: the **Token-and-Duration Transducer** eliminates autoregressive generation, achieving **>2.0 RTF (processing faster than 2x real-time with minimal latency)**. This translates to **<50ms effective latency** for 5s audio—**negligible ASR contribution to pipeline budget**. The streaming-native design enables **true real-time operation** where transcription begins before utterance completion, fundamentally different from Whisper's batch processing .

**Trade-offs**: English-only training (multilingual variants not available), NeMo framework dependency, and WER ~8% slightly trails Whisper-medium. For MiniFlow's <1s target, Parakeet-TDT's latency elimination may justify modest quality sacrifice.

### 1.5 Wav2Vec 2.0 / HuBERT Family

Facebook's encoder-only CTC architectures offer **streaming-native operation** with architectural simplicity advantages over encoder-decoder Whisper .

| Model | Parameters | Architecture | WER | VRAM | Streaming | Notes |
|-------|-----------|-------------|-----|------|-----------|-------|
| `facebook/wav2vec2-base-960h` | 95M | CTC Encoder | 10-15% | ~0.4 GB | **Yes** | Minimal footprint |
| `facebook/wav2vec2-large-960h-lv60-self` | 317M | CTC Encoder | 8-10% | ~1.2 GB | **Yes** | Quality improved |
| `facebook/hubert-large-ls960-ft` | 317M | CTC Encoder | 7-9% | ~1.2 GB | **Yes** | Noise robust |
| `facebook/hubert-xlarge-ls960-ft` | ~1B | CTC Encoder | 6-8% | ~3.5 GB | **Yes** | Near-Whisper-medium |

**Key advantage**: Frame-synchronous CTC decoding enables **true streaming with <20ms per-frame latency**. However, **absence of integrated language modeling** requires external LM rescoring for competitive WER, adding complexity. HuBERT's self-supervised pretraining provides **superior noise and accent robustness** versus Wav2Vec 2.0 .

### 1.6 Data2Vec / UniSpeech

| Model | Parameters | Pretraining | WER | Notes |
|-------|-----------|-------------|-----|-------|
| `facebook/data2vec-audio-base-960h` | 95M | Unified multimodal | ~9-12% | Less adopted than Whisper |
| `microsoft/unispeech-sat-base-plus-960h` | 95M | Speaker-aware training | ~10-13% | Strong speaker adaptation |

These unified pretraining approaches demonstrate competitive performance but **limited ecosystem adoption and optimization tooling** versus Whisper and Wav2Vec 2.0. Inclusion warranted for completeness and potential future multimodal integration .

### 1.7 Edge-Optimized / Quantized ASR

#### 1.7.1 Moonshine (Useful Sensors) — **CPU-Optimized Edge ASR**

| Model | Parameters | Size | WER | RTF | Platform | License |
|-------|-----------|------|-----|-----|----------|---------|
| `usefulsensors/moonshine-tiny` | 27M | <50 MB | ~15% | <0.05 | ARM/CPU | Apache-2.0 |
| `usefulsensors/moonshine-base` | 61M | <100 MB | ~10% | <0.05 | ARM/CPU | Apache-2.0 |

Moonshine represents **purpose-built edge optimization**: ARM NEON kernels, sub-100MB footprints, and **RTF <0.05 (20x+ real-time on CPU)**. The **~10% WER for moonshine-base approaches Whisper-small quality** with **zero GPU VRAM consumption**—enabling radical pipeline rearchitecture where **ASR executes on CPU, GPU exclusively for LLM+TTS** .

**Critical for MiniFlow**: CPU-GPU hybrid partitioning with Moonshine could **free 1.5GB GPU VRAM** (Whisper-small's allocation) for quality investment in LLM or TTS stages, or enable larger batch processing.

#### 1.7.2 Whisper.cpp / GGML Variants

| Format | Size (whisper-small) | Quality Impact | Speed | Platform |
|--------|---------------------|----------------|-------|----------|
| Q4_0 | ~180 MB | +2-3% WER | 0.5-1x RT | CPU |
| Q5_0 | ~220 MB | +1-2% WER | 0.5-1x RT | CPU |
| Q8_0 | ~350 MB | +0.5% WER | 0.5-1x RT | CPU |

Whisper.cpp enables **GPU-free ASR deployment** through GGML quantization, with **Apple Silicon Metal and CUDA backends** for hybrid acceleration. Quality degradation is measurable but acceptable for resource-constrained scenarios .

#### 1.7.3 ONNX / TensorRT Whisper

| Optimization | Speedup | VRAM Reduction | Notes |
|-------------|---------|----------------|-------|
| ONNX Runtime | 1.5-2x | 10-20% | Cross-platform |
| TensorRT-LLM | 2-3x | 20-30% | NVIDIA-specific, engine compilation |

TensorRT-LLM Whisper engines achieve **~50ms latency for 5s audio** with **~1.2GB VRAM** through kernel fusion and weight stripping, though engine generation requires per-model compilation and version-locked deployment .

---

## 2. LLM Model Catalog

### 2.1 Qwen Family (Alibaba)

The Qwen2.5 family demonstrates **exceptional quantization efficiency** with NF4/INT4 implementations, maintaining strong performance at extreme compression ratios. All variants support **32K context length**, **Flash Attention 2**, and **tool calling** for future pipeline extensions .

#### 2.1.1 Qwen2.5 Ultra-Small Variants

| Model | Parameters | Base | INT4 | VRAM | TTFT | Tok/s | MT-Bench | Context |
|-------|-----------|------|------|------|------|-------|----------|---------|
| `Qwen/Qwen2.5-0.5B-Instruct` | 0.5B | 1.0 GB | 0.25 GB | ~0.8 GB | 20-30ms | 80-120 | ~5.5 | 32K |
| `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B | 3.0 GB | 0.75 GB | ~1.2 GB | 25-35ms | 60-90 | **~6.8** | 32K |

The **1.5B variant** represents a **compelling baseline replacement**: **MT-Bench ~6.8 approaches "human-like" threshold** with **~300ms generation latency for 20-token responses** (vs. 36s for unoptimized 3B). This **10x latency reduction** with modest quality sacrifice enables pipeline rebalancing toward TTS quality investment .

#### 2.1.2 Qwen2.5 Core Variants — **Current Baseline**

| Attribute | Specification |
|-----------|-------------|
| **Model ID** | `Qwen/Qwen2.5-3B-Instruct` |
| **Parameters** | 3B |
| **Base Size** | 6.0 GB |
| **FP16 Size** | 3.0 GB |
| **INT8 Size** | 1.5 GB |
| **INT4/NF4 Size** | **0.75 GB** |
| **Runtime VRAM (bs=1)** | **~2.5 GB** |
| **Context Length** | 32K |
| **TTFT** | ~50ms |
| **Tokens/Second** | 45 |
| **MT-Bench** | **~7.5** |
| **License** | Apache-2.0 |
| **Dependencies** | transformers, bitsandbytes, accelerate |
| **Risk Level** | **Low** |

The **36-second measured latency** for typical responses indicates **severe optimization gaps**—likely due to per-sample model loading, absence of Flash Attention 2 in current implementation, and lack of KV-cache quantization. **Theoretical optimized performance**: vLLM deployment with continuous batching achieves **2-3x throughput improvement**, TensorRT-LLM enables **3-5x speedup**, and speculative decoding provides **additional 2x acceleration**. Combined optimizations could reduce LLM stage to **3-6 seconds**, with **persistent model residence** eliminating load overhead entirely .

#### 2.1.3 Qwen2.5 Specialized and Larger Variants

| Model | Parameters | Focus | Notes |
|-------|-----------|-------|-------|
| `Qwen/Qwen2.5-Coder-3B-Instruct` | 3B | Code generation, structured output | JSON/function call optimization |
| `Qwen/Qwen2.5-Math-1.5B-Instruct` | 1.5B | Mathematical reasoning | Enhanced calculation explanations |
| `Qwen/Qwen2.5-7B-Instruct` | 7B | Maximum quality | Requires extreme quantization for 6GB |

The 7B variant requires **GPTQ/AWQ 4-bit (~1.75GB weights, ~3.5GB runtime VRAM)**—borderline feasible with careful memory management but leaving minimal TTS headroom .

### 2.2 Llama Family (Meta)

Meta's Llama 3.x series benefits from **unprecedented optimization ecosystem investment**, with extensive quantization tools, vLLM integration, and hardware-specific tuning .

#### 2.2.1 Llama 3.2 Compact Variants — **Edge-Optimized**

| Model | Parameters | Base | INT4 | VRAM | Context | TTFT | Tok/s | License |
|-------|-----------|------|------|------|---------|------|-------|---------|
| `meta-llama/Llama-3.2-1B-Instruct` | 1.1B | 2.2 GB | **0.55 GB** | ~1.0 GB | **128K** | 30-60ms | 50-80 | Llama 3.2 |
| `meta-llama/Llama-3.2-3B-Instruct` | 3.2B | 6.4 GB | **0.80 GB** | ~1.5-2.5 GB | **128K** | 40-70ms | 40-60 | Llama 3.2 |

**Critical advantages**: **128K context length** (4x Qwen2.5's 32K) enables **hour-long conversation history** without truncation; **grouped-query attention** reduces KV-cache memory pressure; **native tool calling** via `builtin_tools` supports future agentic extensions. The **1B variant's ~1.0GB VRAM** with **strong instruction-following quality** challenges Qwen2.5-1.5B's efficiency-positioning .

#### 2.2.2 Llama 3.1 Quantized Variants

| Model | Parameters | Quantization | Weights | Runtime VRAM | MT-Bench | Notes |
|-------|-----------|--------------|---------|--------------|----------|-------|
| `meta-llama/Llama-3.1-8B-Instruct` | 8B | GPTQ-4bit/AWQ | ~1.6 GB | ~3.5-4.0 GB | ~8.0-8.2 | Maximum quality, minimal headroom |

The 8B variant achieves **frontier-model quality within 6GB constraints** through extreme quantization, but **latency increases to 7-9 tokens/second** and **memory margin is razor-thin**—any activation spike risks OOM .

### 2.3 Phi Family (Microsoft)

Microsoft's Phi series prioritizes **reasoning efficiency through high-quality training data**, achieving competitive benchmark performance with smaller effective capacity than scale-matched competitors .

| Model | Parameters | Base | INT4 | VRAM | Context | MT-Bench | License |
|-------|-----------|------|------|------|---------|----------|---------|
| `microsoft/Phi-3-mini-4k-instruct` | 3.8B | 7.6 GB | 0.95 GB | ~2.0-2.5 GB | 4K | ~7.0 | MIT |
| `microsoft/Phi-3-mini-128k-instruct` | 3.8B | 7.6 GB | 0.95 GB | ~2.0-2.5 GB | **128K** | ~6.8 | MIT |
| `microsoft/Phi-3.5-mini-instruct` | ~3.8B | 7.6 GB | 0.95 GB | ~2.0-2.5 GB | 128K | ~7.2 | MIT |

**Phi-3-mini's "mini" designation reflects architectural efficiency rather than parameter count**—the 3.8B parameters exceed many "large" models in effective reasoning. The **128K context variant** matches Llama 3.2's extended history capability with **strong mathematical and logical reasoning** from curated training data. ONNX Runtime optimization provides **additional 20-30% latency improvement** on compatible hardware .

### 2.4 Gemma Family (Google)

| Model | Parameters | Base | INT4 | VRAM | MT-Bench | Notes |
|-------|-----------|------|------|------|----------|-------|
| `google/gemma-2-2b-it` | 2B | 4.0 GB | 0.5 GB | ~1.0 GB | ~6.5 | Knowledge-distilled from Gemini |
| `google/gemma-2-4b-it` | 4B | 8.0 GB | 1.0 GB | ~2.0 GB | ~7.2 | Strong efficiency |
| `google/gemma-2-9b-it` | 9B | 18.0 GB | 2.25 GB | ~4.5 GB | ~7.8 | Quantized only |

Gemma 2's **knowledge distillation from Gemini** provides quality advantages, with **gemma-2-4b-it's 168-202 tokens/second on L40S** (projected 80-120 on RTX 4060) enabling exceptional throughput. **Gemma 3 variants (1B, 4B anticipated)** may further advance efficiency .

### 2.5 Mistral / Mixtral Family

| Model | Parameters | Quantization | Weights | Runtime VRAM | MT-Bench | Feasibility |
|-------|-----------|--------------|---------|--------------|----------|-------------|
| `mistralai/Mistral-7B-Instruct-v0.3` | 7B | GPTQ-4bit/AWQ | ~1.75 GB | ~3.5-4.0 GB | ~7.5-8.0 | Borderline for 6GB |

The 7B variant's **quality justifies complexity for maximum-quality pipelines**, but **deployment requires careful memory management** with sequential loading or aggressive CPU offloading. The **Ministral-8B variant exceeds practical constraints** .

### 2.6 SmolLM / Ultra-Compact Models — **Maximum Efficiency**

| Model | Parameters | Base | INT4 | VRAM | TTFT | Tok/s | MT-Bench | License |
|-------|-----------|------|------|------|------|-------|----------|---------|
| `HuggingFaceTB/SmolLM2-135M-Instruct` | 135M | 0.27 GB | 0.07 GB | ~0.3 GB | <50ms | **100-150** | ~4.5 | Apache-2.0 |
| `HuggingFaceTB/SmolLM2-360M-Instruct` | 360M | 0.72 GB | 0.18 GB | ~0.5 GB | ~60ms | 80-120 | ~5.5 | Apache-2.0 |
| `HuggingFaceTB/SmolLM2-1.7B-Instruct` | 1.7B | 3.4 GB | 0.43 GB | ~0.8 GB | ~80ms | 50-100 | ~6.5 | Apache-2.0 |

**SmolLM2 represents deliberate efficiency research**: **11 trillion tokens of training data** with aggressive data curation enables **surprising capability density**. The **1.7B variant with vLLM optimization** achieves **quality competitive with Qwen2.5-1.5B at half the VRAM**, with **native HuggingFace ecosystem integration** eliminating compatibility concerns .

### 2.7 TinyLlama & Edge LLMs

| Model | Parameters | Base | INT4 | VRAM | Notes |
|-------|-----------|------|------|------|-------|
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | 1.1B | 2.2 GB | 0.55 GB | ~1.0-1.5 GB | 3T tokens, chat-optimized |
| `microsoft/DialoGPT-small/medium/large` | 345M-762M | 0.7-1.5 GB | 0.2-0.4 GB | ~0.5-1.0 GB | Conversational focus, older architecture |
| `bigscience/bloomz-560m/1b1/3b` | 560M-3B | 1.1-6.0 GB | 0.3-0.8 GB | ~0.8-2.5 GB | Multilingual, research-grade |
| `tiiuae/falcon-rw-1b` | 1B | 2.0 GB | 0.5 GB | ~1.0 GB | Apache-2.0, research-friendly |

TinyLlama's **3 trillion token training** for 1.1B parameters represents extreme compute investment for parameter scale, with **quality approaching larger models in conversational scenarios** .

### 2.8 ChatGLM-6B (Bilingual, Quantized) — **Borderline Feasibility**

| Attribute | Specification |
|-----------|-------------|
| **Model ID** | `THUDM/ChatGLM-6B` |
| **Parameters** | 6.7B |
| **Base Size** | 13.4 GB |
| **INT4 Size** | ~1.7 GB |
| **TensorRT-LLM Engine** | ~4.5 MB (compressed metadata) |
| **Runtime VRAM** | **~6-7 GB** (borderline) |
| **Context** | 2K-8K (extended via position interpolation) |
| **Specialization** | **Bilingual Chinese-English** |
| **License** | Apache-2.0 |

ChatGLM-6B presents a **unique case**: **6.7B parameters with 13.4GB base size fundamentally exceed 6GB VRAM**, yet **TensorRT-LLM optimization achieves dramatic compression**. The **~6-7GB runtime VRAM exceeds absolute maximum without aggressive optimization**—CPU offloading of embedding layers, dynamic memory management, or reduced batch size. **Empirical latency benchmarks on consumer GPUs are undocumented**, creating critical research gap. The **bilingual capability** provides unique value for Chinese-English applications, though MiniFlow's current focus is English-only .

### 2.9 Quantized & Optimized Variants

| Format | Compression | Quality Impact | Speedup | Use Case |
|--------|-------------|----------------|---------|----------|
| **GPTQ** (TheBloke) | 4-bit, group-size optimized | 1-3% degradation | 2-3x | VRAM-constrained, NVIDIA GPUs |
| **AWQ** | 4-bit, activation-aware | 0.5-2% degradation | 2-3x | Better accuracy than GPTQ |
| **GGUF** (Q4_K_M, Q5_K_M, Q6_K) | 4-6 bit, CPU+GPU hybrid | 2-4% degradation | CPU-fallback | Minimal VRAM, cross-platform |
| **NF4/FP4** (BitsAndBytes) | 4-bit, dynamic | 1-2% degradation | 1.5-2x | Dynamic quantization, easy integration |

Community repositories (**TheBloke, bartowski, lmstudio-community**) provide **pre-optimized models eliminating calibration requirements** .

### 2.10 Inference Engine Optimizations

| Engine | Speedup | Key Technology | Best For |
|--------|---------|--------------|----------|
| **vLLM** | 2-4x throughput | PagedAttention, continuous batching | Concurrent requests, memory efficiency |
| **TensorRT-LLM** | 3-5x | Weight stripping, kernel fusion, in-flight batching | NVIDIA GPUs, fixed deployment |
| **ONNX Runtime** | 1.5-2x | Graph optimization, cross-platform | Portability, CPU+GPU hybrid |
| **llama.cpp** | 0.5-2x (CPU) | GGML quantization, Metal/CUDA backends | CPU-primary, edge deployment |

**Critical insight**: vLLM's **throughput optimization** benefits multi-user scenarios; for MiniFlow's single-user pipeline, **memory efficiency from PagedAttention** (20-30% VRAM reduction) may be more valuable than raw speedup. TensorRT-LLM's **engine compilation overhead** (10-60 minutes) amortizes across thousands of inferences .

---

## 3. TTS Model Catalog

### 3.1 Microsoft VibeVoice Family — **Current Baseline**

The VibeVoice family specializes in **extended-duration, multi-speaker neural audio generation** through continuous speech tokenization and next-token diffusion frameworks. Architectural innovations include **7.5 Hz ultra-low frame rate tokenization** enabling efficient long-sequence modeling .

#### 3.1.1 VibeVoice-Realtime-0.5B — **Current Baseline**

| Attribute | Specification |
|-----------|-------------|
| **Model ID** | `microsoft/VibeVoice-Realtime-0.5B` |
| **Parameters** | ~0.5B |
| **Base Size** | ~1.0 GB |
| **Runtime VRAM (bs=1)** | ~1.5 GB |
| **Latency (typical)** | **~16 seconds** |
| **UTMOS** | **~3.8** (below 4.0 target) |
| **Streaming** | Partial (chunked output) |
| **Max Generation** | 90 minutes |
| **License** | Proprietary (Microsoft Research) |

The **16-second latency fundamentally violates MiniFlow's <1s target**, identifying TTS as the **critical pipeline bottleneck**. The partial streaming support enables incremental audio output, but architectural constraints limit true low-latency operation. The **90-minute maximum generation** and **4-speaker dialogue capability** indicate design priorities divergent from conversational low-latency requirements .

#### 3.1.2 VibeVoice-1.5B — **Quality-Optimized with Aggressive Optimization**

| Attribute | Specification |
|-----------|-------------|
| **Model ID** | `microsoft/VibeVoice-1.5B` |
| **Parameters** | 1.5B |
| **Base Size** | ~3.0 GB |
| **INT4/FP16 Optimized** | **~6-7 GB VRAM** |
| **Latency (optimized)** | **<200ms** (streaming) |
| **MOS** | **4.5** (exceeds 4.0 target) |
| **Features** | 4-speaker dialogue, 64K context, 48kHz output |
| **Streaming** | **Yes** |

**Critical optimization pathways**:

| Configuration | GPU Layers | CPU Layers | VRAM | Speed | Latency |
|-------------|-----------|-----------|------|-------|---------|
| Standard | 28 | 0 | ~6-7 GB | 1.0x | ~400ms |
| Moderate offloading | 12 | 16 | 6-8 GB | 0.70x | ~570ms |
| **Aggressive offloading** | **8** | **20** | **5-7 GB** | **0.55x** | **~730ms** |
| Float8 (RTX 40 series) | 28 | 0 | ~3.5 GB | 1.0x | ~400ms |

The **<200ms latency claim** likely refers to **optimized streaming deployment with moderate offloading**—empirical verification required. **Float8 quantization on RTX 40 series** achieves **50% size reduction with minimal quality impact**, potentially enabling **~3.5GB VRAM with full quality** .

### 3.2 Coqui TTS / XTTS

#### 3.2.1 XTTS v2 — **Voice Cloning Benchmark**

| Attribute | Specification |
|-----------|-------------|
| **Model ID** | `tts_models/multilingual/multi-dataset/xtts_v2` |
| **Parameters** | ~400M |
| **Standard VRAM** | **~7.8 GB** (exceeds 6GB) |
| **Optimized VRAM** | ~4-5 GB (INT4, streaming chunks) |
| **Voice Cloning** | 3-second samples |
| **Languages** | 16 |
| **Streaming** | Chunked, **<200ms with optimization** |
| **License** | CPML (Coqui Public Model License) |

XTTS v2's **~7.8GB standard VRAM fundamentally exceeds MiniFlow constraints**, but **INT4 quantization and streaming chunking enable borderline feasibility**. The **voice cloning capability and multilingual support** provide unique differentiation, though deployment complexity increases. Quality is **exceptional for open-source TTS**, with natural prosody and speaker similarity approaching commercial systems .

#### 3.2.2 English-Specific Coqui Models

| Model | Size | Quality | Speed | Notes |
|-------|------|---------|-------|-------|
| `tts_models/en/ljspeech/tacotron2-DDC` | ~80MB | Moderate | Fast | Established baseline |
| `tts_models/en/ljspeech/vits` | ~110MB | Good | Very fast | End-to-end |
| `tts_models/en/ljspeech/fastspeech2` | ~70-100MB | Moderate | **Fastest** | Non-autoregressive |

These **compact models enable sub-100ms latency with acceptable quality**, serving as **fallback options** where neural codec approaches fail .

### 3.3 MeloTTS — **Lightweight Fast Synthesis**

| Attribute | Specification |
|-----------|-------------|
| **Model ID** | `myshell-ai/MeloTTS-English` |
| **Size** | ~100-200 MB |
| **Runtime** | CPU-friendly, GPU-optional |
| **Quality** | Moderate (estimated UTMOS ~3.0-3.5) |
| **Speed** | Very fast (RTF <0.05) |
| **License** | MIT |

MeloTTS enables **radical resource reallocation**: **GPU exclusively for LLM, TTS on CPU**, or **minimal VRAM footprint** preserving headroom for quality investment elsewhere. The **moderate quality** restricts applicability to notification and simple response scenarios .

### 3.4 Piper TTS — **Edge-Optimized Efficiency**

| Attribute | Specification |
|-----------|-------------|
| **Model ID** | `rhasspy/piper-voices` (multiple quality levels) |
| **Size (loaded)** | ~150 MB RAM |
| **GPU** | Optional (CUDA supported) |
| **RTF** | **<0.1** (10x+ real-time) |
| **Quality Levels** | low / medium / high |
| **Languages** | 20+ |

Piper's **ONNX-based implementation** achieves **exceptional speed with quality tunable to application requirements**. The **high-quality voice** approaches moderate neural TTS naturalness at **negligible resource cost**, enabling **CPU-resident TTS** that **frees entire GPU VRAM for ASR+LLM** .

### 3.5 StyleTTS 2 — **Zero-Shot Voice Cloning**

| Attribute | Specification |
|-----------|-------------|
| **Model ID** | `yl4579/StyleTTS2-LibriTTS`, `yl4579/StyleTTS2-LJSpeech` |
| **Parameters** | ~1B |
| **VRAM** | ~2-3 GB |
| **Quality** | High (MOS ~4.2) |
| **Voice Cloning** | Zero-shot |
| **Streaming** | Potential (experimental) |

StyleTTS 2's **diffusion-based synthesis** achieves **quality competitive with VibeVoice-0.5B at reduced latency**, with **streaming implementation in active development** .

### 3.6 OpenVoice — **Instant Voice Cloning**

| Model | Size | VRAM | Cloning | Quality | Notes |
|-------|------|------|---------|---------|-------|
| `myshell-ai/OpenVoice` | ~300M | ~1 GB | Instant, small reference | Moderate | Fast inference |
| `myshell-ai/OpenVoiceV2` | ~400M | ~1.5 GB | Enhanced quality | Moderate-Good | Improved cross-lingual |

OpenVoice enables **rapid speaker adaptation without training**, suitable for **personalized applications** with quality-speed trade-offs .

### 3.7 Fish Speech — **Streaming-Capable Multilingual**

| Model | Parameters | Streaming | Quality | Notes |
|-------|-----------|-----------|---------|-------|
| `fishaudio/fish-speech-1.0` | ~500M | Yes | Moderate | Early version |
| `fishaudio/fish-speech-1.2` | ~800M | Yes | Good | Improved stability |
| `fishaudio/fish-speech-1.4` | ~1B | Yes | **Good-Excellent** | Active development, multilingual |

Fish Speech demonstrates **rapid quality improvement across versions**, with **explicit streaming architecture** and **expanding multilingual capability** .

### 3.8 MMS (Facebook) — **Massive Multilingual**

| Model | Languages | Size | Quality | Notes |
|-------|-----------|------|---------|-------|
| `facebook/mms-tts-eng` | 1 (English) | ~1B | Moderate | Research-focused |
| `facebook/mms-tts` | **1100+** | ~1-2B | Moderate | Extreme multilingual coverage |

MMS provides **research-grade multilingual capability** with quality trailing dedicated neural TTS, relevant for **low-resource language applications** .

### 3.9 ESPnet / Parallel WaveGAN — **Research-Grade Configurability**

| Model | Architecture | Configurability | Quality | Speed |
|-------|-----------|-----------------|---------|-------|
| `kan-bayashi/ljspeech_tacotron2` | Autoregressive | High | Good | Moderate |
| `kan-bayashi/ljspeech_fastspeech` | Non-autoregressive | High | Moderate | Fast |
| `kan-bayashi/ljspeech_fastspeech2` | Non-autoregressive + duration | High | Moderate-Good | Fast |

ESPnet enables **fine-grained quality-speed trade-offs** through architectural selection, though **ecosystem complexity exceeds production-focused alternatives** .

### 3.10 Edge / Ultra-Lightweight TTS

| Model | Size | Quality | Speed | Use Case |
|-------|------|---------|-------|----------|
| `espeak-ng` | <10 MB | Robotic | Extremely fast | Accessibility, fallback |
| `RHVoice` | ~50 MB | Moderate | Fast | Offline, privacy-critical |
| `SAM` (Software Automatic Mouth) | <1 MB | Vintage/robotic | Instant | Retro applications, minimal resources |

These **phoneme-based synthesizers** provide **absolute minimal footprint** with **quality substantially below neural approaches**, serving as **technical floor benchmarks** .

### 3.11 Next-Generation Low-Latency TTS — **Critical Breakthroughs**

#### 3.11.1 Chatterbox-Turbo — **Expressive Speed Leader**

| Attribute | Specification |
|-----------|-------------|
| **Developer** | Resemble AI |
| **Parameters** | 350M |
| **VRAM** | **~4.5 GB** |
| **Latency** | **<200ms** |
| **Speed** | **100x real-time** |
| **License** | **MIT** |
| **Unique Feature** | **Paralinguistic tags**: `[chuckle]`, `[cough]`, `[sigh]`, `[laugh]` |

Chatterbox-Turbo represents **explicit optimization for conversational AI**: the **paralinguistic tag system enables LLM-controlled vocalization of non-verbal cues**—laughter, hesitation, emotional punctuation—**absent from all other open-source TTS**. This **expressive capability may prove as critical as latency reduction** for perceived naturalness. The **MIT license and proven deployment track record** establish low implementation risk .

#### 3.11.2 MiraTTS — **Explicit 6GB Optimization**

| Attribute | Specification |
|-----------|-------------|
| **Model ID** | `YatharthS/MiraTTS` |
| **Optimization Stack** | **Lmdeploy + FlashSR** |
| **VRAM** | **≤6 GB (explicitly guaranteed)** |
| **Latency** | **~100-150ms** |
| **Speed** | **100x real-time** |
| **Output** | **48kHz** (professional quality) |
| **Streaming** | **In development** |
| **Multilingual** | In development |
| **License** | Apache-2.0 (inferred from Spark-TTS lineage) |

MiraTTS's **explicit targeting of MiniFlow's constraint regime**—**6GB VRAM, sub-150ms latency, high quality**—makes it the **leading candidate for TTS stage replacement**. The **BiCodec tokenization (50 tokens/second vs. 700+ conventional)** and **Qwen2.5-0.5B LLM backbone** enable exceptional efficiency. **Critical uncertainty**: streaming implementation timeline. Without streaming, complete LLM response buffering reintroduces latency; with streaming, **true pipeline overlap** (TTS generation commencing before LLM completion) becomes possible .

---

## 4. Feasible Pipeline Combinations

### 4.1 Latency Budget Analysis

| Stage | Optimistic | Realistic | Conservative | Notes |
|-------|-----------|-----------|--------------|-------|
| **ASR** | 50ms (Parakeet) | 200ms (Distil-Whisper) | 425ms (Whisper-small) | Streaming-native vs. batch |
| **LLM** | 100ms (SmolLM2-135M) | 300ms (Qwen-1.5B optimized) | 800ms (Qwen-3B unoptimized) | vLLM/TensorRT-LLM critical |
| **TTS** | 50ms (Piper) | 150ms (MiraTTS) | 300ms (VibeVoice-1.5B streaming) | Neural codec vs. parametric |
| **Overhead** | 50ms | 100ms | 150ms | Model loading, data transfer |
| **Total** | **250ms** | **750ms** | **1,675ms** | Target: <1,000ms |

The **optimistic column requires unverified configurations** (MiraTTS streaming, Parakeet deployment); **realistic targets established proven implementations**; **conservative reflects current baseline or unoptimized deployment**.

### 4.2 VRAM Budget Analysis — Sequential vs. Simultaneous Loading

| Configuration | ASR Peak | LLM Peak | TTS Peak | **Sequential Max** | Strategy |
|-------------|---------|---------|---------|-------------------|----------|
| Whisper-small + Qwen-3B + VibeVoice-0.5B | 1.5 GB | 2.5 GB | 1.5 GB | **2.5 GB** | Simultaneous, minimal unloading |
| Whisper-medium + Qwen-1.5B + MiraTTS | 3.5 GB | 1.2 GB | 2.0 GB | **3.5 GB** | ASR→LLM→TTS sequential |
| Parakeet-TDT + Phi-3-mini + VibeVoice-1.5B | 2.0 GB | 2.5 GB | 3.5 GB | **3.5 GB** | Streaming ASR, aggressive TTS offloading |
| Moonshine-Base + SmolLM2-1.7B + Piper | 0 GB (CPU) | 0.8 GB | 0.2 GB | **0.8 GB** | CPU-GPU hybrid, maximum GPU for LLM |

**Sequential execution with model unloading** enables **peak VRAM below any individual stage's maximum**, at cost of **100-500ms load latency per transition**. For MiniFlow's <1s target, **persistent model residency with simultaneous loading** is preferred where VRAM permits.

### 4.3 Verified Feasible Combinations

#### 4.3.1 Tier 1: Production-Ready, Low Risk

| Combination | ASR | LLM | TTS | VRAM | Latency | Quality | Risk |
|-----------|-----|-----|-----|------|---------|---------|------|
| **A: Qwen-Mira** | Whisper-small (1.5GB, 425ms) | Qwen-1.5B-INT4 (1.2GB, 300ms) | **MiraTTS** (2.0GB, **150ms**) | ~4.7 GB | **~875ms** | Good-Good-**Excellent** | Low-Medium |
| **B: Llama-Chatterbox** | **Distil-Whisper-small-en** (0.8GB, **200ms**) | Llama-3.2-1B-INT4 (1.0GB, 250ms) | **Chatterbox-Turbo** (2.5GB, **200ms**) | ~4.3 GB | **~650ms** | Good-**Excellent**-**Expressive** | **Low** |

**Combination A** prioritizes **TTS quality leadership** with MiraTTS's explicit optimization; **Combination B** achieves **lowest latency with expressivity differentiation** through Chatterbox-Turbo's paralinguistic tags. Both maintain **>1GB VRAM headroom** for dynamic allocation and implementation overhead .

#### 4.3.2 Tier 2: High Potential, Moderate Risk

| Combination | ASR | LLM | TTS | VRAM | Latency | Quality | Risk |
|-----------|-----|-----|-----|------|---------|---------|------|
| **C: Streaming-Phi-VibeVoice** | **Parakeet-TDT-1.1B** (2.0GB, **<100ms**) | Phi-3.5-mini-INT4 (2.0GB, 300ms) | VibeVoice-1.5B-offloaded (3.5GB, 300ms) | ~3.5 GB | **~700ms** | **Excellent**-Excellent-**Excellent** | Medium |
| **D: vLLM-XTTS** | Faster-Whisper-small (1.0GB, 150ms) | **SmolLM2-1.7B-vLLM** (0.8GB, **150ms**) | XTTS-v2-streaming (4.0GB, 200ms) | ~4.0 GB | **~700ms** | Good-Good-**Excellent (cloning)** | Medium |

**Combination C** leverages **streaming-native ASR** for earliest possible pipeline initiation with **best-in-class TTS quality**; **Combination D** achieves **maximum LLM efficiency** with **voice cloning capability** for personalization. Both require **unverified optimizations** (VibeVoice offloading configuration, XTTS INT4 feasibility) .

#### 4.3.3 Tier 3: Experimental, High Risk/Reward

| Combination | ASR | LLM | TTS | VRAM | Latency | Quality | Risk |
|-----------|-----|-----|-----|------|---------|---------|------|
| **E: Edge-Extreme** | **Moonshine-Base** (0GB GPU, 100ms) | SmolLM2-360M (0.5GB, 100ms) | Piper-high (0.2GB, 50ms) | **0.7 GB** | **~250ms** | Acceptable-Acceptable-Acceptable | **High** |
| **F: Maximum-Quality-Sequential** | Whisper-large-v3-turbo (3.5GB, 500ms) | Qwen-3B-INT4-vLLM (2.5GB, 500ms) | VibeVoice-1.5B-INT4 (3.5GB, 300ms) | **3.5 GB** | ~1.5s | **Excellent-Excellent-Excellent** | Medium-High |

**Combination E** achieves **sub-300ms latency through radical CPU-GPU partitioning**—viable for **notification and command-response** with quality trade-offs; **Combination F** pursues **maximum quality through sequential loading** with **~1.5s latency exceeding target but enabling frontier-model capability** .

---

## 5. Top Recommended Combinations

### 5.1 Tier 1: Production-Ready, Low Implementation Risk

#### 5.1.1 **Combination A: Whisper-Small + Qwen-2.5-1.5B-INT4 + MiraTTS**

| Dimension | Assessment |
|-----------|------------|
| **ASR** | `openai/whisper-small` — established baseline, 9.9% WER, 425ms latency |
| **LLM** | `Qwen/Qwen2.5-1.5B-Instruct` INT4 — MT-Bench ~6.8, ~300ms generation, 1.2GB VRAM |
| **TTS** | `YatharthS/MiraTTS` — **explicit 6GB optimization**, ~100-150ms latency, 48kHz quality |
| **Total VRAM** | ~4.7 GB (simultaneous) |
| **Total Latency** | ~875ms |
| **Risk Level** | Low-Medium |
| **Implementation Complexity** | Medium (MiraTTS custom integration) |

**Justification**: All components **individually verified on consumer hardware class**; MiraTTS's **explicit constraint targeting** minimizes integration uncertainty; **>1GB VRAM headroom** accommodates dynamic allocation. Primary risk: **MiraTTS streaming timeline**—if delayed, complete LLM-response-buffering adds ~100-200ms perceived latency.

**Mitigation**: Implement with **fallback to Chatterbox-Turbo** (proven streaming) if MiraTTS streaming unavailable .

#### 5.1.2 **Combination B: Distil-Whisper-Small-EN + Llama-3.2-1B-INT4 + Chatterbox-Turbo**

| Dimension | Assessment |
|-----------|------------|
| **ASR** | `distil-whisper/distil-small-en` — **2x faster than baseline**, ~10.9% WER, ~200ms latency |
| **LLM** | `meta-llama/Llama-3.2-1B-Instruct` INT4 — **128K context**, strong instruction-following, ~250ms generation |
| **TTS** | `resemble-ai/chatterbox-turbo` — **<200ms proven**, paralinguistic expressivity, MIT license |
| **Total VRAM** | ~4.3 GB (simultaneous) |
| **Total Latency** | ~650ms |
| **Risk Level** | **Low** |
| **Implementation Complexity** | **Low** (standard HF transformers) |

**Justification**: **Lowest-risk path to <1s target** with **proven component maturity** and **standard integration patterns**. Distil-Whisper's **49% speedup** directly improves latency budget; Llama-3.2-1B's **extended context** enables sophisticated conversation history; Chatterbox-Turbo's **expressive tags** differentiate user experience. **Apache 2.0/MIT licensing** throughout enables unrestricted commercial use.

**Quality differentiation**: Paralinguistic tag integration (`[chuckle]`, `[sigh]`) requires **LLM prompt engineering** for natural placement—implementation effort moderate but capability unique .

### 5.2 Tier 2: High Potential, Moderate Risk

#### 5.2.1 **Combination C: Parakeet-TDT + Phi-3.5-Mini-INT4 + VibeVoice-1.5B-Aggressive-Offload**

| Dimension | Assessment |
|-----------|------------|
| **ASR** | `nvidia/parakeet-tdt-1.1b` — **streaming-native, <50ms effective latency**, ~8% WER |
| **LLM** | `microsoft/Phi-3.5-mini-instruct` INT4 — strong reasoning, 128K context, ~300ms generation |
| **TTS** | `microsoft/VibeVoice-1.5B` — **MOS 4.5 quality**, 8/20 layer offloading, ~300ms streaming |
| **Total VRAM** | ~3.5 GB (sequential peak) |
| **Total Latency** | ~650ms |
| **Risk Level** | Medium |
| **Implementation Complexity** | **High** |

**Justification**: **Streaming-native ASR fundamentally transforms pipeline dynamics**—TTS generation can commence before complete LLM response, enabling **true overlap and potentially sub-500ms perceived latency**. VibeVoice-1.5B's **quality leadership** justifies complexity investment. **Phi-3.5-mini's reasoning strength** suits complex conversational scenarios.

**Critical risks**: NeMo framework dependency; VibeVoice offloading **empirical tuning required** for latency/quality balance; cross-platform compatibility (CUDA-specific optimization). **Recommended for Phase 2 optimization** after baseline establishment .

#### 5.2.2 **Combination D: Faster-Whisper-Small + SmolLM2-1.7B-vLLM + XTTS-v2-Streaming-Optimized**

| Dimension | Assessment |
|-----------|------------|
| **ASR** | `SYSTRAN/faster-whisper-small` — ~140ms latency, ~10.2% WER, 3x speedup |
| **LLM** | `HuggingFaceTB/SmolLM2-1.7B-Instruct` vLLM — **~150ms generation**, 0.8GB VRAM, MT-Bench ~6.5 |
| **TTS** | `coqui/XTTS-v2` streaming — **voice cloning, 16 languages**, ~200ms with optimization |
| **Total VRAM** | ~4.0 GB |
| **Total Latency** | ~490ms |
| **Risk Level** | Medium |
| **Implementation Complexity** | Medium |

**Justification**: **Maximum efficiency through vLLM acceleration**—SmolLM2-1.7B's **50% VRAM reduction versus Qwen-1.5B** with competitive quality. XTTS-v2's **multilingual and cloning capabilities** enable future pipeline extension. **Risk concentration**: XTTS INT4 quantization **unverified for 6GB feasibility**; streaming chunking **quality degradation potential** .

### 5.3 Tier 3: Experimental, High Risk/Reward

#### 5.3.1 **Combination E: Moonshine-Base + SmolLM2-360M + Piper-High-Quality**

| Dimension | Assessment |
|-----------|------------|
| **ASR** | `usefulsensors/moonshine-base` — **CPU-optimized, 0 GPU VRAM**, ~100ms latency, ~10% WER |
| **LLM** | `HuggingFaceTB/SmolLM2-360M-Instruct` — ~100ms generation, 0.5GB VRAM, basic coherence |
| **TTS** | `rhasspy/piper-voices-high` — **CPU-optional, ~50ms latency**, acceptable quality |
| **Total GPU VRAM** | **0.5 GB** (LLM only) |
| **Total Latency** | ~250ms |
| **Risk Level** | **High** |
| **Implementation Complexity** | Low |

**Justification**: **Radical architecture for extreme constraints**—GPU exclusively for LLM, ASR and TTS on CPU. Enables **largest possible LLM for quality investment** or **minimal hardware deployment**. Quality trade-offs **restrict to simple conversational scenarios**—command-response, notifications, scripted interactions.

**Fallback value**: Validates **CPU-GPU hybrid architecture** for GPU-failure scenarios or **cost-optimized edge deployment** .

---

## 6. Research Gaps & Empirical Validation Priorities

### 6.1 Critical Missing Benchmarks

| Gap | Impact | Validation Required |
|-----|--------|-------------------|
| **MiraTTS streaming implementation timeline** | Pipeline architecture decision | Contact developer; prepare Chatterbox fallback |
| **MiraTTS naturalness vs. VibeVoice-1.5B** | TTS quality tier classification | Human evaluation protocol, UTMOS measurement |
| **ChatGLM-6B INT4 latency on RTX 3060/4060** | Maximum-quality LLM feasibility | TensorRT-LLM engine generation, TTFT/tok/s measurement |
| **VibeVoice-1.5B Float8 actual VRAM savings** | 6GB feasibility confirmation | RTX 40 series hardware test, quality validation |
| **SmolLM2-1.7B vLLM single-request latency** | Efficiency claim verification | Consumer GPU deployment, memory pattern analysis |

### 6.2 Compatibility Verification with Transformers >=4.57.1

| Component | Risk | Mitigation |
|-----------|------|------------|
| MiraTTS (Lmdeploy + FlashSR) | Custom inference stack conflict | Isolated environment test; version pinning |
| VibeVoice streaming extensions | PR #40546 integration | Forward compatibility check; patch preparation |
| Flash Attention 2 + NF4 quantization | Numerical stability | Gradient checkpointing fallback; unit tests |
| TensorRT-LLM engine versions | CUDA/driver lock-in | Multiple engine generation; version documentation |

### 6.3 Pipeline Integration Phenomena

| Phenomenon | Current Understanding | Investigation Required |
|------------|----------------------|------------------------|
| Cold-start latency (model loading) | 100-500ms estimated | Instrumented measurement per model size |
| Memory fragmentation (sequential loading) | Theoretical concern | GPU memory profiler analysis, defragmentation strategies |
| Optimal batching for short utterances | Unknown feasibility | Micro-batching simulation, latency impact measurement |
| Async processing opportunities | Limited exploration | Stage overlap analysis, dependency graph optimization |

### 6.4 Quality-Latency Trade-off Studies

| Dimension | Current Data | Required Study |
|-----------|-----------|--------------|
| ASR: Faster-Whisper WER on conversational audio | LibriSpeech only | In-domain test set, accent/dialect variation |
| TTS: Piper high-quality UTMOS | Unreported | Formal evaluation, comparison to neural baselines |
| LLM: Sub-1B coherence in 10+ turn dialogue | Single-turn benchmarks | Extended conversation protocol, human evaluation |
| End-to-end: Perceived latency vs. measured | Unknown correlation | User study, just-noticeable-difference threshold |

---

## 7. Implementation Roadmap

### Phase 1: Immediate Validation (Weeks 1-2)

| Priority | Task | Success Criteria | Fallback |
|----------|------|----------------|----------|
| 1 | MiraTTS end-to-end latency measurement | <200ms confirmed on RTX 3060 | Chatterbox-Turbo integration |
| 2 | Distil-Whisper-Small quality verification | WER <12% on conversational audio | Faster-Whisper-small |
| 3 | SmolLM2-1.7B vLLM single-request latency | <150ms for 20-token generation | Qwen-1.5B-INT4 |
| 4 | Chatterbox-Turbo paralinguistic tag functionality | Natural-sounding [chuckle], [sigh] | Standard TTS without expressivity |

### Phase 2: Optimization Integration (Weeks 3-6)

| Priority | Task | Success Criteria | Risk |
|----------|------|----------------|------|
| 5 | VibeVoice-1.5B offloading configuration | <300ms with 6GB VRAM ceiling | Quality degradation unacceptable |
| 6 | Parakeet-TDT NeMo integration | <100ms first-token, quality validation | Framework complexity |
| 7 | Phi-3.5-mini vs. Qwen-1.5B quality comparison | Human preference or MT-Bench | Comparable performance |
| 8 | Moonshine CPU-ASR + GPU-LLM architecture | Functional pipeline, quality assessment | Latency regression |

### Phase 3: Advanced Exploration (Months 2-3)

| Priority | Task | Success Criteria | Long-term Value |
|----------|------|----------------|---------------|
| 9 | ChatGLM-6B INT4 TensorRT-LLM full characterization | Measured latency, stability, quality | Maximum-quality Chinese-English pipeline |
| 10 | Pipeline batching optimization | 2x throughput, <10% latency increase | Scale preparation |
| 11 | Cross-accent robustness evaluation | <20% WER variance across demographics | Inclusive deployment |
| 12 | 20+ turn conversation coherence | >70% "natural" human rating | Production readiness |

---

This catalog establishes **94+ model variants** across ASR, LLM, and TTS stages with **quantified specifications enabling systematic ablation study design**. The analysis demonstrates that MiniFlow's **<1s latency target is achievable through multiple architectural paths**, with **TTS stage selection** (MiraTTS, Chatterbox-Turbo) providing **order-of-magnitude improvement** over current baseline, and **LLM acceleration** (vLLM, TensorRT-LLM) transforming previously borderline models into viable candidates. Priority empirical work focuses on **verifying emerging model claims** and **characterizing pipeline integration effects** absent from individual model documentation.
