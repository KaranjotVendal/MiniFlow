1. PROJECT CONTEXT & OVERVIEW
What is MiniFlow?
MiniFlow is a low-latency speech-to-speech (STS) conversational AI pipeline designed to run entirely on consumer-grade hardware with limited resources. The system processes audio input through three sequential stages to enable real-time voice conversations.
Architecture (ASR → LLM → TTS)
The pipeline follows a cascading architecture:
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 1: Automatic Speech Recognition (ASR)                   │
│  • Input: Audio waveform (WAV, 16kHz+)                         │
│  • Output: Text transcription                                  │
│  • Current: Whisper-small (39M parameters)                     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 2: Large Language Model (LLM)                           │
│  • Input: ASR transcription + conversation history             │
│  • Output: Generated conversational response                   │
│  • Current: Qwen2.5-3B-Instruct with 4-bit NF4 quantization   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 3: Text-to-Speech (TTS)                                 │
│  • Input: LLM-generated text                                   │
│  • Output: Synthesized audio waveform                          │
│  • Current: VibeVoice-Realtime-0.5B / XTTS v2                  │
└─────────────────────────────────────────────────────────────────┘
Current Implementation Details
Framework Stack:
- Core: HuggingFace Transformers (>=4.57.1), PyTorch 2.9+
- Quantization: BitsAndBytes (4-bit NF4/FP4), GPTQ, AWQ support
- ASR: Transformers pipeline API with Whisper variants
- TTS: Coqui-TTS, VibeVoice (custom streaming inference)
- Metrics: jiwer (WER), UTMOS (speech quality), custom latency tracking
Model Loading Strategy:
- Models are loaded per-sample and unloaded after use
- GPU memory is explicitly cleared between stages
- VibeVoice uses singleton pattern for model caching
- All models must be HuggingFace-compatible or have Python API wrappers
---
2. PROJECT OBJECTIVE
Primary Goal
Conduct comprehensive ablation experiments to identify optimal model combinations that achieve:
- Real-time latency: <1 second end-to-end for short utterances (<5 seconds audio)
- Maximum quality: Best possible WER (ASR), conversational coherence (LLM), and naturalness (TTS)
- Hardware feasibility: Must run on 6GB GPU VRAM and 16GB system RAM
Success Criteria
| Metric | Target | Acceptable Range |
|--------|--------|------------------|
| End-to-end latency | <1s | <2s |
| ASR WER | <10% | <15% |
| LLM response quality | Human-like | Coherent, contextual |
| TTS naturalness | UTMOS >4.0 | UTMOS >3.5 |
| GPU VRAM | <5GB | <6GB (absolute max) |
| System RAM | <12GB | <16GB |
Current Baseline Performance
| Stage | Model | Latency | VRAM | Quality |
|-------|-------|---------|------|---------|
| ASR | Whisper-small | 425ms | ~1.5GB | WER: ~12% |
| LLM | Qwen2.5-3B (4-bit NF4) | 36s | ~2.5GB | Good |
| TTS | VibeVoice-Realtime-0.5B | 16s | ~1.5GB | UTMOS: ~3.8 |
| Total | | ~53s | ~5.5GB | Moderate |
Key Bottleneck: LLM inference (36s) and TTS synthesis (16s) dominate latency.
---
3. RESEARCH REQUEST
Scope
Compile an extensive, exhaustive catalog of ALL viable models for each pipeline stage (ASR, LLM, TTS) that could potentially be used in ablation studies. This is NOT about recommending the best 2-3 options—this is about building a comprehensive database of candidates.
Research Requirements
MUST INCLUDE:
4. All model sizes/variants (tiny, small, base, medium, large where applicable)
5. All quantization formats (FP32, FP16, INT8, INT4, NF4, GPTQ, AWQ, GGUF)
6. All architectural variants (encoder-only, decoder-only, encoder-decoder)
7. Optimization frameworks (TensorRT, ONNX, OpenVINO, CTranslate2, vLLM)
8. Streaming-capable models vs. non-streaming
9. Multi-lingual vs. English-only variants
10. Commercial vs. open-source (clearly marked)
11. Hardware-specific optimizations (CUDA, ROCm, Apple Silicon, CPU)
MUST PROVIDE FOR EACH MODEL:
- Exact model identifier (HuggingFace path, GitHub repo, or official name)
- Parameter count (exact or approximate)
- Memory requirements:
  - Base model size (GB)
  - FP16 size (GB)
  - INT8 size (GB)
  - INT4/NF4 size (GB)
  - Runtime VRAM with typical batch_size=1 (GB)
- Inference speed benchmarks (if available):
  - Tokens/second (LLM)
  - RTF (real-time factor) (ASR/TTS)
  - Time-to-first-token (LLM)
  - Latency for 5-second audio (ASR)
- Quality metrics (if available):
  - WER on LibriSpeech/other benchmarks (ASR)
  - Perplexity / MT-Bench / MMLU (LLM)
  - UTMOS / MOS scores (TTS)
  - Human evaluation rankings
- Key features:
  - Streaming support (yes/no)
  - Voice cloning capability (TTS)
  - Multi-speaker support
  - Accent robustness
  - Context length (LLM)
- Dependencies & compatibility:
  - Transformers version requirements
  - Special libraries needed
  - License type
  - Known issues or limitations
- Use case fit: Why this model might be suitable (or unsuitable) for real-time STS
---
4. MODEL CATEGORIES TO INVESTIGATE
4.1 AUTOMATIC SPEECH RECOGNITION (ASR)
For example:
OpenAI Whisper Family, Distil-Whisper Variants, Faster-Whisper (CTranslate2 Optimized, NVIDIA NeMo / Canary, and many more which are relevant and are SOTS

---
4.2 LARGE LANGUAGE MODELS (LLM)
Qwen Family (Alibaba), Phi Family (Microsoft), Gemma Family (Google), Mistral / Mixtral Family, SmolLM / Small Models, Other Notable Small LLMs that are SOTA and are related to the project.

---
4.3 TEXT-TO-SPEECH (TTS)
Microsoft VibeVoice, Coqui TTS / XTTS, StyleTTS 2, Edge TTS / Lightweight, Real-time/Streaming TTS, and other SOTA models which relevant to the projects.

---
5. ADDITIONAL RESEARCH AREAS
5.1 Optimization Techniques
For each promising model, research:
Quantization Methods:
- BitsAndBytes (NF4, FP4, INT8)
- AutoGPTQ / GPTQ-for-LLaMa
- AutoAWQ
- GGUF / llama.cpp quantization
- ONNX quantization
- TensorRT optimization
Inference Acceleration:
- Flash Attention 2 / SDPA
- KV-cache quantization (HQQ, QLoRA-style)
- Continuous batching (vLLM)
- Speculative decoding
- Prompt caching
- Model sharding / pipeline parallelism
Memory Optimization:
- Gradient checkpointing (for training, but relevant)
- CPU offloading (accelerate's device_map)
- Disk caching
- Model quantization-aware training
5.2 Pipeline Orchestration Strategies
Research non-model factors that affect latency:
- Streaming architecture: Start TTS as soon as first LLM tokens arrive vs. wait for full response
- Model persistence: Keep models in RAM, load to GPU on-demand vs. keep on GPU
- Batch processing: Group multiple short utterances (if batch_size > 1 feasible)
- CPU-GPU hybrid: Run ASR on CPU to save GPU VRAM for LLM+TTS
- Async processing: Parallelize where possible
---
6. DELIVERABLE SPECIFICATIONS
Required Output Format
Provide results in a structured format (markdown tables, JSON, or CSV) with the following sections:
Section 1: ASR Model Catalog
Table with columns:
- Model ID | Parameters | Base Size | INT4 Size | Est. VRAM | WER (LibriSpeech) | RTF | Streaming | Notes
Section 2: LLM Model Catalog
Table with columns:
- Model ID | Parameters | Base Size | INT4 Size | Est. VRAM | Context | TTFT | Tok/sec | Quality Score | Notes
Section 3: TTS Model Catalog
Table with columns:
- Model ID | Size | VRAM | RTF | UTMOS/MOS | Streaming | Voice Cloning | Notes
Section 4: Feasible Combinations Matrix
Table showing:
- ASR Model | LLM Model | TTS Model | Total VRAM | Est. Latency | Risk Level
Section 5: Top Recommendations
- 10-15 highest-potential combinations with justification
- Risk assessment (experimental, well-tested, deprecated, etc.)
- Implementation complexity (low/medium/high)
Section 6: Research Gaps
- Models lacking sufficient documentation/benchmarks
- Areas needing empirical testing
- Known compatibility issues with transformers>=4.57.1
---
7. CONSTRAINTS & BOUNDARIES
Hard Constraints (Non-negotiable)
- GPU VRAM: Absolute maximum 6GB
- System RAM: Maximum 16GB
- Transformers: Must work with transformers>=4.57.1 (VibeVoice compatibility)
- Local execution: No cloud APIs (OpenAI, Google Cloud, Azure, etc.)
- License: Must permit research and potential commercial use (avoid GPL-3.0 if possible)
Soft Constraints (Preferable but flexible)
- English focus: English ASR, English LLM, English TTS preferred
- Streaming: Strong preference for streaming-capable TTS
- Open source: Apache 2.0, MIT, BSD preferred
- Active maintenance: Models updated within last 12 months preferred
- HuggingFace Hub: Models available on HF Hub preferred for ease of use
Out of Scope (DO NOT INCLUDE)
- Cloud-based APIs (OpenAI Whisper API, Google Speech-to-Text, etc.)
- Models requiring >8GB VRAM (unless extreme quantization makes them viable)
- Proprietary/closed models without inference code
- Training frameworks (we're only doing inference)
- Mobile-only models (iOS/Android CoreML/NNAPI)
---
8. QUALITY ASSURANCE CHECKLIST
Before submitting research results, verify:
- [ ] At least 30+ ASR models documented
- [ ] At least 40+ LLM models documented (including quantized variants)
- [ ] At least 25+ TTS models documented
- [ ] Memory estimates include quantization calculations
- [ ] Latency estimates based on published benchmarks or reasonable extrapolation
- [ ] All model IDs are exact and verifiable (HuggingFace paths, GitHub repos)
- [ ] Licenses are specified for each model
- [ ] Known issues/limitations are documented
- [ ] Combination matrix covers at least 50 different viable combinations
- [ ] Research includes both popular models AND obscure/edge candidates
---
9. EXAMPLE ENTRY (Reference Format)
ASR Example: Whisper-Small
model_id: "openai/whisper-small"
family: "Whisper"
parameters: 244M
base_size_gb: 0.97
fp16_size_gb: 0.49
int8_size_gb: 0.24
int4_size_gb: 0.12
estimated_vram_gb: 1.5
wer_librispeech: 9.9%
rtf_realtime_factor: 0.085  # ~8.5% of audio duration
streaming_support: false
license: "MIT"
transformers_compatible: true
dependencies: ["transformers", "torchaudio"]
notes: |
  Current baseline model. Good balance of accuracy and speed.
  Not streaming-native; requires full audio input.
  English multi-accent performance: Good.
  Compatible with transformers>=4.57.1.
risk_level: "Low"
LLM Example: Qwen2.5-3B-Instruct (4-bit)
model_id: "Qwen/Qwen2.5-3B-Instruct"
family: "Qwen2.5"
parameters: 3B
base_size_gb: 6.0
fp16_size_gb: 3.0
int8_size_gb: 1.5
int4_size_gb: 0.75  # NF4
estimated_vram_gb: 2.5
context_length: 32k
ttf_ms: 50
tokens_per_second: 45
quality_score: "Strong (MT-Bench: ~7.5)"
streaming_support: true
license: "Apache-2.0"
transformers_compatible: true
dependencies: ["transformers", "bitsandbytes", "accelerate"]
notes: |
  Current baseline. Excellent quantization efficiency.
  Fast token generation with Flash Attention 2.
  Strong conversational quality for 3B model.
  Supports tool calling.
  Well-maintained by Alibaba.
risk_level: "Low"
---
10. FINAL INSTRUCTIONS
This research is the foundation for extensive ablation testing. The goal is completeness, not brevity.
Think expansively:
- Include models that seem too large (we can quantize)
- Include models that are experimental (flag them as such)
- Include models from different ecosystems (HuggingFace, GitHub, research papers)
- Include edge cases (tiny models, giant models with extreme quantization)
Do not filter prematurely—the ablation study will determine what actually works. Your job is to ensure we don't miss any viable candidates.
Prioritize accuracy over speculation: If benchmarks aren't available, say so. If you're extrapolating, explain your reasoning.
