# Benchmark Framework Implementation - Ticket Breakdown

## 🎯 Milestone 0: Project Setup and Dependencies
**Goal**: Prepare project infrastructure and dependencies before implementation begins

---

### **Task 0.1: Add Required Dependencies to Project**
**Description**: Add numpy, scipy, and pyyaml to project dependencies for benchmark framework support

**Context**:
The benchmark framework requires statistical analysis (numpy, scipy) and YAML configuration loading (pyyaml). These dependencies need to be added to the project's dependency management system (pyproject.toml for uv).

**Implementation Details**:
- Add `numpy` for statistical calculations
- Add `scipy` for statistical tests (t-tests, confidence intervals)
- Add `pyyaml` for YAML config loading
- Optional: Add `nvitop` for GPU power monitoring
- Run `uv sync` to update lock file
- Verify imports work in Python REPL

**Acceptance Criteria**:
- [ ] All dependencies install without errors via `uv sync`
- [ ] `import numpy`, `import scipy.stats`, `import yaml` work in Python
- [ ] `pyproject.toml` updated with new dependencies
- [ ] Dependencies are in correct section (main vs optional)

---

### **Task 0.2: Create Benchmark Directory Structure**
**Description**: Create the `src/benchmark/` directory hierarchy with all required subdirectories

**Context**:
The benchmark framework requires a specific directory structure to organize modular components. This task creates the empty directory structure that will be populated in subsequent milestones.

**Implementation Details**:
Create the following directory structure:
```
src/benchmark/
├── core/
├── metrics/
├── collectors/
├── models/
├── storage/
├── analysis/
├── runner/
└── config/

tests/benchmark/
├── core/
└── metrics/
```

**Acceptance Criteria**:
- [ ] All directories created with `__init__.py` files
- [ ] Directories are importable from Python (`from src.benchmark import core`)
- [ ] Directory structure matches specification in benchmark_implementation.md
- [ ] No code yet - just structure

---

## 🎯 Milestone 1: Core Infrastructure (Foundation)
**Goal**: Implement base classes, registry, and configuration system
**Depends on**: Milestone 0

---

### **Task 1.1: Implement Base Classes (BaseMetric, BaseCollector, MetricContext)**
**Description**: Create abstract base classes that define the contract for all metrics and collectors

**Context**:
All metrics in the framework must implement a consistent interface. BaseMetric defines the `start()` and `end()` methods that every metric must implement. BaseCollector defines how metrics wrap operations. MetricContext carries context through the pipeline.

**Implementation Details**:
Create `src/benchmark/core/base.py` with:
- `Stage` enum (ASR, LLM, TTS, PIPELINE)
- `MetricContext` dataclass (stage, trial_id, config, timestamp, metadata)
- `BaseMetric` abstract class with `start()`, `end()`, `get_value()`, `is_enabled()` methods
- `BaseCollector` abstract class with `collect()` method
- Use `from abc import ABC, abstractmethod`
- Use dataclasses from standard library

**Acceptance Criteria**:
- [ ] `BaseMetric` cannot be instantiated directly (abstract)
- [ ] Subclassing `BaseMetric` and implementing `start()`/`end()` works
- [ ] `MetricContext` can be created and accessed
- [ ] Type hints are correct (run ty if available)
- [ ] Docstrings explain the interface contract

---

### **Task 1.2: Implement Metric Registry System**
**Description**: Create the plugin architecture registry for dynamic metric registration

**Context**:
The registry pattern allows metrics to be registered via decorators and looked up by name at runtime. This enables configuration-driven metric selection without hardcoding.

**Implementation Details**:
Create `src/benchmark/core/registry.py` with:
- `MetricRegistry` class with class-level `_metrics` dict
- `register(name)` decorator method that adds classes to registry
- `get(name)` method to retrieve metric class by name
- `list_metrics()` method to get all registered metric names
- Handle duplicate registrations gracefully (warn or overwrite)

**Acceptance Criteria**:
- [ ] Can register a metric class: `@MetricRegistry.register("test_metric")`
- [ ] Can retrieve registered class: `MetricRegistry.get("test_metric")`
- [ ] `list_metrics()` returns list of registered names
- [ ] Duplicate registration handled gracefully
- [ ] Unit tests pass (test in tests/benchmark/core/test_registry.py)

---

### **Task 1.3: Implement Configuration Loading and Validation**
**Description**: Create YAML config loader and dataclass-based validation system

**Context**:
Experiments are configured via YAML files. The configuration system needs to load YAML, validate structure, and provide typed access to config values. Using dataclasses (not Pydantic) keeps dependencies minimal.

**Implementation Details**:
Create files in `src/benchmark/config/`:
1. `loader.py`: `load_benchmark_config(path)` function using `yaml.safe_load()`
2. `validation.py`: Dataclasses for `MetricConfig`, `MetricsConfig`, `BenchmarkConfig`
3. `defaults.py`: Default configuration values

Config structure:
```yaml
benchmark:
  experiment_name: str
  num_samples: int = 20
  metrics:
    enabled: [str]
    configurations: {metric_name: {parameters}}
```

**Acceptance Criteria**:
- [ ] Can load YAML config file without errors
- [ ] Config validation catches missing required fields
- [ ] Default values applied when not specified
- [ ] Config object has `.to_dict()` method
- [ ] Unit tests for config loading and validation pass

---

### **Task 1.4: Create Abstract Storage Interface**
**Description**: Define the storage abstraction layer with base interface

**Context**:
The storage layer abstracts how benchmark results are persisted. The base interface defines methods for saving trials, summaries, and configs, while allowing different implementations (JSONL, database, cloud).

**Implementation Details**:
Create `src/benchmark/storage/base.py` with:
- `BaseStorage` abstract class
- Abstract methods: `save_trial()`, `save_summary()`, `save_config()`, `load_trials()`
- Constructor takes `output_dir: Path`
- Type hints for all parameters

**Acceptance Criteria**:
- [ ] `BaseStorage` is abstract (cannot instantiate)
- [ ] All abstract methods defined with proper signatures
- [ ] Subclass can be created by implementing all methods
- [ ] Interface documented with docstrings
- [ ] No implementation yet - just interface

---

## 🎯 Milestone 2: Essential Metrics (5 Classes)
**Goal**: Implement all 5 modular metric classes with full functionality
**Depends on**: Milestone 1

---

### **Task 2.1: Implement HardwareMetrics (Unified Basic + Detailed Modes)**
**Description**: Create unified hardware monitoring class with GPU memory, utilization, and optional fragmentation analysis

**Context**:
HardwareMetrics is the most important universal metric. It monitors GPU memory (allocated, reserved, peak), utilization, and optionally tracks fragmentation (waste ratio, inactive blocks, segments). The same class serves both `hardware_basic` and `hardware_detailed` modes via configuration.

**Implementation Details**:
Create `src/benchmark/metrics/hardware.py`:
- Class `HardwareMetrics` inheriting from `BaseMetric`
- Register with both `@MetricRegistry.register("hardware_basic")` and `@MetricRegistry.register("hardware_detailed")`
- Config options: `device`, `track_power`, `track_fragmentation`, `waste_threshold`
- In `start()`: Reset peak memory stats, capture baseline
- In `end()`: Collect metrics based on config flags
- Use `torch.cuda.memory_allocated()`, `memory_reserved()`, `max_memory_allocated()`, `memory_stats()`
- Fragmentation metrics only collected when `track_fragmentation=True`

**Acceptance Criteria**:
- [ ] Class registers successfully under both names
- [ ] Basic mode collects: gpu_memory_allocated_mb, gpu_memory_reserved_mb, gpu_memory_peak_mb, gpu_memory_efficiency
- [ ] Detailed mode additionally collects: fragmentation_waste_ratio, inactive_blocks, segment_count, pool_fraction, is_fragmented
- [ ] Works with and without CUDA (graceful fallback)
- [ ] Unit tests pass for both modes (tests/benchmark/metrics/test_hardware.py)

---

### **Task 2.2: Implement TimingMetrics**
**Description**: Create universal timing metric for stage and total latency tracking

**Context**:
TimingMetrics tracks execution time for configurable pipeline stages (ASR, LLM, TTS) and total pipeline latency. It's reusable for any pipeline, not just MiniFlow.

**Implementation Details**:
Create `src/benchmark/metrics/timing.py`:
- Class `TimingMetrics` with `@MetricRegistry.register("timing")`
- Config: `stages` list (e.g., ["asr", "llm", "tts"])
- Methods: `start()`, `record_stage_start(name)`, `record_stage_end(name)`, `end()`
- Track stage start times in dict, calculate latencies in `end()`
- Return: `total_latency`, `stage_latencies` dict

**Acceptance Criteria**:
- [ ] Can track total pipeline latency
- [ ] Can track individual stage latencies when stages are configured
- [ ] Stage timing methods work correctly (start/end pairs)
- [ ] Returns correct data structure with all timings
- [ ] Unit tests pass

---

### **Task 2.3: Implement ModelLifecycleMetrics**
**Description**: Create metric for tracking model loading times and cache hits/misses

**Context**:
ModelLifecycleMetrics tracks how long models take to load from disk, transfer to GPU, and whether they were cached. This helps identify loading bottlenecks and cache efficiency.

**Implementation Details**:
Create `src/benchmark/metrics/lifecycle.py`:
- Class `ModelLifecycleMetrics` with `@MetricRegistry.register("model_lifecycle")`
- Public methods: `record_load_start(model_name, source)`, `record_load_end(model_name, cached)`
- Track load events with timestamps
- Calculate: total_model_load_time, cache_hits, cache_misses
- Return list of load events for detailed analysis

**Acceptance Criteria**:
- [ ] Can record model load start and end events
- [ ] Correctly calculates total load time
- [ ] Tracks cache hits and misses accurately
- [ ] Returns load events list with all details
- [ ] Unit tests pass

---

### **Task 2.4: Implement TokenMetrics**
**Description**: Create LLM-specific metric for token generation tracking (TTFT, TPS)

**Context**:
TokenMetrics is LLM-specific. It tracks tokens generated, time to first token (TTFT), and tokens per second (TPS). It needs to integrate with streaming token generation.

**Implementation Details**:
Create `src/benchmark/metrics/tokens.py`:
- Class `TokenMetrics` with `@MetricRegistry.register("tokens")`
- Config: `track_ttft` (default True)
- Methods: `start()`, `on_token_generated(token)`, `end()`
- Track: token_count, first_token_time, start_time
- Calculate: tokens_generated, ttft, tokens_per_sec, time_per_token
- Handle case where no tokens generated

**Acceptance Criteria**:
- [ ] Counts tokens generated correctly
- [ ] Calculates TTFT accurately (time to first on_token_generated call)
- [ ] Calculates tokens/sec correctly (excluding TTFT time)
- [ ] Handles streaming token generation scenario
- [ ] Unit tests pass (tests/benchmark/metrics/test_tokens.py)

---

### **Task 2.5: Implement QualityMetrics**
**Description**: Create quality evaluation metric with pluggable evaluator system

**Context**:
QualityMetrics evaluates output quality using domain-specific metrics like WER (Word Error Rate) for ASR and UTMOS for speech quality. The evaluator system is pluggable so custom evaluators can be added.

**Implementation Details**:
Create `src/benchmark/metrics/quality.py`:
- Class `QualityMetrics` with `@MetricRegistry.register("quality")`
- Config: `evaluators` list (e.g., ["wer", "utmos"])
- Evaluator interface: classes with `name` attribute and `evaluate(prediction, reference)` method
- Implement `WEREvaluator` using jiwer library
- Implement placeholder `UTMOSEvaluator`
- Method: `evaluate(prediction, reference)` called during pipeline
- Return: Aggregated scores by evaluator name

**Acceptance Criteria**:
- [ ] Can configure which evaluators to use
- [ ] WER evaluator works correctly (use jiwer library)
- [ ] Evaluators can be called during pipeline execution
- [ ] Returns mean scores for each evaluator type
- [ ] Pluggable design allows adding new evaluators
- [ ] Unit tests pass

---

### **Task 2.6: Implement Context Managers and Decorators**
**Description**: Create convenient context managers and decorators for metric collection

**Context**:
Context managers and decorators provide a convenient API for collecting metrics around code blocks and functions. This makes it easy to instrument existing code.

**Implementation Details**:
Create `src/benchmark/collectors/context_managers.py`:
- `@contextmanager def track_latency(name, metrics_dict)`
- `@contextmanager def track_memory(name, metrics_dict)`

Create `src/benchmark/collectors/decorators.py`:
- `def track_latency_decorator(metric_name)` - decorator factory

**Acceptance Criteria**:
- [ ] Context managers work with `with` statement
- [ ] Decorators work with `@` syntax
- [ ] Metrics are recorded correctly in provided dict
- [ ] Exceptions don't break metric collection
- [ ] Unit tests pass

---

## 🎯 Milestone 3: Integration & Storage (Runner)
**Goal**: Connect metrics with storage and create the experiment runner
**Depends on**: Milestones 1-2

---

### **Task 3.1: Implement JSONL Storage Backend**
**Description**: Create concrete storage implementation using JSONL format

**Context**:
JSONL (JSON Lines) is the current storage format used by MiniFlow. This implementation persists trial metrics, summaries, and configs to disk in a structured format.

**Implementation Details**:
Create `src/benchmark/storage/jsonl_storage.py`:
- Class `JSONLStorage` inheriting from `BaseStorage`
- Implement all abstract methods:
  - `save_trial()`: Append to `raw_logs.jsonl`
  - `save_summary()`: Write to `summary.json`
  - `save_config()`: Write to `config.json`
  - `load_trials()`: Read all trials from `raw_logs.jsonl`
- Handle file paths using `pathlib.Path`
- Create output directory if it doesn't exist

**Acceptance Criteria**:
- [ ] Can save trial metrics to JSONL file
- [ ] Can save summary to JSON file
- [ ] Can save config to JSON file
- [ ] Can load all trials from JSONL file
- [ ] Output directory created automatically
- [ ] Files are valid JSON/JSONL format
- [ ] Unit tests pass

---

### **Task 3.2: Implement ExperimentRunner**
**Description**: Create the main experiment runner that orchestrates metric collection across trials

**Context**:
ExperimentRunner is the main entry point. It loads metrics from registry based on config, runs warmup trials, executes main trials, collects metrics, and generates summaries.

**Implementation Details**:
Create `src/benchmark/runner/experiment_runner.py`:
- Class `ExperimentRunner`
- `from_config()` classmethod to create from config dict
- `__init__()` takes config and metric names list
- `run()` method: Run warmup, then main trials, save results
- `_run_trial()` method: Execute one trial with metric collection
- `_generate_summary()` method: Aggregate metrics using statistics module
- Integrate with storage layer
- Create MetricContext for each trial

Create `src/benchmark/runner/warmup.py`:
- Warmup logic to stabilize GPU before measurements

**Acceptance Criteria**:
- [ ] Can create runner from config
- [ ] Loads correct metrics from registry based on config
- [ ] Runs specified number of warmup trials
- [ ] Runs specified number of main trials
- [ ] Collects metrics from all enabled metric classes
- [ ] Saves results to storage
- [ ] Generates summary statistics
- [ ] End-to-end test passes

---

### **Task 3.3: Create Backwards Compatibility Layer**
**Description**: Implement adapter to maintain existing MiniFlow API while using new framework internally

**Context**:
Existing MiniFlow code uses `run_benchmark()` function and `Metrics` dataclass. The compatibility layer allows existing code to work unchanged while internally using the new modular framework.

**Implementation Details**:
Create `src/benchmark/compat.py`:
- Import existing `Metrics` dataclass and extend if needed
- `run_benchmark(config)` function that:
  - Accepts old config format
  - Migrates to new format internally
  - Uses ExperimentRunner
  - Returns results in old format
- `_migrate_config()` helper to transform old configs
- Maintain field name mappings if different

**Acceptance Criteria**:
- [ ] Existing `run_benchmark()` calls work without modification
- [ ] Existing `Metrics` dataclass usage works
- [ ] Old config format accepted and migrated correctly
- [ ] Output format matches existing structure
- [ ] Integration test with existing sts_pipeline.py passes

---

## 🎯 Milestone 4: Analysis & Reporting
**Goal**: Implement statistical analysis and reporting capabilities
**Depends on**: Milestone 3

---

### **Task 4.1: Implement Statistical Analysis Functions**
**Description**: Create statistical utilities for analyzing benchmark results

**Context**:
Research-grade benchmarks need proper statistical analysis: mean, median, std, percentiles, confidence intervals, t-tests for comparing experiments, and effect sizes.

**Implementation Details**:
Create `src/benchmark/analysis/statistics.py`:
- `calculate_statistics(values)` - mean, median, std, min, max, p95, p99
- `paired_t_test(baseline, optimized)` - paired t-test for A/B comparison
- `cohens_d(x, y)` - effect size calculation
- `confidence_interval(data, confidence=0.95)` - confidence interval
- Use numpy and scipy.stats

**Acceptance Criteria**:
- [ ] All functions work with list of float values
- [ ] Statistical calculations are accurate (compare with known values)
- [ ] Handles edge cases (empty lists, single values)
- [ ] Returns Python native types (not numpy types)
- [ ] Unit tests pass with known statistical examples

---

### **Task 4.2: Implement Experiment Comparison Utilities**
**Description**: Create utilities for comparing two benchmark experiments

**Context**:
Phase-1 experiments compare baseline vs optimized configurations. The comparison utility analyzes statistical significance and generates comparison reports.

**Implementation Details**:
Create `src/benchmark/analysis/comparison.py`:
- `compare_experiments(baseline_path, optimized_path)` function
- Load trials from both experiments
- Compare each metric statistically using statistics.py functions
- Generate comparison dict with:
  - Metric differences
  - P-values from t-tests
  - Effect sizes (Cohen's d)
  - Confidence intervals
  - Statistical significance indicators

**Acceptance Criteria**:
- [ ] Can load two experiments from JSONL files
- [ ] Compares all numeric metrics
- [ ] Calculates p-values correctly
- [ ] Identifies statistically significant differences (p < 0.05)
- [ ] Generates comprehensive comparison report
- [ ] Unit tests pass

---

### **Task 4.3: Implement Report Generation**
**Description**: Create formatted reports (Markdown/HTML) from benchmark results

**Context**:
Reports present benchmark results in a readable format suitable for research papers and documentation. Include tables, statistical summaries, and visualizations.

**Implementation Details**:
Create `src/benchmark/analysis/reporting.py`:
- `generate_markdown_report(results, output_path)` - Markdown format
- `generate_html_report(results, output_path)` - HTML format (optional)
- Include:
  - Experiment metadata
  - Summary statistics table
  - Per-metric detailed stats
  - Comparison results (if provided)
  - Configuration details
- Use tables for readability
- Save to file

**Acceptance Criteria**:
- [ ] Generates valid Markdown report
- [ ] Tables are properly formatted
- [ ] All metrics included in report
- [ ] Report is human-readable and research-appropriate
- [ ] File saved to specified path
- [ ] Sample report reviewed for quality

---

## 🎯 Milestone 5: Testing, Documentation & Validation
**Goal**: Ensure quality through comprehensive testing and documentation
**Depends on**: Milestones 1-4

---

### **Task 5.1: Write Unit Tests for All Metrics**
**Description**: Create comprehensive unit tests for each of the 5 metric classes

**Context**:
Each metric class needs independent unit tests to verify correctness. Tests should cover normal operation, edge cases, and configuration options.

**Implementation Details**:
Create tests in `tests/benchmark/metrics/`:
- `test_hardware.py`: Test both basic and detailed modes, with/without CUDA
- `test_timing.py`: Test stage timing, total latency
- `test_lifecycle.py`: Test load events, cache tracking
- `test_tokens.py`: Test token counting, TTFT calculation
- `test_quality.py`: Test evaluators, aggregation

Use pytest fixtures for common setup. Mock external dependencies (CUDA, models) where appropriate.

**Acceptance Criteria**:
- [ ] Each metric has comprehensive unit tests
- [ ] Tests cover all configuration options
- [ ] Tests cover edge cases (empty data, errors)
- [ ] All tests pass with `pytest tests/benchmark/metrics/`
- [ ] Code coverage >= 90% for metric classes

---

### **Task 5.2: Write Integration Tests**
**Description**: Create end-to-end tests for the complete benchmark framework

**Context**:
Integration tests verify that all components work together correctly: config loading, metric registration, runner execution, storage, and result generation.

**Implementation Details**:
Create `tests/benchmark/test_integration.py`:
- Test full experiment run with mock pipeline
- Test config loading and validation
- Test metric registration and lookup
- Test storage read/write
- Test backwards compatibility layer
- Use temporary directories for test outputs

**Acceptance Criteria**:
- [ ] Full experiment run test passes
- [ ] Config roundtrip test passes (load → save → load)
- [ ] Metric registry integration test passes
- [ ] Storage integration test passes
- [ ] Backwards compatibility test passes
- [ ] All integration tests pass

---

### **Task 5.3: Create API Documentation and Usage Examples**
**Description**: Write comprehensive documentation and examples for using the benchmark framework

**Context**:
Documentation enables other developers to use the framework correctly. Include API reference, usage guides, and working examples.

**Implementation Details**:
Create documentation:
1. **API Reference**: Docstrings for all public classes/methods
2. **Usage Guide**:
   - How to configure experiments
   - How to add custom metrics
   - How to run benchmarks
   - How to analyze results
3. **Examples**:
   - `examples/basic_benchmark.py` - Minimal setup
   - `examples/custom_metric.py` - Creating custom metric
   - `examples/compare_experiments.py` - Comparing results

Update README or create docs/ directory.

**Acceptance Criteria**:
- [ ] All public APIs have docstrings
- [ ] Usage guide is comprehensive and clear
- [ ] Examples run without errors
- [ ] Documentation reviewed for clarity
- [ ] Examples demonstrate key features

---

### **Task 5.4: Run Validation Experiments**
**Description**: Execute validation experiments to verify framework produces correct results

**Context**:
Before using the framework for Phase-1 experiments, we need to validate it produces accurate, consistent results comparable to the old benchmark system.

**Implementation Details**:
1. Run baseline experiment with old system
2. Run equivalent experiment with new framework
3. Compare results:
   - Latency measurements should match within margin of error
   - Memory metrics should be similar
   - Token metrics should be identical
4. Run with all 5 metrics enabled
5. Run with different configurations (basic vs detailed hardware)
6. Document any discrepancies and investigate

**Acceptance Criteria**:
- [ ] Validation experiments executed successfully
- [ ] Results compared between old and new systems
- [ ] Discrepancies documented and explained (if any)
- [ ] Framework produces consistent results across multiple runs
- [ ] Performance overhead of framework is acceptable (< 5%)
- [ ] Validation report created

---

## 📋 Summary

| Milestone | Tasks | Est. Duration | Dependencies |
|-----------|-------|---------------|--------------|
| **0: Project Setup** | 2 tasks | 1 day | None |
| **1: Core Infrastructure** | 4 tasks | 2 days | Milestone 0 |
| **2: Essential Metrics** | 6 tasks | 3 days | Milestone 1 |
| **3: Integration & Storage** | 3 tasks | 2 days | Milestones 1-2 |
| **4: Analysis & Reporting** | 3 tasks | 2 days | Milestone 3 |
| **5: Testing & Docs** | 4 tasks | 2 days | Milestones 1-4 |
| **Total** | **22 tasks** | **12 days** | - |

**Critical Path**: Milestones 0 → 1 → 2 → 3 → 4 (Analysis can parallelize with 5)
**Parallel Work**: Milestone 4 (Analysis) can start after Milestone 3, parallel with Milestone 5

---

## Task Quality Checklist

Each task in this document meets the following criteria:

- ✅ **Atomic**: Single committable unit of work
- ✅ **Validated**: Has explicit acceptance criteria ("Done when: ...")
- ✅ **Clear**: Technical, specific, uses imperative verbs
- ✅ **Demoable**: Each milestone produces runnable/testable increment
- ✅ **Builds on prior**: Dependencies clearly stated

---

**Status Key**:
- [ ] Not started
- [ ] In progress
- [ ] Completed

**Last Updated**: January 2026





Absolutely! Here’s a **clear, detailed, coherent, and highly structured prompt**—crafted so it’s understandable by anyone: a productivity coach, a curious teenager, or even an AI system parsing your request with precision. It captures your challenge, your goals, and your real-world context in a way that invites practical, thoughtful solutions.

---
