# Notes about concurrency runner

# For HTTP mode: target is an endpoint accepting POST with raw audio bytes (your FastAPI endpoint). The concurrency runner posts samples in parallel then collects latencies.

# For in-process mode: you can call run_concurrency_test from a Python REPL or a script with a target being a Python callable that accepts a sample and returns the processing result. This is more efficient but requires you to run Python in-process (no Docker isolation).

# The concurrency runner writes per-request JSONL results and a small summary JSON.


#!/usr/bin/env python3
import argparse
from pathlib import Path
import base64
import wave

from src.concurrency.concurrency_runner import run_concurrency_test
from src.prepare_data import stream_dataset_samples
from src.sts_pipeline import process_sample

def load_audio_bytes_from_wav(path: Path) -> bytes:
    # read entire file bytes (for HTTP payload)
    return path.read_bytes()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["http", "inproc"], required=True)
    parser.add_argument("--target", required=True, help="URL for http mode or 'inproc' for inproc mode")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--requests_per_worker", type=int, default=5)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--dataset_split", type=str, default="test")
    args = parser.parse_args()

    # prepare samples (use stream_dataset_samples to get audio samples)
    samples = []
    for i, s in enumerate(stream_dataset_samples(num_samples=args.num_samples, split=args.dataset_split)):
        # for http mode we need raw wav bytes; write a temp wav to memory
        # simplify: save to temp file and read bytes
        tmp_wav = Path(f"/tmp/_bench_sample_{i}.wav")
        torchaudio.save(tmp_wav, s.audio_tensor, s.sampling_rate)
        b = tmp_wav.read_bytes()
        samples.append(b)
        tmp_wav.unlink(missing_ok=True)

    if args.mode == "http":
        results_file, summary = run_concurrency_test(
            mode="http",
            target=args.target,
            samples=samples,
            concurrency=args.concurrency,
            requests_per_worker=args.requests_per_worker,
            output_dir="concurrency_results"
        )
    else:
        # inproc mode target should be a callable; we'll use process_sample wrapper
        # create a small wrapper to call process_sample with minimal args
        def inproc_fn(sample_bytes):
            # we must reconstruct AudioSample from bytes; for now we simply call process_sample with next dataset
            # Instead, for inproc mode, user should pass 'inproc' target and we will use stream_dataset_samples again
            raise NotImplementedError("Inproc mode requires a callable. Use custom script to call process_sample directly.")

        print("Note: inproc mode not implemented in CLI wrapper. Use the concurrency_runner.run_concurrency_test() from Python directly with a callable.")

if __name__ == "__main__":
    main()
