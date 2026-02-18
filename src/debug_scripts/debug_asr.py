import time

from src.debug_scripts.common import build_collector
from src.prepare_data import stream_dataset_samples
from src.stt.stt_pipeline import run_asr


def main() -> None:
    config, collector, device = build_collector("configs/baseline.yml")
    asr_config = config["asr"]

    sample = next(stream_dataset_samples(num_samples=1, split="test"))

    collector.start_trial(trial_id="debug_asr_1", sample_id="debug_asr_1", is_warmup=False)
    wall_start = time.perf_counter()

    try:
        transcription = run_asr(
            config=asr_config,
            audio_tensor=sample.audio_tensor,
            sampling_rate=sample.sampling_rate,
            groundtruth=sample.transcript,
            collector=collector,
            device=device,
        )
        trial_metrics = collector.end_trial(status="success")
    except Exception:
        trial_metrics = collector.end_trial(status="error", error="ASRDebugFailure")
        raise

    wall_seconds = time.perf_counter() - wall_start

    print("ASR debug run: success")
    print(f'Transcription: "{transcription}"')
    print(f"Trial wall time: {wall_seconds:.3f}s")
    print(f"Trial status: {trial_metrics.get('status')}")


if __name__ == "__main__":
    main()
