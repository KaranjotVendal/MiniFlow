from src.debug_scripts.common import build_collector
from src.prepare_data import stream_dataset_samples
from src.sts_pipeline import process_sample


def main() -> None:
    config, collector, device = build_collector("configs/2_TTS-to-vibevoice.yml")
    sample = next(stream_dataset_samples(num_samples=1, split="test"))

    collector.start_trial(
        trial_id="debug_process_sample_1",
        sample_id="debug_process_sample_1",
        is_warmup=False,
    )

    try:
        processed = process_sample(
            config=config,
            sample=sample,
            run_id="debug_process_sample_1",
            collector=collector,
            device=device,
            folder="./",
            history=None,
            stream_audio=False,
        )
        trial_metrics = collector.end_trial(status="success")
    except Exception:
        trial_metrics = collector.end_trial(status="error", error="PipelineDebugFailure")
        raise

    print("Pipeline debug run: success")
    print(f'ASR transcript: "{processed.asr_transcript}"')
    print(f'LLM response: "{processed.llm_response}"')
    print(f"TTS sample rate: {processed.tts_waveform_output_sr}")
    print(f"Trial status: {trial_metrics.get('status')}")


if __name__ == "__main__":
    main()
