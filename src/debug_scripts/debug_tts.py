from src.debug_scripts.common import build_collector
from src.tts.tts_pipelines import run_tts


def main() -> None:
    config, collector, device = build_collector("configs/2_TTS-to-vibevoice.yml")
    tts_config = config["tts"]

    text = (
        "The quick brown fox jumps over the lazy dog. "
        "This is a longer sentence to test TTS performance."
    )

    collector.start_trial(trial_id="debug_tts_1", sample_id="debug_tts_1", is_warmup=False)

    try:
        waveform, output_sample_rate = run_tts(
            config=tts_config,
            llm_response=text,
            device=device,
            collector=collector,
        )
        trial_metrics = collector.end_trial(status="success")
    except Exception:
        trial_metrics = collector.end_trial(status="error", error="TTSDebugFailure")
        raise

    print("TTS debug run: success")
    print(f"Output sample rate: {output_sample_rate} Hz")
    print(f"Waveform samples: {len(waveform)}")
    print(f"Trial status: {trial_metrics.get('status')}")


if __name__ == "__main__":
    main()
