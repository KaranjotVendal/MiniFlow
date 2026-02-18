from src.debug_scripts.common import build_collector
from src.llm.llm_pipeline import run_llm


def main() -> None:
    config, collector, device = build_collector("configs/baseline.yml")
    llm_config = config["llm"]

    collector.start_trial(trial_id="debug_llm_1", sample_id="debug_llm_1", is_warmup=False)

    try:
        response, history = run_llm(
            config=llm_config,
            transcription="Hello, what is the weather today?",
            device=device,
            collector=collector,
            history=None,
        )
        trial_metrics = collector.end_trial(status="success")
    except Exception:
        trial_metrics = collector.end_trial(status="error", error="LLMDebugFailure")
        raise

    print("LLM debug run: success")
    print(f'Response: "{response}"')
    print(f"History turns: {len(history)}")
    print(f"Trial status: {trial_metrics.get('status')}")


if __name__ == "__main__":
    main()
