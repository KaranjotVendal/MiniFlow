import argparse
from pathlib import Path

from src.config.inspect_config import inspect_config
from src.benchmark.runner import run_benchmark
from src.logger.logging import initialise_logger

logger = initialise_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run an experiment from yaml config.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--dry-run", action="store_true", help="when enabled only prints config"
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    logger.info(f"starting experiment using config: {config_path}")
    config = inspect_config(config_path)
    if args.dry_run:
        logger.info("Dry run enabled. Config loaded successfully. Exiting.")
        return

    summary = run_benchmark(config)
    logger.info("Experiment complete.")
    logger.info(f"Summary:\n{summary}")


if __name__ == "__main__":
    main()
