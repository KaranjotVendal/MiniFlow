#!/usr/bin/env python3
import argparse
from pathlib import Path

from src.benchmark.sweep_runner import run_sweep
from src.logger.logging import initialise_logger

logger = initialise_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run a sweep of experiment configs.")
    parser.add_argument(
        "--sweep",
        required=True,
        help="Path to YAML sweep definition file.",
    )
    args = parser.parse_args()

    sweep_path = Path(args.sweep)
    if not sweep_path.exists():
        raise FileNotFoundError(f"Sweep file not found: {sweep_path}")

    logger.info(f"Starting sweep: {sweep_path}")
    summary = run_sweep(str(sweep_path))
    logger.info("\n Sweep summary:")
    logger.info(summary)


if __name__ == "__main__":
    main()
