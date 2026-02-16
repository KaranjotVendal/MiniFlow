#!/usr/bin/env python3
from src.config.load_config import load_yaml_config
import argparse
from pathlib import Path

from src.benchmark.runner.sweep_runner import run_sweep
from src.logger.logging import initialise_logger

logger = initialise_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run a sweep of experiment configs.")
    parser.add_argument(
        "--sweep",
        required=True,
        help="Path to YAML sweep definition file.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print config and exit")
    args = parser.parse_args()

    sweep_path = Path(args.sweep)
    if not sweep_path.exists():
        raise FileNotFoundError(f"Sweep file not found: {sweep_path}")

    if args.dry_run:
        config = load_yaml_config(args.sweep)
        logger.info("Dry run enabled. Config loaded successfully:")
        logger.info(f"  Sweep name: {config['sweep_name']}")
        logger.info(f"  Experiments: {len(config['sweep'])}")
        for cfg in config["sweep"]:
            logger.info(f"    - {cfg}")
        return

    logger.info(f"Starting sweep: {sweep_path}")
    run_sweep(sweep_path)
    logger.info("\n Sweep summary:")


if __name__ == "__main__":
    main()
