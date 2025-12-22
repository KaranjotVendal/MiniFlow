#!/usr/bin/env python3
import argparse
from pathlib import Path
from src.report.html_generator import generate_html_report_from_md


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_dir", required=True, help="Experiment folder with report.md"
    )
    args = parser.parse_args()
    path = Path(args.exp_dir)
    if not path.exists():
        raise FileNotFoundError("exp_dir does not exist")
    out = generate_html_report_from_md(path)
    print(f"HTML report written to: {out}")


if __name__ == "__main__":
    main()
