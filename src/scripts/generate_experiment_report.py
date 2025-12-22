import argparse

from src.report.report_generator import generate_experiment_report


def main():
    parser = argparse.ArgumentParser(description="Geneate benchmark experiment report")
    parser.add_argument("--exp-dir", required=True, help="experiment folder")
    args = parser.parse_args()

    report = generate_experiment_report(args.exp_dir)
    print(f"Report written to: {report}")


if __name__ == "__main__":
    main()
