import argparse

from src.report.sweep_report_generator import generate_sweep_report

def main():
    parser = argparse.ArgumentParser(description="Generate sweep report")
    parser.add_argument("--sweep_dir", required=True)
    args = parser.parse_args()

    report = generate_sweep_report(args.sweep_dir)
    print(f"sweep report written to: {report}")


if __name__ == "__main__":
    main()
