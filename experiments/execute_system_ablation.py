"""
Execute system ablation experiment with configurable parameters.

Usage:
    # Dry run (100 samples each)
    python experiments/execute_system_ablation.py --dry-run

    # Full experiment
    python experiments/execute_system_ablation.py

    # Custom limits
    python experiments/execute_system_ablation.py --dpi-limit 500 --ipi-limit 500
"""
import argparse
import sys
import traceback
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_system_ablation import run


def execute_experiment(args):
    """Execute the system ablation experiment."""
    print("=" * 80)
    if args.dry_run:
        print("SYSTEM ABLATION EXPERIMENT - DRY RUN")
    else:
        print("SYSTEM ABLATION EXPERIMENT - FULL RUN")
    print("=" * 80)
    print()
    print(f"DPI limit: {args.dpi_limit if args.dpi_limit else 'None (full dataset)'}")
    print(f"IPI limit: {args.ipi_limit if args.ipi_limit else 'None (full dataset)'}")
    print(f"Batch size: {args.batch_size}")
    print(f"Progress interval: {args.progress_interval}")
    print()

    try:
        print("Starting experiment...")
        print()
        result = run(args)

        print()
        print("=" * 80)
        print("EXPERIMENT COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print()
        print("Output files generated:")
        print("-" * 80)
        for key, path in result["outputs"].items():
            print(f"{key:15s}: {path}")
        print()
        print("Sample counts:")
        print("-" * 80)
        for key, value in result["sample_counts"].items():
            print(f"{key:20s}: {value}")
        print()

        return result

    except Exception as e:
        print()
        print("=" * 80)
        print("ERROR OCCURRED")
        print("=" * 80)
        print(f"Error: {e}")
        traceback.print_exc()
        sys.exit(1)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Execute system ablation experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run with 100 samples each
  python experiments/execute_system_ablation.py --dry-run

  # Full experiment with all samples
  python experiments/execute_system_ablation.py

  # Custom sample limits
  python experiments/execute_system_ablation.py --dpi-limit 500 --ipi-limit 500
        """
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run with limited samples (100 DPI + 100 IPI) for testing"
    )
    parser.add_argument(
        "--dpi-limit",
        type=int,
        default=None,
        help="Limit number of DPI samples (default: None = all samples)"
    )
    parser.add_argument(
        "--ipi-limit",
        type=int,
        default=None,
        help="Limit number of IPI samples (default: None = all samples)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for DPI predictions (default: 32)"
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=None,
        help="Print progress every N samples (default: 1000 for full, 100 for dry-run)"
    )

    args = parser.parse_args()

    # Apply dry-run defaults
    if args.dry_run:
        if args.dpi_limit is None:
            args.dpi_limit = 100
        if args.ipi_limit is None:
            args.ipi_limit = 100
        if args.progress_interval is None:
            args.progress_interval = 100
    else:
        if args.progress_interval is None:
            args.progress_interval = 1000

    return args


if __name__ == "__main__":
    args = parse_args()
    execute_experiment(args)
