"""Regenerate system ablation PNGs from existing CSV with larger fonts."""
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_system_ablation import plot_black_white, plot_timing_black_white

SYSTEM_DIR = PROJECT_ROOT / "outputs" / "results" / "system"
TIMESTAMP = "20260517_000508"


def load_rows(path: Path, numeric_fields):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    for row in rows:
        for field in numeric_fields:
            if field in row and row[field] != "":
                row[field] = float(row[field])
    return rows


def main() -> None:
    metrics_csv = SYSTEM_DIR / f"system_ablation_{TIMESTAMP}.csv"
    timing_csv = SYSTEM_DIR / f"system_ablation_timing_{TIMESTAMP}.csv"

    metric_rows = load_rows(
        metrics_csv,
        ["accuracy", "precision", "recall", "f1", "direct_recall", "indirect_recall"],
    )
    timing_rows = load_rows(
        timing_csv, ["mean_ms", "median_ms", "p95_ms", "min_ms", "max_ms"]
    )

    metrics_png = SYSTEM_DIR / f"system_ablation_compare_{TIMESTAMP}.png"
    timing_png = SYSTEM_DIR / f"system_ablation_timing_compare_{TIMESTAMP}.png"

    plot_black_white(metric_rows, metrics_png)
    print(f"Wrote {metrics_png.relative_to(PROJECT_ROOT)}")

    plot_timing_black_white(timing_rows, timing_png)
    print(f"Wrote {timing_png.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
