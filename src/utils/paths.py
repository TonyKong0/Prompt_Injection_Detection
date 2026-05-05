from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = OUTPUT_DIR / "model"
RESULTS_DIR = OUTPUT_DIR / "results"
LOGS_DIR = OUTPUT_DIR / "logs"

# ===== Common sub-structure =====

DPI_DATA_DIR = DATA_DIR / "dpi"
IPI_DATA_DIR = DATA_DIR / "ipi"

DPI_MODEL_DIR = MODEL_DIR / "dpi_detector"

RESULTS_DPI_DIR = RESULTS_DIR / "dpi"
RESULTS_IPI_DIR = RESULTS_DIR / "ipi"


def ensure_dir(path: Path) -> Path:
    """
    Ensure a directory exists and return the same path.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def dpi_parquet_path(split: str) -> Path:
    """
    split: "train" | "test"
    """
    if split not in ("train", "test"):
        raise ValueError(f"Unsupported split: {split}")
    return DPI_DATA_DIR / f"{split}-00000-of-00001.parquet"


def ipi_synthetic_indirect_dataset_path() -> Path:
    return IPI_DATA_DIR / "synthetic" / "synthetic_indirect_ipi.json"