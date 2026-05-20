"""Convert blue scatter markers in pr_curve_*.png to black, save as *_bw.png.

The original `plot_pr` in run_ipi_experiment.py draws PR scatter points with
matplotlib's default C0 blue (31, 119, 180); the underlying Optuna study is
not persisted, so we cannot redraw from raw data. Instead, we remap any
blue-dominant pixel to black at the image level, leaving axes/text intact.
"""
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IPI_DIR = PROJECT_ROOT / "outputs" / "results" / "ipi"


def is_blue(rgb: np.ndarray) -> np.ndarray:
    r = rgb[..., 0].astype(np.int16)
    g = rgb[..., 1].astype(np.int16)
    b = rgb[..., 2].astype(np.int16)
    # blue-dominant: B clearly larger than R and G
    return (b - r >= 20) & (b - g >= 10) & (b >= 100)


def convert(src: Path) -> Path:
    img = Image.open(src).convert("RGBA")
    arr = np.array(img)
    rgb = arr[..., :3]
    alpha = arr[..., 3]

    mask = is_blue(rgb)
    if not mask.any():
        print(f"  (no blue pixels found in {src.name})")

    # Map blue markers to black; preserve perceived darkness via luminance,
    # so anti-aliased edges stay smooth instead of turning into hard blobs.
    lum = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.uint8)
    gray = np.stack([lum, lum, lum], axis=-1)
    arr_out = arr.copy()
    arr_out[mask, :3] = gray[mask]

    dst = src.with_name(src.stem + "_bw.png")
    Image.fromarray(arr_out, mode="RGBA").save(dst)
    return dst


def main() -> None:
    targets = sorted(IPI_DIR.glob("pr_curve_*.png"))
    targets = [p for p in targets if not p.stem.endswith("_bw")]

    if not targets:
        print("No pr_curve_*.png to convert.")
        return

    for src in targets:
        dst = convert(src)
        print(f"Wrote {dst.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
