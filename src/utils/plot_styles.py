from pathlib import Path


BAR_HATCHES = ("", "++", "///", "\\\\", "xx", "..", "--")
BAR_FACE_COLORS = ("white", "#d9d9d9", "#bdbdbd", "#969696", "#737373", "#525252", "#252525")
LINE_STYLES = ("-", "--", "-.", ":")
LINE_MARKERS = ("o", "s", "^", "D", "v", "P", "X")


def bw_output_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}_bw{path.suffix}")


def bar_style(index: int) -> dict:
    return {
        "facecolor": BAR_FACE_COLORS[index % len(BAR_FACE_COLORS)],
        "edgecolor": "black",
        "linewidth": 1.0,
        "hatch": BAR_HATCHES[index % len(BAR_HATCHES)],
    }


def line_style(index: int) -> dict:
    return {
        "color": "black",
        "linestyle": LINE_STYLES[index % len(LINE_STYLES)],
        "marker": LINE_MARKERS[index % len(LINE_MARKERS)],
        "linewidth": 1.8,
        "markersize": 4.5,
    }


def apply_bw_axis_style(ax) -> None:
    ax.set_facecolor("white")
    ax.grid(axis="y", linestyle=":", color="0.75", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.0)


def save_bw_figure(fig, output_path: Path, dpi: int = 200) -> Path:
    target = bw_output_path(output_path)
    fig.tight_layout()
    fig.savefig(target, dpi=dpi, bbox_inches="tight")
    return target
