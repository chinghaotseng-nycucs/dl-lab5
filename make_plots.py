"""Report figures and numbers from the log.csv of finished runs. Extra, not required.

Spec 4.1 asks for wandb plots of Tasks 1, 2 and 3. This script adds the comparison figures (sample efficiency, ablation)
and the number of steps each run needed, from the same evaluation rows.

Usage:
    python make_plots.py --out figures/task3_vs_vanilla.png --target 19 \
        --run "Enhanced DQN=runs/task3" --run "Vanilla DQN=runs/task2"

Each --run is label=folder. The first run is the baseline for "extra steps".
"""

import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")  # draw to a file: no window is needed on a server
import matplotlib.pyplot as plt

# Line colors in a fixed order, so the same position always gets the same color. Checked for color-blind separation.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"


def read_log(folder):
    """Return (steps, means) from one run's log.csv, sorted by environment step."""
    with open(os.path.join(folder, "log.csv"), newline="") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda row: int(row["env_step"]))  # lambda: a one-line function that picks the sort key
    return [int(row["env_step"]) for row in rows], [float(row["mean"]) for row in rows]


def first_step_at(steps, means, target):
    """The first evaluation at or above the target, or None."""
    for step, mean in zip(steps, means):
        if mean >= target:
            return step
    return None


def convergence_step(steps, means, target):
    """The first step after which every evaluation stays at or above the target, or None (fundamentals page 10, Step 5)."""
    answer = None
    for step, mean in zip(steps, means):
        if mean >= target:
            answer = answer if answer is not None else step
        else:
            answer = None  # it fell back, so the search starts again
    return answer


def step_label(step):
    """A readable step count: 900000 -> '900k', 1000000 -> '1M', 2500000 -> '2.5M'."""
    if step >= 1000000:
        return f"{step / 1000000:g}M"  # :g drops trailing zeros, so 1.0 prints as 1
    return f"{step / 1000:g}k"


def main():
    parser = argparse.ArgumentParser(description="Comparison figure and numbers from log.csv files")
    parser.add_argument("--run", action="append", required=True)  # no default: label=folder, repeat for each run
    parser.add_argument("--out", required=True)  # no default: the PNG path
    parser.add_argument("--target", type=float, default=19.0)  # default 19 (spec 2.3); use 480 for Task 1
    args = parser.parse_args()

    runs = [item.split("=", 1) for item in args.run]  # split at the first = only: 'a=b=c' -> ['a', 'b=c']
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    figure, axis = plt.subplots(figsize=(7, 4), dpi=200)
    baseline = None
    print(f"{'run':<28}{'first':>8}{'stays':>8}{'best':>8}{'extra':>8}")
    for position, (label, folder) in enumerate(runs):
        steps, means = read_log(folder)
        color = COLORS[position]
        axis.plot(steps, means, color=color, linewidth=1.5, label=label)
        stays = convergence_step(steps, means, args.target)
        if baseline is None:
            baseline = stays
        extra = "" if stays is None or baseline is None or position == 0 else step_label(stays - baseline)
        first = first_step_at(steps, means, args.target)
        print(
            f"{label:<28}{step_label(first) if first else '-':>8}{step_label(stays) if stays else '-':>8}"
            f"{max(means):>8.2f}{extra:>8}"
        )

    # The target line, labeled in place, so that no color has to be decoded.
    axis.axhline(args.target, color=INK_SECONDARY, linewidth=1, linestyle=(0, (4, 3)))
    axis.annotate(
        f"target {args.target:g}",
        xy=(0, args.target),
        xycoords=("axes fraction", "data"),
        xytext=(4, 4),
        textcoords="offset points",
        color=INK_SECONDARY,
        fontsize=8,
    )
    axis.set_xlabel("Environment steps", color=INK_SECONDARY, fontsize=9)
    axis.set_ylabel("Average score over 20 games", color=INK_SECONDARY, fontsize=9)
    axis.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: step_label(x) if x > 0 else "0"))
    axis.tick_params(colors=INK_SECONDARY, labelsize=8, length=0)
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    axis.spines["bottom"].set_color(GRID)
    if len(runs) > 1:
        axis.legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower right")
    figure.tight_layout()
    figure.savefig(args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
