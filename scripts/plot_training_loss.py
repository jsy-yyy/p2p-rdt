#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOSS_PATTERN = re.compile(r"global_step=(?P<global_step>\d+).*?training_loss=(?P<training_loss>[-+eE0-9.]+)")

def parse_args():
    parser = argparse.ArgumentParser(description="Extract training_loss from training logs and plot it.")
    parser.add_argument("logs", nargs="+", type=Path, help="One or more log files such as train_adjust_bottle_random_init.log")
    parser.add_argument("--output", type=Path, default=Path("training_loss.png"), help="Output image path. Default: training_loss.png")
    parser.add_argument("--csv-dir", type=Path, default=None, help="Optional directory for extracted CSV files.")
    parser.add_argument("--show-smooth", action="store_true", help="Also draw a moving-average smoothing curve.")
    parser.add_argument("--smooth-window", type=int, default=11, help="Moving-average window used with --show-smooth. Default: 11")
    parser.add_argument("--title", type=str, default="Training Loss", help="Plot title.")
    return parser.parse_args()

def extract_points(log_path):
    step_to_loss = {}
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = LOSS_PATTERN.search(line)
            if not match:
                continue
            step = int(match.group("global_step"))
            loss = float(match.group("training_loss"))
            step_to_loss[step] = loss
    return sorted(step_to_loss.items())

def moving_average(values, window):
    if not values:
        return []
    window = max(1, min(window, len(values)))
    averaged = []
    running_sum = 0.0
    for index, value in enumerate(values):
        running_sum += value
        if index >= window:
            running_sum -= values[index - window]
        current_window = min(index + 1, window)
        averaged.append(running_sum / current_window)
    return averaged

def write_csv(points, csv_path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["global_step", "training_loss"])
        writer.writerows(points)

def main():
    args = parse_args()
    plt.figure(figsize=(10, 6))
    any_points = False
    for log_path in args.logs:
        if not log_path.is_file():
            raise FileNotFoundError(f"Log file not found: {log_path}")
        points = extract_points(log_path)
        if not points:
            print(f"[warn] no training_loss found in {log_path}")
            continue
        any_points = True
        steps = [step for step, _ in points]
        losses = [loss for _, loss in points]
        label = log_path.stem
        raw_label = f"{label} raw" if args.show_smooth else label
        plt.plot(steps, losses, alpha=0.85, linewidth=1.2, label=raw_label)
        if args.show_smooth:
            smoothed = moving_average(losses, args.smooth_window)
            plt.plot(steps, smoothed, linewidth=2.0, label=f"{label} smooth")
        if args.csv_dir is not None:
            csv_path = args.csv_dir / f"{log_path.stem}.csv"
            write_csv(points, csv_path)
            print(f"[ok] wrote CSV: {csv_path}")
        print(f"[ok] parsed {len(points)} points from {log_path} (step {steps[0]} -> {steps[-1]})")
    if not any_points:
        raise RuntimeError("No training_loss entries were found in the provided logs.")
    plt.title(args.title)
    plt.xlabel("Global Step")
    plt.ylabel("Training Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=180)
    print(f"[ok] wrote plot: {args.output}")

if __name__ == "__main__":
    main()
