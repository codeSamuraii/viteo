#!/usr/bin/env python3
"""Benchmark: viteo vs OpenCV frame extraction performance."""

import argparse
import statistics
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt

import viteo

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "tests" / "samples"


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark viteo vs OpenCV")
    parser.add_argument("videos", nargs="*", help="Video files to benchmark")
    parser.add_argument("--save", metavar="PATH", help="Save chart to PNG")
    parser.add_argument("--frames", type=int, default=300, help="Frames per video")
    return parser.parse_args()


def discover_videos(paths: list[str]) -> list[Path]:
    if paths:
        return [Path(p) for p in paths]
    videos = sorted(SAMPLES_DIR.glob("*.mp4"))
    if not videos:
        raise SystemExit(f"No .mp4 files found in {SAMPLES_DIR}")
    return videos


def bench_viteo(path: Path, n: int) -> list[float]:
    times = []
    with viteo.open(str(path)) as extractor:
        it = iter(extractor)
        for _ in range(n):
            t0 = time.perf_counter()
            try:
                next(it)
            except StopIteration:
                break
            times.append(time.perf_counter() - t0)
    return times


def bench_opencv(path: Path, n: int) -> list[float]:
    times = []
    cap = cv2.VideoCapture(str(path))
    try:
        for _ in range(n):
            t0 = time.perf_counter()
            ret, _ = cap.read()
            times.append(time.perf_counter() - t0)
            if not ret:
                times.pop()
                break
    finally:
        cap.release()
    return times


def percentile(sorted_vals: list[float], pct: float) -> float:
    idx = int(len(sorted_vals) * pct)
    return sorted_vals[min(idx, len(sorted_vals) - 1)]


def compute_metrics(times: list[float]) -> dict:
    sorted_t = sorted(times)
    ms_avg = statistics.mean(times) * 1000
    ms_p50 = percentile(sorted_t, 0.50) * 1000
    ms_p90 = percentile(sorted_t, 0.90) * 1000
    ms_p99 = percentile(sorted_t, 0.99) * 1000
    return {
        "frames": len(times),
        "ms_avg": ms_avg,
        "ms_p50": ms_p50,
        "ms_p90": ms_p90,
        "ms_p99": ms_p99,
        "fps_avg": 1000 / ms_avg,
        "fps_p50": 1000 / ms_p50,
        "fps_p90": 1000 / ms_p90,
        "fps_p99": 1000 / ms_p99,
    }


def print_table(results: list[dict]):
    header = f"{'Backend':<10} {'Frames':>6} {'avg ms':>8} {'p50 ms':>8} {'p90 ms':>8} {'p99 ms':>8} {'avg FPS':>9} {'p50 FPS':>9} {'p90 FPS':>9} {'p99 FPS':>9}"
    sep = "-" * len(header)

    for entry in results:
        print(f"\n  {entry['video']}")
        print(f"  {sep}")
        print(f"  {header}")
        print(f"  {sep}")
        for backend in ("viteo", "opencv"):
            m = entry[backend]
            print(
                f"  {backend:<10} {m['frames']:>6} "
                f"{m['ms_avg']:>8.2f} {m['ms_p50']:>8.2f} {m['ms_p90']:>8.2f} {m['ms_p99']:>8.2f} "
                f"{m['fps_avg']:>9.1f} {m['fps_p50']:>9.1f} {m['fps_p90']:>9.1f} {m['fps_p99']:>9.1f}"
            )
    print()


def _label_bars(ax, bars):
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=7)


def plot_charts(results: list[dict], save: str | None):
    labels = [r["video"] for r in results]
    n = len(labels)
    x = range(n)
    w = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(12, n * 3), 6))

    # --- Chart 1: Average FPS ---
    v_avg = [r["viteo"]["fps_avg"] for r in results]
    o_avg = [r["opencv"]["fps_avg"] for r in results]
    bars_v = ax1.bar([i - w / 2 for i in x], v_avg, w, label="viteo", color="#4C72B0")
    bars_o = ax1.bar([i + w / 2 for i in x], o_avg, w, label="OpenCV", color="#DD8452")
    _label_bars(ax1, bars_v)
    _label_bars(ax1, bars_o)
    ax1.set_ylabel("FPS")
    ax1.set_title("Average FPS")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax1.legend()

    # --- Chart 2: Percentile ms/frame ---
    pcts = ["p50", "p90", "p99"]
    bar_w = 0.13
    offsets = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
    colors_v = ["#4C72B0", "#7BA3D4", "#AFC6E4"]
    colors_o = ["#DD8452", "#EEAC82", "#F5CFB0"]

    for j, pct in enumerate(pcts):
        v_vals = [r["viteo"][f"ms_{pct}"] for r in results]
        o_vals = [r["opencv"][f"ms_{pct}"] for r in results]
        ax2.bar([i + offsets[j] * bar_w for i in x], v_vals, bar_w,
                label=f"viteo {pct}", color=colors_v[j])
        ax2.bar([i + offsets[j + 3] * bar_w for i in x], o_vals, bar_w,
                label=f"OpenCV {pct}", color=colors_o[j])

    ax2.set_ylabel("ms / frame")
    ax2.set_title("Latency Percentiles")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax2.legend(fontsize=7, ncol=2)

    fig.suptitle("Frame Extraction: viteo vs OpenCV", fontsize=13, y=1.02)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
        print(f"Chart saved to {save}")
    else:
        plt.show()


def main():
    args = parse_args()
    videos = discover_videos(args.videos)
    n = args.frames

    results = []
    for video in videos:
        print(f"Benchmarking {video.name} ({n} frames)...")
        vt = bench_viteo(video, n)
        ot = bench_opencv(video, n)
        results.append({
            "video": video.name,
            "viteo": compute_metrics(vt),
            "opencv": compute_metrics(ot),
        })

    print_table(results)
    plot_charts(results, args.save)


if __name__ == "__main__":
    main()
