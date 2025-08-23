#!/usr/bin/env python3
"""
Make ONE comparison plot for a single metric (e.g., KL), overlaying:
- all runs found under <root> (recursively finds aggregate_metrics.json),
- all hook groups in each run (residual_out, attn_out, mlp_out, logit if present),
- baseline 'final' as dashed horizontal line per run if present.

Output: <root>/<metric>_comparison_all.png

Default axes are log–log. Change with --xscale/--yscale.
"""

from __future__ import annotations
import argparse, json, re
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

LAYER_KEY_RE = re.compile(r"layer_(\d+)$")
HOOK_ORDER = ["residual_out", "attn_out", "mlp_out", "logit"]

def find_aggregate_files(root: Path) -> List[Path]:
    return sorted(root.rglob("aggregate_metrics.json"))

def run_name(run_dir: Path) -> str:
    # e.g. ".../evaluation/gpt2/TunedLens" -> "gpt2/TunedLens"
    parent = run_dir.parent.name
    return f"{parent}/{run_dir.name}"

def extract_series_for_metric(data: Dict, metric: str) -> Dict[str, Tuple[List[int], List[float]]]:
    """
    Returns: { hook_name: ([x=layer+1], [y=value]) } for the requested metric.
    Skips hooks that don't have layer_* entries for this metric.
    """
    series = {}
    for hook_name, block in data.items():
        if not isinstance(block, dict) or metric not in block: 
            continue
        metric_map = block[metric]
        if not isinstance(metric_map, dict): 
            continue
        pairs = []
        for k, v in metric_map.items():
            m = LAYER_KEY_RE.fullmatch(k)
            if m:
                try:
                    idx = int(m.group(1))
                    pairs.append((idx, float(v)))
                except Exception:
                    pass
        if not pairs:
            continue
        pairs.sort(key=lambda t: t[0])
        xs = [i + 1 for (i, _) in pairs]   # layer 0 -> x=1 (safe for log-x)
        ys = [y for (_, y) in pairs]
        series[hook_name] = (xs, ys)
    return series

def extract_baseline_final(data: Dict, metric: str):
    base = data.get("baseline")
    if not isinstance(base, dict):
        return None
    fm = base.get(metric)
    if isinstance(fm, dict) and "final" in fm:
        try:
            val = float(fm["final"])
            return val if val > 0 else None
        except Exception:
            return None
    return None

def main():
    ap = argparse.ArgumentParser(description="ONE comparison plot for a single metric across all runs/hooks.")
    ap.add_argument("root", type=Path, help="Evaluation root directory (searches recursively for aggregate_metrics.json)")
    ap.add_argument("--metric", default="kl", help="Metric to plot (e.g., kl, ce, entropy). Default: kl")
    ap.add_argument("--xscale", choices=["log","linear"], default="log")
    ap.add_argument("--yscale", choices=["log","linear"], default="log")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    files = find_aggregate_files(args.root)
    if not files:
        print(f"[warn] No aggregate_metrics.json under {args.root}")
        return

    # Collect per-run data for the chosen metric
    per_run = []  # list of (label, series_dict, baseline_val)
    for f in files:
        try:
            data = json.loads(f.read_text())
        except Exception as e:
            print(f"[skip] {f}: {e}")
            continue
        label = run_name(f.parent)
        series = extract_series_for_metric(data, args.metric)
        base = extract_baseline_final(data, args.metric)
        per_run.append((label, series, base))

    # Create ONE figure
    plt.figure(figsize=(10, 6))

    any_plotted = False
    all_xs = []

    # Plot each run's hook curves on the same axes; fixed hook order for consistency
    for label, series, base in per_run:
        # If there are multiple hooks, we’ll append hook-name to legend label
        # Example curve names: "gpt2/TunedLens: residual_out"
        # If a run has only one hook, keep it concise.
        multiple_hooks = len(series) > 1
        # Plot hooks in a consistent order
        for hook in [h for h in HOOK_ORDER if h in series] + [h for h in series if h not in HOOK_ORDER]:
            xs, ys = series[hook]
            # drop non-positives for log safety
            xs_pos = [x for x, y in zip(xs, ys) if x > 0 and y > 0]
            ys_pos = [y for x, y in zip(xs, ys) if x > 0 and y > 0]
            if not xs_pos:
                continue
            all_xs.extend(xs_pos)
            curve_label = f"{label}: {hook}" if multiple_hooks else label
            plt.plot(xs_pos, ys_pos, marker="o", linewidth=1.6, label=curve_label)
            any_plotted = True

        # Baseline as dashed line (per run), spanning visible x-range
        if base is not None and base > 0:
            # if we have any x data already, span real range; else span [1,2] just to show it
            x_min, x_max = (min(all_xs), max(all_xs)) if all_xs else (1, 2)
            plt.hlines(base, x_min, x_max, linestyles="dashed", alpha=0.6, label=f"{label}: baseline={base:.3g}")

    plt.xscale(args.xscale)
    plt.yscale(args.yscale)
    plt.xlabel("Layer index + 1" + (" (log)" if args.xscale == "log" else ""))
    plt.ylabel(args.metric.upper())
    plt.title(f"All runs/hooks — {args.metric.upper()} ({args.xscale}-{args.yscale})")
    plt.grid(True, which="both", alpha=0.25)
    if any_plotted:
        plt.legend(fontsize=9, ncol=2)
    plt.tight_layout()

    out_path = args.root / f"{args.metric}_comparison_all.png"
    plt.savefig(out_path, dpi=args.dpi)
    plt.close()
    print(f"[saved] {out_path}")

if __name__ == "__main__":
    main()