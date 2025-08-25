#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any, Iterable, Optional

import matplotlib.pyplot as plt

LAYER_RE = re.compile(r"^layer_(\d+)$")
KNOWN_HOOKS = {"residual_out", "attn_out", "mlp_out"}

def load_json_objects(path: Path) -> Iterable[Dict[str, Any]]:
    """Load JSON content from a file (object, list of objects, or JSONL)."""
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return

    # Try plain JSON first
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            yield data
            return
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    yield item
            return
    except Exception:
        pass

    # Fall back to JSONL
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                yield obj
        except Exception:
            continue

def extract_layer_series(metric_dict: Dict[str, Any]) -> Optional[List[Tuple[int, float]]]:
    """Return sorted (layer_idx, value) pairs if dict has layer_* keys, else None."""
    pairs = []
    for k, v in metric_dict.items():
        m = LAYER_RE.match(k)
        if m:
            try:
                idx = int(m.group(1))
                val = float(v)
                pairs.append((idx, val))
            except (ValueError, TypeError):
                continue
    if not pairs:
        return None
    pairs.sort(key=lambda t: t[0])
    return pairs

def collect_series_from_object(
    obj: Dict[str, Any],
    source_tag: str
) -> Dict[Tuple[str, str], Dict[str, List[float]]]:
    """
    From a single JSON object, collect per-(hook, metric) series.
    - Known hooks (residual_out/attn_out/mlp_out): use directly.
    - Unknown top keys (e.g., "logit", "tuned"): treat as residual_out.
    - Ignore metrics that don't have per-layer values.
    - IMPORTANT: Top key "logit" gets a normalized series name "logit"
      so we can deduplicate across files later.
    """
    by_hook_metric: Dict[Tuple[str, str], Dict[str, List[float]]] = {}

    def add_series(hook: str, metric: str, series_name: str, layers: List[Tuple[int, float]]):
        key = (hook, metric)
        series_map = by_hook_metric.setdefault(key, {})
        series_map[series_name] = [v for _, v in layers]

    for top_key, top_val in obj.items():
        if not isinstance(top_val, dict):
            continue

        if top_key in KNOWN_HOOKS:
            hook = top_key
            for metric_name, metric_data in top_val.items():
                if not isinstance(metric_data, dict):
                    continue
                layers = extract_layer_series(metric_data)
                if layers is None:
                    continue
                series_name = f"{source_tag}"
                add_series(hook, metric_name, series_name, layers)
        else:
            # Default hook fallback
            hook = "residual_out"
            for metric_name, metric_data in top_val.items():
                if not isinstance(metric_data, dict):
                    continue
                layers = extract_layer_series(metric_data)
                if layers is None:
                    continue
                # Normalize series name for 'logit' so we can dedupe later
                if top_key == "logit":
                    series_name = "logit"
                else:
                    series_name = f"{source_tag} · {top_key}"
                add_series(hook, metric_name, series_name, layers)

    return by_hook_metric

def _series_almost_equal(a: List[float], b: List[float], tol: float = 1e-4) -> bool:
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if abs(x - y) > tol:
            return False
    return True

def merge_all_series(
    collected: List[Dict[Tuple[str, str], Dict[str, List[float]]]]
) -> Dict[Tuple[str, str], Dict[str, List[float]]]:
    """
    Merge object-level collections.
    Special handling for 'logit' series:
      - Only keep one per (hook, metric) even if many files contain essentially
        the same curve (treat nearly-identical as duplicates).
    """
    merged: Dict[Tuple[str, str], Dict[str, List[float]]] = {}
    for item in collected:
        for hk_metric, series_map in item.items():
            dest = merged.setdefault(hk_metric, {})
            for s_name, values in series_map.items():
                # If it's a 'logit' series, dedupe against any existing 'logit' curves
                if s_name == "logit":
                    found_duplicate = False
                    for existing_name, existing_vals in dest.items():
                        if existing_name == "logit" or existing_name.startswith("logit"):
                            if _series_almost_equal(existing_vals, values, tol=1e-4):
                                found_duplicate = True
                                break
                    if found_duplicate:
                        continue  # skip duplicate 'logit'

                    # If another 'logit' already exists but is different, keep only the first one.
                    # Comment the next block if you'd rather keep *all distinct* logits.
                    if any(n == "logit" or n.startswith("logit") for n in dest.keys()):
                        # Skip additional different 'logit' curves; keep the first only.
                        continue

                    # Otherwise, accept this 'logit' series with canonical name
                    dest["logit"] = values
                    continue

                # Non-logit series: merge normally
                if s_name in dest:
                    if _series_almost_equal(dest[s_name], values):
                        continue  # identical; skip
                    # make a unique suffix if different
                    counter = 2
                    new_name = f"{s_name} ({counter})"
                    while new_name in dest and not _series_almost_equal(dest[new_name], values):
                        counter += 1
                        new_name = f"{s_name} ({counter})"
                    dest[new_name] = values
                else:
                    dest[s_name] = values
    return merged

def plot_all(merged: Dict[Tuple[str, str], Dict[str, List[float]]], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    for (hook, metric), series_map in sorted(merged.items()):
        if not series_map:
            continue

        n_series = len(series_map)
        place_below = True  # ✅ always put legends below now
        figsize = (8, 5.75)

        plt.figure(figsize=figsize)
        for s_name, values in sorted(series_map.items()):
            xs = list(range(len(values)))
            plt.plot(xs, values, label=s_name)

        plt.title(f"{hook} – {metric}")
        plt.xlabel("Layer index")
        plt.ylabel(metric)
        plt.yscale("log")   # log scale on metric axis
        plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)  # ✅ add grid

        if place_below:
            # legend below, compact style
            ncol = min(4, max(2, (n_series + 5) // 6))
            plt.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.15),
                ncol=ncol,
                fontsize="small",
                frameon=True,
                framealpha=0.9,
                borderaxespad=0.0,
                handlelength=2.0,
                columnspacing=0.8,
                labelspacing=0.3,
            )
            plt.tight_layout()
            fname = f"{hook}_{metric}.png".replace("/", "_")
            plt.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Plot evaluation metrics by hook point and metric (aggregate_metrics.json only).")
    parser.add_argument("--base-dir", type=str, default="../../evaluation",
                        help="Root directory to search (default: ../../evaluation)")
    parser.add_argument("--out-dir", type=str, default="./plots",
                        help="Directory to save plots (default: ./plots)")
    parser.add_argument("--max-files", type=int, default=0,
                        help="Debug limit: stop after reading this many files (default: 0 = no limit)")
    args = parser.parse_args()

    base = Path(args.base_dir).resolve()
    out = Path(args.out_dir).resolve()

    collected_objects: List[Dict[Tuple[str, str], Dict[str, List[float]]]] = []
    files_seen = 0

    # Only match aggregate_metrics.json anywhere under base
    for path in base.rglob("aggregate_metrics.json"):
        if not path.is_file():
            continue

        for obj in load_json_objects(path):
            try:
                rel = path.relative_to(base)
            except Exception:
                rel = path.name
            # ✅ strip trailing 'aggregate_metrics.json'
            source_tag = str(rel).replace("/aggregate_metrics.json", "").replace("\\aggregate_metrics.json", "")

            series_from_obj = collect_series_from_object(obj, source_tag=source_tag)
            if series_from_obj:
                collected_objects.append(series_from_obj)

        files_seen += 1
        if args.max_files > 0 and files_seen >= args.max_files:
            break

    merged = merge_all_series(collected_objects)
    plot_all(merged, out)

    print(f"Processed {files_seen} aggregate_metrics.json file(s).")
    print(f"Generated {len(merged)} plot image(s) under: {out}")

if __name__ == "__main__":
    main()