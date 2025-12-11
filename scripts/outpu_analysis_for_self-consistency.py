
import json
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def extract_dataframe(rows):
    records = []

    for r in rows:
        model_out = r.get("model_output", {})
        metrics = r.get("metrics", {})
        sc = metrics.get("self_consistency", {})

        if "consistency" not in sc or "confidence" not in model_out:
            continue

        records.append({
            "id": r.get("id"),
            "claim": r.get("claim"),
            "label": model_out.get("label"),
            "confidence": float(model_out.get("confidence")),
            "consistency": float(sc.get("consistency")),
            "risk": float(sc.get("risk"))
        })

    return pd.DataFrame(records)


def aggregate_stats(df):
    return {
        "mean_consistency": df["consistency"].mean(),
        "median_consistency": df["consistency"].median(),
        "std_consistency": df["consistency"].std(),
        "mean_risk": df["risk"].mean(),
    }


def per_label_stats(df):
    return (
        df.groupby("label")
        .agg(
            avg_consistency=("consistency", "mean"),
            std_consistency=("consistency", "std"),
            avg_risk=("risk", "mean"),
            count=("consistency", "count")
        )
        .reset_index()
    )


def _apply_plot_style():
    plt.grid(True, linestyle="--", alpha=0.6)


def plot_histogram(series, title, xlabel, out_path):
    plt.figure()
    plt.hist(series, bins=20, rwidth=0.85)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    _apply_plot_style()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_scatter(df, title, out_path):
    plt.figure()
    plt.scatter(df["confidence"], df["consistency"])
    plt.xlabel("Confidence")
    plt.ylabel("Self-Consistency")
    plt.title(title)
    _apply_plot_style()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def show_examples(df, k=3):
    print("\n=== Highest Consistency Examples ===")
    for _, r in df.sort_values("consistency", ascending=False).head(k).iterrows():
        print(f"\n[{r['label']} | consistency={r['consistency']:.3f}]")
        print(r["claim"])

    print("\n=== Lowest Consistency Examples ===")
    for _, r in df.sort_values("consistency").head(k).iterrows():
        print(f"\n[{r['label']} | consistency={r['consistency']:.3f}]")
        print(r["claim"])


def main(args):
    rows = load_jsonl(args.input)
    df = extract_dataframe(rows)

    model_name = args.model_name

    print(f"\nLoaded {len(df)} valid samples")

    print("\n=== Aggregate Metrics ===")
    stats = aggregate_stats(df)
    for k, v in stats.items():
        print(f"{k}: {v:.4f}")

    print("\n=== Per-label Consistency ===")
    label_stats = per_label_stats(df)
    print(label_stats.to_string(index=False))

    corr, p = pearsonr(df["confidence"], df["consistency"])
    print(f"\nConfidence–Consistency Pearson r = {corr:.3f} (p={p:.3g})")

    plot_histogram(
        df["consistency"],
        f"Self-Consistency Distribution ({model_name})",
        "Self-Consistency",
        f"{args.outdir}/consistency_hist.png"
    )

    plot_histogram(
        df["risk"],
        f"Risk Distribution ({model_name})",
        "Risk (1 − Consistency)",
        f"{args.outdir}/risk_hist.png"
    )

    plot_scatter(
        df,
        f"Confidence vs Self-Consistency ({model_name})",
        f"{args.outdir}/confidence_vs_consistency.png"
    )

    show_examples(df, k=args.examples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to JSONL output file")
    parser.add_argument("--outdir", default="results", help="Output directory")
    parser.add_argument("--examples", type=int, default=3)
    parser.add_argument(
        "--model-name",
        default="LLaMA 70B",
        help="Model name to use in plot titles (e.g. 'LLaMA 8B')"
    )
    args = parser.parse_args()

    main(args)