import argparse
import json
import os
from collections import Counter
from typing import Dict, List

import matplotlib.pyplot as plt


def load_rows(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def summarize(rows: List[dict]) -> Dict:
    verdict_counts = Counter()
    evidence_counts: List[int] = []
    evidence_ge2 = 0
    agree_flags: List[bool] = []
    agree_on_cov_flags: List[bool] = []
    binary_used = binary_correct = 0
    pred_labels = Counter()
    unknown_count = 0

    for r in rows:
        m = r.get("metrics", {})
        cv = m.get("claim_verification", {}) or {}
        mv = m.get("model_vs_verifier", {}) or {}
        ml = m.get("model_label_metrics", {}) or {}

        verdict = cv.get("verdict", "nei") or "nei"
        verdict_counts[verdict] += 1

        evc = int(cv.get("evidence_count", 0) or 0)
        evidence_counts.append(evc)
        if evc >= 2:
            evidence_ge2 += 1

        agree = mv.get("agree")
        if agree is not None:
            agree_flags.append(bool(agree))

        aoc = mv.get("agree_on_covered")
        if aoc is not None:
            agree_on_cov_flags.append(bool(aoc))

        pred_label = ml.get("pred_label")
        if pred_label is not None:
            pred_labels[pred_label] += 1
            if pred_label == "unknown":
                unknown_count += 1

        if ml.get("is_binary"):
            binary_used += 1
            if ml.get("correct_binary"):
                binary_correct += 1

    n = len(rows)
    coverage_count = n - verdict_counts.get("nei", 0)
    return {
        "n": n,
        "coverage_count": coverage_count,
        "coverage_rate": coverage_count / n if n else 0.0,
        "verdict_counts": verdict_counts,
        "evidence_counts": evidence_counts,
        "evidence_ge2_count": evidence_ge2,
        "agree_rate": sum(agree_flags) / len(agree_flags) if agree_flags else 0.0,
        "agree_on_cov_rate": sum(agree_on_cov_flags) / len(agree_on_cov_flags) if agree_on_cov_flags else 0.0,
        "agree_on_cov_n": len(agree_on_cov_flags),
        "binary_acc": binary_correct / binary_used if binary_used else 0.0,
        "binary_n": binary_used,
        "pred_labels": pred_labels,
        "unknown_count": unknown_count,
    }


def bar_with_delta(title: str, labels, vals, counts, outfile: str):
    plt.figure()
    bars = plt.bar(labels, [v * 100 for v in vals])
    for b, c in zip(bars, counts):
        plt.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{c}", ha="center", va="bottom", fontsize=8)
    if len(vals) == 2:
        delta_pp = (vals[1] - vals[0]) * 100
        plt.title(f"{title}\nΔpp (right-left): {delta_pp:.1f}")
    else:
        plt.title(title)
    plt.ylabel("Percent")
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()


def two_bar_with_rel(
    title: str,
    labels,
    vals,
    ns,
    outfile: str,
    show_rel: bool = True,
    show_pp: bool = True,
    percent: bool = True,
):
    plt.figure()
    display_vals = [v * 100 if percent else v for v in vals]
    bars = plt.bar(labels, display_vals)
    for b, disp in zip(bars, display_vals):
        unit = "%" if percent else ""
        plt.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{disp:.1f}{unit}", ha="center", va="bottom", fontsize=8)
    subtitle_parts = []
    if len(vals) == 2:
        if show_pp:
            delta_pp = (vals[1] - vals[0]) * (100 if percent else 1)
            subtitle_parts.append(f"Δpp: {delta_pp:.1f}" if percent else f"Δ: {delta_pp:.3f}")
        if show_rel and vals[0] != 0:
            rel = (vals[1] - vals[0]) / vals[0] * 100
            subtitle_parts.append(f"rel: {rel:.1f}%")
    plt.title(title + ("\n" + " | ".join(subtitle_parts) if subtitle_parts else ""))
    if percent:
        plt.ylabel("Percent")
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()


def stacked_verdict(title: str, labels, verdict_counts_list, outfile: str):
    order = ["entail", "contradict", "nei"]
    bottoms = [0] * len(labels)
    plt.figure()
    for v in order:
        heights = [vc.get(v, 0) for vc in verdict_counts_list]
        plt.bar(labels, heights, bottom=bottoms, label=v)
        bottoms = [b + h for b, h in zip(bottoms, heights)]
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()


def hist_two(title: str, data_a, data_b, labels, outfile: str):
    plt.figure()
    plt.hist([data_a, data_b], bins=10, label=labels, alpha=0.7)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()


def summary_table(tag: str, rows: Dict[str, Dict], outfile: str):
    headers = [
        "run",
        "coverage",
        "evidence>=2",
        "binary_acc",
        "agree_on_cov",
        "unknown_rate",
    ]
    table_data = []
    for name, s in rows.items():
        table_data.append(
            [
                name,
                f"{s['coverage_rate']*100:.1f}% ({s['coverage_count']}/{s['n']})",
                f"{(s['evidence_ge2_count']/s['n']*100):.1f}% ({s['evidence_ge2_count']}/{s['n']})",
                f"{s['binary_acc']*100:.1f}% (n={s['binary_n']})",
                f"{s['agree_on_cov_rate']*100:.1f}% (n={s['agree_on_cov_n']})",
                f"{(s['unknown_count']/s['n']*100):.1f}% ({s['unknown_count']}/{s['n']})",
            ]
        )

    fig, ax = plt.subplots()
    ax.axis("off")
    table = ax.table(cellText=table_data, colLabels=headers, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)
    plt.title(f"Summary {tag}")
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wiki8", required=True)
    ap.add_argument("--wiki70", required=True)
    ap.add_argument("--fever8", required=True)
    ap.add_argument("--fever70", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    summaries = {
        "wiki_8B": summarize(load_rows(args.wiki8)),
        "wiki_70B": summarize(load_rows(args.wiki70)),
        "fever_8B": summarize(load_rows(args.fever8)),
        "fever_70B": summarize(load_rows(args.fever70)),
    }

    # Corpus effect per model
    for model, key8, key70 in [("8B", "wiki_8B", "fever_8B"), ("70B", "wiki_70B", "fever_70B")]:
        w = summaries[key8]
        f = summaries[key70]
        labels = ["wiki853", "FEVER50k"]
        bar_with_delta(
            "Coverage (wiki vs FEVER)",
            labels,
            [w["coverage_rate"], f["coverage_rate"]],
            [w["coverage_count"], f["coverage_count"]],
            os.path.join(args.outdir, f"coverage_corpus_{model}.png"),
        )
        two_bar_with_rel(
            "Evidence count ≥2 (wiki vs FEVER)",
            labels,
            [w["evidence_ge2_count"] / w["n"] if w["n"] else 0.0, f["evidence_ge2_count"] / f["n"] if f["n"] else 0.0],
            [w["n"], f["n"]],
            os.path.join(args.outdir, f"evidence_ge2_corpus_{model}.png"),
            show_rel=False,
        )
        two_bar_with_rel(
            "Mean evidence count (wiki vs FEVER)",
            labels,
            [
                sum(w["evidence_counts"]) / w["n"] if w["n"] else 0.0,
                sum(f["evidence_counts"]) / f["n"] if f["n"] else 0.0,
            ],
            [w["n"], f["n"]],
            os.path.join(args.outdir, f"evidence_mean_corpus_{model}.png"),
            show_rel=False,
            show_pp=False,
            percent=False,
        )
        stacked_verdict(
            "Verdict mix counts (wiki vs FEVER)",
            labels,
            [w["verdict_counts"], f["verdict_counts"]],
            os.path.join(args.outdir, f"verdict_corpus_{model}.png"),
        )
        two_bar_with_rel(
            "Agree overall (wiki vs FEVER)",
            labels,
            [w["agree_rate"], f["agree_rate"]],
            [w["n"], f["n"]],
            os.path.join(args.outdir, f"agree_corpus_{model}.png"),
            show_rel=False,
        )
        two_bar_with_rel(
            "Agree on covered (wiki vs FEVER)",
            labels,
            [w["agree_on_cov_rate"], f["agree_on_cov_rate"]],
            [w["agree_on_cov_n"], f["agree_on_cov_n"]],
            os.path.join(args.outdir, f"agree_cov_corpus_{model}.png"),
            show_rel=False,
        )
        hist_two(
            f"Evidence count dist ({model})",
            w["evidence_counts"],
            f["evidence_counts"],
            labels,
            os.path.join(args.outdir, f"evidence_hist_corpus_{model}.png"),
        )

    # Model effect per corpus
    for corpus, key8, key70 in [("wiki853", "wiki_8B", "wiki_70B"), ("FEVER50k", "fever_8B", "fever_70B")]:
        s8 = summaries[key8]
        s70 = summaries[key70]
        labels = ["8B", "70B"]
        two_bar_with_rel(
            f"Model label acc (binary) {corpus} (unknown excluded)",
            labels,
            [s8["binary_acc"], s70["binary_acc"]],
            [s8["binary_n"], s70["binary_n"]],
            os.path.join(args.outdir, f"model_acc_{corpus}.png"),
            show_rel=True,
            show_pp=True,
        )
        two_bar_with_rel(
            f"Model vs verifier agree {corpus}",
            labels,
            [s8["agree_rate"], s70["agree_rate"]],
            [s8["n"], s70["n"]],
            os.path.join(args.outdir, f"agree_model_{corpus}.png"),
            show_rel=True,
            show_pp=True,
        )
        two_bar_with_rel(
            f"Agree on covered {corpus}",
            labels,
            [s8["agree_on_cov_rate"], s70["agree_on_cov_rate"]],
            [s8["agree_on_cov_n"], s70["agree_on_cov_n"]],
            os.path.join(args.outdir, f"agree_cov_model_{corpus}.png"),
            show_rel=True,
            show_pp=True,
        )
        two_bar_with_rel(
            f"Unknown rate {corpus}",
            labels,
            [s8["unknown_count"] / s8["n"] if s8["n"] else 0.0, s70["unknown_count"] / s70["n"] if s70["n"] else 0.0],
            [s8["n"], s70["n"]],
            os.path.join(args.outdir, f"unknown_model_{corpus}.png"),
            show_rel=True,
            show_pp=True,
        )

    # Summary table graphic
    summary_table(
        "wiki vs FEVER (200 LIAR claims)",
        summaries,
        os.path.join(args.outdir, "summary_table.png"),
    )

    # Print brief stats to console
    for name, s in summaries.items():
        print(
            name,
            {
                "coverage": f"{s['coverage_count']}/{s['n']}",
                "evidence_ge2": f"{s['evidence_ge2_count']}/{s['n']}",
                "binary_acc": f"{s['binary_acc']:.3f} (n={s['binary_n']})",
                "agree": f"{s['agree_rate']:.3f}",
                "agree_on_cov": f"{s['agree_on_cov_rate']:.3f} (n={s['agree_on_cov_n']})",
                "unknown": s["unknown_count"],
            },
        )


if __name__ == "__main__":
    main()
