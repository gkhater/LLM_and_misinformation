"""
Scaffolding to tune NLI thresholds on a labeled dev set.

Input CSV columns:
  claim, evidence, label   # label in {support, refute, unknown}

Usage:
  python scripts/calibrate_nli_thresholds.py --dev dev_labels.csv --model ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli

This runs a small grid over entail/contradict thresholds and reports macro accuracy.
"""

import argparse
import csv
from itertools import product

import numpy as np
from transformers import pipeline


def load_dev(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((row["claim"], row["evidence"], row["label"].lower()))
    return rows


def eval_grid(dev_rows, model_name, thresholds):
    clf = pipeline(
        "text-classification",
        model=model_name,
        tokenizer=model_name,
        return_all_scores=True,
    )

    best = None
    for ent, contra in thresholds:
        correct = 0
        total = 0
        for claim, evidence, label in dev_rows:
            pred = clf({"text": evidence, "text_pair": claim})[0]
            lbl = pred["label"].upper()
            score = pred["score"]
            if "ENTAIL" in lbl and score >= ent:
                got = "support"
            elif "CONTRAD" in lbl and score >= contra:
                got = "refute"
            else:
                got = "unknown"
            correct += int(got == label)
            total += 1
        acc = correct / total if total else 0.0
        if best is None or acc > best[0]:
            best = (acc, ent, contra)
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", required=True, help="CSV with claim,evidence,label")
    parser.add_argument("--model", required=True, help="NLI model name")
    args = parser.parse_args()

    dev_rows = load_dev(args.dev)
    thresholds = [(e, c) for e, c in product([0.5, 0.6, 0.7, 0.8], repeat=2)]
    best = eval_grid(dev_rows, args.model, thresholds)
    if best:
        acc, ent, contra = best
        print(f"Best accuracy {acc:.3f} with entail={ent}, contradict={contra}")
    else:
        print("No results.")


if __name__ == "__main__":
    main()
