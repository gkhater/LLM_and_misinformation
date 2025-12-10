"""
Quick NLI sanity check to validate label mapping and premise/hypothesis order.

Usage:
  python scripts/nli_sanity.py --model typeform/distilbert-base-uncased-mnli
"""

import argparse

from transformers import pipeline


def main():
    parser = argparse.ArgumentParser(description="Run a small NLI sanity test.")
    parser.add_argument("--model", default="typeform/distilbert-base-uncased-mnli")
    args = parser.parse_args()

    nli = pipeline("text-classification", model=args.model, tokenizer=args.model, return_all_scores=True, truncation=True)

    tests = [
        ("Dogs are animals.", "Dogs are animals.", "entail expected"),
        ("Dogs are animals.", "Dogs are plants.", "contradict expected"),
        ("Dogs are animals.", "Dogs exist.", "entail moderate expected"),
    ]
    for premise, hypothesis, note in tests:
        scores_raw = nli({"text": premise, "text_pair": hypothesis})[0]
        mapped = {}
        for s in scores_raw:
            lbl = s["label"].lower()
            if "entail" in lbl or lbl == "entailment":
                mapped["entail"] = float(s["score"])
            elif "contrad" in lbl or "refute" in lbl or lbl == "contradiction":
                mapped["contradict"] = float(s["score"])
            elif "neutral" in lbl:
                mapped["neutral"] = float(s["score"])
        mapped.setdefault("entail", 0.0)
        mapped.setdefault("contradict", 0.0)
        mapped.setdefault("neutral", 0.0)
        print(f"\nPremise: {premise}\nHypothesis: {hypothesis}\nNote: {note}\nScores: {mapped}")


if __name__ == "__main__":
    main()
