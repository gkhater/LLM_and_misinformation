import argparse
import json
import math
import os
import random
from collections import Counter, defaultdict
from typing import Dict, List, Tuple


def load_rows(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def extract_label(row: dict) -> Tuple[int, bool, str]:
    """Return pred_polarity, is_binary, raw_pred_label."""
    ml = (row.get("metrics") or {}).get("model_label_metrics") or {}
    return ml.get("pred_polarity"), bool(ml.get("is_binary")), ml.get("pred_label")


def extract_gold(row: dict) -> int:
    ml = (row.get("metrics") or {}).get("model_label_metrics") or {}
    return ml.get("gold_polarity")


def extract_correct(row: dict) -> bool:
    ml = (row.get("metrics") or {}).get("model_label_metrics") or {}
    return bool(ml.get("correct_binary")) if ml.get("is_binary") else None


def extract_verifier(row: dict) -> int:
    mv = (row.get("metrics") or {}).get("model_vs_verifier") or {}
    return mv.get("verifier_polarity")


def cohen_kappa(labels_a: List[int], labels_b: List[int]) -> float:
    # Three-way: -1, 0, 1
    n = len(labels_a)
    if n == 0:
        return float("nan")
    counts = Counter(zip(labels_a, labels_b))
    marg_a = Counter(labels_a)
    marg_b = Counter(labels_b)
    observed = sum(v for (a, b), v in counts.items() if a == b) / n
    expected = sum((marg_a[k] / n) * (marg_b[k] / n) for k in {-1, 0, 1})
    if math.isclose(1 - expected, 0.0):
        return float("nan")
    return (observed - expected) / (1 - expected)


def mcnemar(b01: int, b10: int) -> float:
    """Exact McNemar p-value with continuity correction (approx if large)."""
    from math import comb

    n = b01 + b10
    if n == 0:
        return float("nan")
    # two-sided binomial test with p=0.5
    k = min(b01, b10)
    p = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    p *= 2
    return min(1.0, p)


def bootstrap_ci_delta(acc8: List[int], acc70: List[int], iters: int = 5000, seed: int = 13) -> Tuple[float, float, float]:
    rng = random.Random(seed)
    n = len(acc8)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    deltas = []
    for _ in range(iters):
        idx = [rng.randrange(n) for _ in range(n)]
        a8 = sum(acc8[i] for i in idx) / n
        a70 = sum(acc70[i] for i in idx) / n
        deltas.append(a70 - a8)
    deltas.sort()
    return deltas[int(0.5 * iters)], deltas[int(0.025 * iters)], deltas[int(0.975 * iters)]


def confusion(acc_list: List[Tuple[int, int]]) -> Dict[str, int]:
    # entries are (gold, pred)
    cm = Counter(acc_list)
    return {
        "tp": cm[(1, 1)],
        "tn": cm[(-1, -1)],
        "fp": cm[( -1, 1)],
        "fn": cm[(1, -1)],
    }


def summarize(path8: str, path70: str) -> Dict[str, Dict]:
    rows8 = {r["id"]: r for r in load_rows(path8)}
    rows70 = {r["id"]: r for r in load_rows(path70)}
    common_ids = set(rows8) & set(rows70)

    # preload verifier correctness vs gold where possible
    def verifier_correct(row: dict):
        gold = extract_gold(row)
        vpol = extract_verifier(row)
        if gold is None or vpol is None or vpol == 0:
            return None
        return gold == vpol

    stats = defaultdict(dict)
    for label, rows in [("8B", rows8), ("70B", rows70)]:
        total = len(rows)
        unknown = 0
        binary_used = 0
        binary_correct = 0
        preds = []
        golds = []
        verifiers = []
        agree_flags = []
        agree_on_cov = []
        agree_on_cov_commit = []
        conf_pairs = []
        ev_counts = []
        ev_counts_ge2 = 0
        unknown_on_cov = 0
        cov_total = 0
        model_correct_verifier_wrong = 0
        model_wrong_verifier_correct = 0
        for r in rows.values():
            pred_pol, is_bin, pred_label = extract_label(r)
            gold_pol = extract_gold(r)
            correct = extract_correct(r)
            v_pol = extract_verifier(r)
            cv = (r.get("metrics") or {}).get("claim_verification") or {}
            evc = int(cv.get("evidence_count", 0) or 0)
            ev_counts.append(evc)
            if evc >= 2:
                ev_counts_ge2 += 1

            if pred_label == "unknown":
                unknown += 1
            if is_bin:
                binary_used += 1
                if correct:
                    binary_correct += 1
            if pred_pol is not None and gold_pol is not None:
                preds.append(pred_pol)
                golds.append(gold_pol)
                conf_pairs.append((gold_pol, pred_pol))
            if pred_pol is not None and v_pol is not None:
                verifiers.append(v_pol)
                agree_flags.append(pred_pol == v_pol)
                if v_pol != 0:
                    agree_on_cov.append(pred_pol == v_pol)
                    cov_total += 1
                    if pred_label == "unknown":
                        unknown_on_cov += 1
                    else:
                        agree_on_cov_commit.append(pred_pol == v_pol)
                        vcorr = verifier_correct(r)
                        if vcorr is not None and gold_pol is not None and is_bin:
                            if correct and (vcorr is False):
                                model_correct_verifier_wrong += 1
                            if (not correct) and (vcorr is True):
                                model_wrong_verifier_correct += 1

        stats[label] = {
            "total": total,
            "unknown": unknown,
            "unknown_rate": unknown / total if total else 0.0,
            "binary_used": binary_used,
            "binary_acc": binary_correct / binary_used if binary_used else 0.0,
            "conf_pairs": conf_pairs,
            "preds": preds,
            "golds": golds,
            "verifiers": verifiers,
            "agree_rate": sum(agree_flags) / len(agree_flags) if agree_flags else 0.0,
            "agree_on_cov_rate": sum(agree_on_cov) / len(agree_on_cov) if agree_on_cov else 0.0,
            "agree_on_cov_n": len(agree_on_cov),
            "unknown_on_cov": unknown_on_cov,
            "cov_total": cov_total,
            "agree_on_cov_commit_rate": sum(agree_on_cov_commit) / len(agree_on_cov_commit) if agree_on_cov_commit else float("nan"),
            "agree_on_cov_commit_n": len(agree_on_cov_commit),
            "ev_counts": ev_counts,
            "ev_ge2_count": ev_counts_ge2,
            "model_correct_verifier_wrong": model_correct_verifier_wrong,
            "model_wrong_verifier_correct": model_wrong_verifier_correct,
        }

    # Paired analyses on common ids with binary labels
    bin_pairs = []
    agree_pairs = []
    for cid in common_ids:
        r8 = rows8[cid]
        r70 = rows70[cid]
        ml8 = (r8.get("metrics") or {}).get("model_label_metrics") or {}
        ml70 = (r70.get("metrics") or {}).get("model_label_metrics") or {}
        if not (ml8.get("is_binary") and ml70.get("is_binary")):
            continue
        c8 = bool(ml8.get("correct_binary"))
        c70 = bool(ml70.get("correct_binary"))
        bin_pairs.append((c8, c70))
        mv8 = (r8.get("metrics") or {}).get("model_vs_verifier") or {}
        mv70 = (r70.get("metrics") or {}).get("model_vs_verifier") or {}
        a8 = mv8.get("agree")
        a70 = mv70.get("agree")
        if a8 is not None and a70 is not None:
            agree_pairs.append((bool(a8), bool(a70)))

    # McNemar for binary correctness
    b01 = sum(1 for c8, c70 in bin_pairs if c8 and not c70)
    b10 = sum(1 for c8, c70 in bin_pairs if (not c8) and c70)
    stats["paired"] = {
        "bin_pairs_n": len(bin_pairs),
        "mcnemar_b01": b01,
        "mcnemar_b10": b10,
        "mcnemar_p": mcnemar(b01, b10),
        "bootstrap_delta": bootstrap_ci_delta([int(c8) for c8, _ in bin_pairs], [int(c70) for _, c70 in bin_pairs]),
        "agree_pairs_n": len(agree_pairs),
        "agree_better": sum(1 for a8, a70 in agree_pairs if (not a8) and a70),
        "agree_worse": sum(1 for a8, a70 in agree_pairs if a8 and (not a70)),
    }

    # Kappa
    stats["kappa"] = {
        "model_vs_gold_8B": cohen_kappa(stats["8B"]["preds"], stats["8B"]["golds"]),
        "model_vs_gold_70B": cohen_kappa(stats["70B"]["preds"], stats["70B"]["golds"]),
        "model_vs_verifier_8B": cohen_kappa(stats["8B"]["preds"], stats["8B"]["verifiers"]),
        "model_vs_verifier_70B": cohen_kappa(stats["70B"]["preds"], stats["70B"]["verifiers"]),
    }

    # Confusion matrices
    stats["confusion"] = {
        "8B": confusion(stats["8B"]["conf_pairs"]),
        "70B": confusion(stats["70B"]["conf_pairs"]),
    }

    # Evidence correlation buckets
    def bucket_summary(rows_dict):
        hi_correct = lo_correct = hi_total = lo_total = 0
        for r in rows_dict.values():
            ml = (r.get("metrics") or {}).get("model_label_metrics") or {}
            if not ml.get("is_binary"):
                continue
            correct = bool(ml.get("correct_binary"))
            evc = int((r.get("metrics") or {}).get("claim_verification", {}).get("evidence_count", 0) or 0)
            if evc >= 2:
                hi_total += 1
                hi_correct += int(correct)
            else:
                lo_total += 1
                lo_correct += int(correct)
        return {
            "hi_acc": hi_correct / hi_total if hi_total else float("nan"),
            "hi_n": hi_total,
            "lo_acc": lo_correct / lo_total if lo_total else float("nan"),
            "lo_n": lo_total,
        }

    stats["evidence_bucket"] = {
        "8B": bucket_summary(rows8),
        "70B": bucket_summary(rows70),
    }

    return stats


def to_md(stats: Dict, outpath: str):
    lines = []
    s8 = stats["8B"]
    s70 = stats["70B"]
    lines.append("## Model metrics (n=200)")
    lines.append(f"- 8B: unknown {s8['unknown_rate']*100:.1f}% ({s8['unknown']}/{s8['total']}), binary acc {s8['binary_acc']*100:.1f}% (n={s8['binary_used']}), agree_on_cov {s8['agree_on_cov_rate']*100:.1f}% (n={s8['agree_on_cov_n']}); unknown_on_cov {s8['unknown_on_cov']}/{s8['cov_total']}, agree_on_cov_given_commit {s8['agree_on_cov_commit_rate']*100:.1f}% (n={s8['agree_on_cov_commit_n']})")
    lines.append(f"- 70B: unknown {s70['unknown_rate']*100:.1f}% ({s70['unknown']}/{s70['total']}), binary acc {s70['binary_acc']*100:.1f}% (n={s70['binary_used']}), agree_on_cov {s70['agree_on_cov_rate']*100:.1f}% (n={s70['agree_on_cov_n']}); unknown_on_cov {s70['unknown_on_cov']}/{s70['cov_total']}, agree_on_cov_given_commit {s70['agree_on_cov_commit_rate']*100:.1f}% (n={s70['agree_on_cov_commit_n']})")
    lines.append("")
    lines.append("## Paired delta (70B vs 8B on same claims)")
    p = stats["paired"]
    med, lo, hi = p["bootstrap_delta"]
    lines.append(f"- Binary acc Δ (70B-8B): median {med*100:.1f} pp, 95% CI [{lo*100:.1f}, {hi*100:.1f}] (n={p['bin_pairs_n']}); McNemar b01={p['mcnemar_b01']}, b10={p['mcnemar_b10']}, p={p['mcnemar_p']:.3f}")
    lines.append(f"- Agree changes: better {p['agree_better']}, worse {p['agree_worse']} (n={p['agree_pairs_n']})")
    lines.append("")
    lines.append("## Kappa (chance-corrected agreement)")
    k = stats["kappa"]
    lines.append(f"- Model vs gold: 8B {k['model_vs_gold_8B']:.3f}, 70B {k['model_vs_gold_70B']:.3f}")
    lines.append(f"- Model vs verifier: 8B {k['model_vs_verifier_8B']:.3f}, 70B {k['model_vs_verifier_70B']:.3f}")
    lines.append("")
    lines.append("## Confusion (gold, pred) counts")
    conf = stats["confusion"]
    lines.append(f"- 8B: {conf['8B']}")
    lines.append(f"- 70B: {conf['70B']}")
    lines.append("")
    lines.append("## Accuracy by evidence bucket (evidence_count >=2 vs <2)")
    eb = stats["evidence_bucket"]
    lines.append(f"- 8B: hi {eb['8B']['hi_acc']*100:.1f}% (n={eb['8B']['hi_n']}), lo {eb['8B']['lo_acc']*100:.1f}% (n={eb['8B']['lo_n']})")
    lines.append(f"- 70B: hi {eb['70B']['hi_acc']*100:.1f}% (n={eb['70B']['hi_n']}), lo {eb['70B']['lo_acc']*100:.1f}% (n={eb['70B']['lo_n']})")
    lines.append("")
    lines.append("## Model vs verifier errors on covered (model commits)")
    lines.append(f"- 8B: model_correct_verifier_wrong {s8['model_correct_verifier_wrong']}, model_wrong_verifier_correct {s8['model_wrong_verifier_correct']}")
    lines.append(f"- 70B: model_correct_verifier_wrong {s70['model_correct_verifier_wrong']}, model_wrong_verifier_correct {s70['model_wrong_verifier_correct']}")
    with open(outpath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval8", required=True)
    ap.add_argument("--eval70", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    stats = summarize(args.eval8, args.eval70)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    to_md(stats, args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
