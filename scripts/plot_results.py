import json, argparse, os
from collections import Counter, defaultdict
import matplotlib.pyplot as plt

def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: 
                continue
            rows.append(json.loads(line))
    return rows

def summarize(rows):
    verdicts = []
    coverage = 0
    n = 0
    lc_used = 0
    lc_success = 0
    bin_used = 0
    bin_success = 0
    evidence_counts = []

    for r in rows:
        n += 1
        m = (r.get("metrics") or {})
        cv = (m.get("claim_verification") or {})
        lc = (m.get("label_consistency") or {})

        v = cv.get("verdict", "nei")
        verdicts.append(v)
        if v != "nei":
            coverage += 1

        evidence_counts.append(int(cv.get("evidence_count", 0) or 0))

        # label_consistency success if present
        if "success" in lc:
            lc_used += 1
            if lc["success"]:
                lc_success += 1

        # binary accuracy: only when target_polarity is +/-1 and predicted_polarity is +/-1
        tp = lc.get("target_polarity", 0)
        pp = lc.get("predicted_polarity", 0)
        if tp in (-1, 1) and pp in (-1, 1):
            bin_used += 1
            if tp == pp:
                bin_success += 1

    return {
        "n": n,
        "coverage_rate": coverage / max(1, n),
        "verdict_counts": Counter(verdicts),
        "lc_acc": (lc_success / max(1, lc_used)) if lc_used else None,
        "binary_acc": (bin_success / max(1, bin_used)) if bin_used else None,
        "binary_n": bin_used,
        "evidence_counts": evidence_counts,
    }

def bar_two(title, labels, vals, outpath):
    plt.figure()
    plt.bar(labels, vals)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def stacked_verdict(title, labels, verdict_counts_list, outpath):
    # verdicts in fixed order
    order = ["entail", "contradict", "nei"]
    bottoms = [0]*len(labels)

    plt.figure()
    for v in order:
        heights = [vc.get(v, 0) for vc in verdict_counts_list]
        plt.bar(labels, heights, bottom=bottoms, label=v)
        bottoms = [b+h for b,h in zip(bottoms, heights)]
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def hist_counts(title, data_a, data_b, labels, outpath):
    plt.figure()
    plt.hist([data_a, data_b], bins=10, label=labels)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval8", required=True)
    ap.add_argument("--eval70", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--tag", required=True)  # e.g. wiki853_200 or fever_200
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    r8 = read_jsonl(args.eval8)
    r70 = read_jsonl(args.eval70)

    s8 = summarize(r8)
    s70 = summarize(r70)

    # coverage
    bar_two(f"Coverage (non-NEI) {args.tag}",
            ["8B","70B"], [s8["coverage_rate"], s70["coverage_rate"]],
            os.path.join(args.outdir, f"coverage_{args.tag}.png"))

    # binary acc
    bar_two(f"Binary accuracy {args.tag} (n_used: {s8['binary_n']}/{s70['binary_n']})",
            ["8B","70B"], [s8["binary_acc"] or 0.0, s70["binary_acc"] or 0.0],
            os.path.join(args.outdir, f"binary_acc_{args.tag}.png"))

    # verdict mix
    stacked_verdict(f"Verdict mix {args.tag}",
                    ["8B","70B"], [s8["verdict_counts"], s70["verdict_counts"]],
                    os.path.join(args.outdir, f"verdict_mix_{args.tag}.png"))

    # evidence count hist
    hist_counts(f"Evidence count distribution {args.tag}",
                s8["evidence_counts"], s70["evidence_counts"],
                ["8B","70B"],
                os.path.join(args.outdir, f"evidence_hist_{args.tag}.png"))

    print("DONE")
    print("8B:", s8)
    print("70B:", s70)
