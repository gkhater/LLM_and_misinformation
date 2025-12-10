import json, os
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def read_jsonl(path):
    rows=[]
    with open(path,'r',encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def summarize(rows):
    verdicts=Counter(); coverage=0; n=0; lc_used=lc_success=0; bin_used=bin_success=0; ev=[]
    for r in rows:
        n+=1
        m=r.get('metrics') or {}
        cv=m.get('claim_verification') or {}
        lc=m.get('label_consistency') or {}
        v=cv.get('verdict','nei'); verdicts[v]+=1
        if v!='nei': coverage+=1
        ev.append(int(cv.get('evidence_count',0) or 0))
        if 'success' in lc:
            lc_used+=1
            if lc.get('success'): lc_success+=1
        tp=lc.get('target_polarity',0); pp=lc.get('predicted_polarity',0)
        if tp in (-1,1) and pp in (-1,1):
            bin_used+=1
            if tp==pp: bin_success+=1
    return {
        'coverage': coverage/max(1,n),
        'verdicts': verdicts,
        'lc_acc': (lc_success/max(1,lc_used)) if lc_used else None,
        'binary_acc': (bin_success/max(1,bin_used)) if bin_used else None,
        'binary_n': bin_used,
        'ev': ev,
    }

def bar_two(title, labels, vals, outpath):
    plt.figure(); plt.bar(labels, vals); plt.title(title); plt.tight_layout(); plt.savefig(outpath, dpi=200); plt.close()

def stacked_verdict(title, labels, verdict_counts_list, outpath):
    order=['entail','contradict','nei']; bottoms=[0]*len(labels); plt.figure()
    for v in order:
        heights=[vc.get(v,0) for vc in verdict_counts_list]
        plt.bar(labels, heights, bottom=bottoms, label=v)
        bottoms=[b+h for b,h in zip(bottoms, heights)]
    plt.title(title); plt.legend(); plt.tight_layout(); plt.savefig(outpath, dpi=200); plt.close()

def hist_counts(title, data_a, data_b, labels, outpath):
    plt.figure(); plt.hist([data_a, data_b], bins=10, label=labels); plt.title(title); plt.legend(); plt.tight_layout(); plt.savefig(outpath, dpi=200); plt.close()

def run(eval8, eval70, outdir, tag):
    os.makedirs(outdir, exist_ok=True)
    s8=summarize(read_jsonl(eval8)); s70=summarize(read_jsonl(eval70))
    bar_two(f"Coverage (non-NEI) {tag}", ['8B','70B'], [s8['coverage'], s70['coverage']], os.path.join(outdir, f"coverage_{tag}.png"))
    bar_two(f"Binary acc {tag} (n8={s8['binary_n']}, n70={s70['binary_n']})", ['8B','70B'], [s8['binary_acc'] or 0.0, s70['binary_acc'] or 0.0], os.path.join(outdir, f"binary_acc_{tag}.png"))
    stacked_verdict(f"Verdict mix {tag}", ['8B','70B'], [s8['verdicts'], s70['verdicts']], os.path.join(outdir, f"verdict_mix_{tag}.png"))
    hist_counts(f"Evidence count {tag}", s8['ev'], s70['ev'], ['8B','70B'], os.path.join(outdir, f"evidence_hist_{tag}.png"))
    print(tag, '8B', s8)
    print(tag, '70B', s70)

run('outputs/eval_8b_wiki853_200.jsonl','outputs/eval_70b_wiki853_200.jsonl','outputs/plots','wiki853_200')
run('outputs/eval_8b_fever_200.jsonl','outputs/eval_70b_fever_200.jsonl','outputs/plots','fever50k_200')
