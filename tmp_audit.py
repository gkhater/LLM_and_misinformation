import csv, json, os
from collections import Counter

audit_dir = 'outputs/audit'
os.makedirs(audit_dir, exist_ok=True)
claims_path='data/liar_slice_200.csv'
gen8_path='outputs/gen_results_groq_llama-3_1-8b-instant_20251210T194051.jsonl'
gen70_path='outputs/gen_results_groq_llama-3_3-70b-versatile_20251210T195219.jsonl'

def load_claim_labels(path):
    labels={}
    with open(path, newline='', encoding='utf-8') as f:
        reader=csv.DictReader(f)
        for row in reader:
            try:
                cid=int(row['id'])
            except Exception:
                continue
            labels[cid]=row.get('label')
    return labels

def norm_label(lbl):
    return lbl.strip().lower() if lbl else None

def scan_gen(path, gold):
    missing=0
    total=0
    eq=0; eq_used=0
    pred_counts=Counter()
    examples_missing=[]
    with open(path, encoding='utf-8') as f:
        for line in f:
            total+=1
            try:
                row=json.loads(line)
            except Exception:
                missing+=1
                continue
            cid=row.get('id')
            mo=row.get('model_output') or {}
            pred=norm_label(mo.get('label'))
            if not pred:
                missing+=1
                if len(examples_missing)<3:
                    examples_missing.append(mo)
            else:
                pred_counts[pred]+=1
            gold_lbl=norm_label(gold.get(cid))
            if pred is not None and gold_lbl is not None:
                eq_used+=1
                if pred==gold_lbl:
                    eq+=1
    return {
        'path': path,
        'total': total,
        'pred_missing': missing,
        'pred_present': total-missing,
        'pred_equals_gold_rate': (eq/eq_used) if eq_used else None,
        'eq_used': eq_used,
        'pred_counts': pred_counts.most_common(),
        'examples_missing': examples_missing,
    }

gold=load_claim_labels(claims_path)
report={
    'gen8': scan_gen(gen8_path, gold),
    'gen70': scan_gen(gen70_path, gold),
}
with open(os.path.join(audit_dir,'generation_audit.json'),'w',encoding='utf-8') as f:
    json.dump(report,f,indent=2)
print(json.dumps(report,indent=2))
