import json
paths=['outputs/gen_results_groq_llama-3_1-8b-instant_20251210T194051.jsonl','outputs/gen_results_groq_llama-3_3-70b-versatile_20251210T195219.jsonl']
for p in paths:
    with open(p,encoding='utf-8') as f:
        line=next(f)
    row=json.loads(line)
    print('\n',p)
    print('keys', row.keys())
    print('model_output sample:', json.dumps(row.get('model_output',{}), indent=2)[:500])
