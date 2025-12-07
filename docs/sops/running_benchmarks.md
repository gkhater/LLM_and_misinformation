# SOP: Running Benchmarks

1) **Activate venv**
```powershell
.\\.venv\\Scripts\\activate
```

2) **Pick backend config**
- Groq 70B hosted (`llama3-3-70b-versatile`): `config/model_llama70b_groq.yaml`
- Self-hosted vLLM: `config/model_llama8b_vllm.yaml`
- Local HF pipeline: `config/model_llama8b_hf.yaml`

3) **Optional dataset split**
- Set `dataset.split_filter` in `config/base.yaml` (`eval`, `test`, or `null` for all).

4) **Run**
```powershell
python -m src.cli --model-config config/model_llama70b_groq.yaml
```
Add `--max-rows 5` for quick smoke tests.

5) **Outputs**
- JSONL saved under `outputs/` with backend/model in the filename.
