# SOP: Running Benchmarks

1) **Activate venv**
```powershell
.\\.venv\\Scripts\\activate
```

2) **Two-phase flow (recommended)**
- **Generation only:** choose backend config (Groq/vLLM/HF), run:
  ```powershell
  python -m src.cli --mode generation --model-config config/model_llama70b_groq.yaml --max-rows 5
  ```
  Writes `outputs/gen_*.jsonl`.
- **Evaluation (offline):** point to the generation file and eval config:
  ```powershell
  python -m src.cli --mode evaluation --eval-config config/eval_liar.yaml
  ```
  Runs retrieval+NLI once per claim and computes metrics, no Groq calls.

3) **Optional dataset split**
- Set `dataset.split_filter` in `config/base.yaml` (`eval`, `test`, or `null` for all).

4) **Outputs**
- Generation JSONL under `outputs/` (no metrics).
- Evaluation JSONL under `outputs/` with metrics merged in; NLI cache stored separately for reuse.
