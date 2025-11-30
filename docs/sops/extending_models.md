# SOP: Adding a New Backend/Model

1) **Decide backend type**
- Hosted OpenAI-compatible → mirror `src/models/llama_vllm.py`
- Local HF pipeline → mirror `src/models/llama_hf.py`

2) **Implement classifier**
- Add a new class in `src/models/` inheriting `BaseClassifier`.
- Ensure `classify` returns the parsed dict from `parse_json_output` plus metadata (`raw_output`, `model_name`, `backend`, `latency_ms`).

3) **Create model config**
- Add `config/model_<name>.yaml` with `model.name`, `model.backend`, and backend-specific keys.

4) **Run smoke test**
```powershell
python -m src.cli --model-config config/model_<name>.yaml --max-rows 2
```

5) **Document any deviations**
- Append rationale to `decision.md` if decoding/prompting must change.
