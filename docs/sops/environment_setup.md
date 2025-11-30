# SOP: Environment Setup

1) **Create venv**
```powershell
python -m venv .venv
.\\.venv\\Scripts\\activate
```

2) **Install dependencies**
```powershell
pip install -r requirements.txt
```

3) **Configure secrets**
- Copy `.env.example` to `.env`.
- Set `GROQ_API_KEY` (required for Groq backend).
- Optional: set `VLLM_API_KEY` if your vLLM endpoint enforces auth.

4) **Sanity check**
```powershell
python -m compileall src
```
