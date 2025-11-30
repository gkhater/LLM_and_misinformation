import argparse
import yaml
from dotenv import load_dotenv

from src.runner import run_model_on_dataset
from src.models.llama_groq import GroqLlamaClassifier
from src.models.llama_vllm import VLLMLlamaClassifier
from src.models.llama_hf import HFLlamaClassifier


def load_config(base_path, model_path):
    with open(base_path, "r", encoding="utf-8") as f:
        base = yaml.safe_load(f)
    with open(model_path, "r", encoding="utf-8") as f:
        model = yaml.safe_load(f)
    base["model"] = model["model"]
    return base


def build_classifier(config):
    m = config["model"]
    backend = m["backend"]
    name = m["name"]

    if backend == "groq":
        return GroqLlamaClassifier(name, m["groq"]["api_key_env"])
    if backend == "vllm":
        v = m["vllm"]
        return VLLMLlamaClassifier(name, v["base_url"], v.get("api_key", ""))
    if backend == "hf":
        h = m["hf"]
        return HFLlamaClassifier(
            name,
            h.get("device_map", "auto"),
            h.get("dtype", "auto"),
            h.get("max_new_tokens", 512),
        )
    raise ValueError(f"Unknown backend {backend}")


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run misinformation benchmark.")
    parser.add_argument("--base-config", default="config/base.yaml")
    parser.add_argument(
        "--model-config",
        default="config/model_llama70b_groq.yaml",
        help="Per-backend model config. Defaults to Groq 70B.",
    )
    parser.add_argument("--max-rows", type=int, default=None, help="Optional cap for quick smoke tests.")
    args = parser.parse_args()

    config = load_config(args.base_config, args.model_config)
    classifier = build_classifier(config)

    out_path = run_model_on_dataset(config, classifier, max_rows=args.max_rows)
    print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()
