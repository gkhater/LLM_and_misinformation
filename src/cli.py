import os
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

import argparse
import yaml
from dotenv import load_dotenv

from src.runner import run_model_on_dataset
from src.models.llama_groq import GroqLlamaClassifier
from src.models.llama_vllm import VLLMLlamaClassifier
from src.models.llama_hf import HFLlamaClassifier
from src.pipeline.generation import run_generation
from src.pipeline.evaluation import run_evaluation


def load_config(base_path, model_path):
    """Load base + model configs; allow model config to override dataset block."""
    with open(base_path, "r", encoding="utf-8") as f:
        base = yaml.safe_load(f)
    with open(model_path, "r", encoding="utf-8") as f:
        model = yaml.safe_load(f)
    base["model"] = model["model"]
    # Optional dataset override inside model config for dataset swaps without CLI flags.
    if "dataset" in model:
        base["dataset"] = {**base.get("dataset", {}), **model["dataset"]}
    return base


def build_classifier(config):
    m = config["model"]
    backend = m["backend"]
    name = m["name"]
    gen = config.get("generation", {})
    temperature = gen.get("temperature", 0.0)
    top_p = gen.get("top_p", 1.0)
    max_tokens = gen.get("max_tokens", 512)

    if backend == "groq":
        return GroqLlamaClassifier(
            name,
            m["groq"]["api_key_env"],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
    if backend == "vllm":
        v = m["vllm"]
        return VLLMLlamaClassifier(
            name,
            v["base_url"],
            v.get("api_key", ""),
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
    if backend == "hf":
        h = m["hf"]
        return HFLlamaClassifier(
            name,
            h.get("device_map", "auto"),
            h.get("dtype", "auto"),
            h.get("max_new_tokens", 512),
            temperature=temperature,
            top_p=top_p,
        )
    raise ValueError(f"Unknown backend {backend}")


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run misinformation benchmark.")
    parser.add_argument("--mode", choices=["generation", "evaluation", "legacy"], default="generation")
    parser.add_argument("--base-config", default="config/base.yaml")
    parser.add_argument(
        "--model-config",
        default="config/model_llama70b_groq.yaml",
        help="Per-backend model config (generation/legacy). Defaults to Groq 70B.",
    )
    parser.add_argument(
        "--eval-config",
        default=None,
        help="Evaluation config (used when --mode evaluation).",
    )
    parser.add_argument("--max-rows", type=int, default=None, help="Optional cap for quick smoke tests.")
    parser.add_argument("--smoke", action="store_true", help="Smoke mode for evaluation: health checks only.")
    parser.add_argument("--debug-ids", default=None, help="Comma-separated claim ids to inspect during evaluation.")
    args = parser.parse_args()

    if args.mode == "evaluation":
        if not args.eval_config:
            parser.error("--eval-config is required when mode=evaluation")
        debug_ids = [int(x) for x in args.debug_ids.split(",")] if args.debug_ids else None
        out_path = run_evaluation(args.eval_config, max_rows=args.max_rows, smoke=args.smoke, debug_ids=debug_ids)
        print(f"Evaluation saved to: {out_path}")
        return

    config = load_config(args.base_config, args.model_config)

    classifier = build_classifier(config)

    if args.mode == "legacy":
        out_path = run_model_on_dataset(config, classifier, max_rows=args.max_rows)
    else:
        out_path = run_generation(config, classifier, max_rows=args.max_rows)
    print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()
