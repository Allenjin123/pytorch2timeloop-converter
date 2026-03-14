#!/usr/bin/env python3
"""
Generate Timeloop workloads for all target LLM models.

Generates prefill and decode phase workloads at multiple sequence/cache lengths
for LLaMA 3.1 (8B, 70B) and Qwen3 MoE (235B-A22B, 30B-A3B).

Usage:
    conda run -n mozart python generate_llm_workloads.py
"""
import os
from pytorch2timeloop.convert_llm import convert_llm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIGS = os.path.join(SCRIPT_DIR, "test", "configs")
SAVE_DIR = os.path.join(SCRIPT_DIR, "llm_workloads")

# LLaMA 3.1 70B config is in the parent Mozart directory
LLAMA_70B_CONFIG = os.path.join(os.path.dirname(SCRIPT_DIR),
                                 "Llama-3.1-70B-Instruct", "config.json")

MODELS = [
    ("llama3.1_8b", os.path.join(CONFIGS, "llama3.1_8b", "config.json")),
    ("llama3.1_70b", LLAMA_70B_CONFIG),
    ("qwen3_235b_a22b", os.path.join(CONFIGS, "qwen3_235b_a22b", "config.json")),
    ("qwen3_30b_a3b", os.path.join(CONFIGS, "qwen3_30b_a3b", "config.json")),
]

PREFILL_SEQ_LENS = [512, 1024, 2048, 4096]
DECODE_KV_LENS = [512, 1024, 2048, 4096]


def main():
    for model_name, config_path in MODELS:
        if not os.path.exists(config_path):
            print(f"SKIP: {model_name} (config not found at {config_path})")
            continue

        print(f"\n{'#' * 60}")
        print(f"# {model_name}")
        print(f"{'#' * 60}")

        for s in PREFILL_SEQ_LENS:
            convert_llm(config_path=config_path, save_dir=SAVE_DIR,
                        model_name=model_name, batch_size=1, seq_len=s,
                        phase='prefill')

        for kv in DECODE_KV_LENS:
            convert_llm(config_path=config_path, save_dir=SAVE_DIR,
                        model_name=model_name, batch_size=1, seq_len=1,
                        phase='decode', kv_cache_len=kv)

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    total = 0
    for d in sorted(os.listdir(SAVE_DIR)):
        full = os.path.join(SAVE_DIR, d)
        if os.path.isdir(full):
            count = len(os.listdir(full))
            total += count
            print(f"  {d:50s} {count:>4} files")
    print(f"\n  TOTAL: {total} files")


if __name__ == '__main__':
    main()
