#!/usr/bin/env python3
"""
Generate Timeloop workloads for all target LLM and ViT models.

Generates prefill and decode phase workloads at multiple sequence/cache lengths
for LLaMA 3.1 (8B, 70B) and Qwen3 MoE (235B-A22B, 30B-A3B).
Also generates ViT workloads (ViT-B/16, ViT-L/16, ViT-H/14).

Usage:
    conda run -n mozart python generate_llm_workloads.py
"""
import json
import os
import yaml
from pytorch2timeloop.convert_llm import convert_llm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIGS = os.path.join(SCRIPT_DIR, "test", "configs")
SAVE_DIR = os.path.join(SCRIPT_DIR, "llm_workloads")

MODELS = [
    ("llama3.1_8b", os.path.join(CONFIGS, "llama3.1_8b", "config.json")),
    ("llama3.1_70b", os.path.join(CONFIGS, "llama3.1_70b", "config.json")),
    ("qwen3_235b_a22b", os.path.join(CONFIGS, "qwen3_235b_a22b", "config.json")),
    ("qwen3_30b_a3b", os.path.join(CONFIGS, "qwen3_30b_a3b", "config.json")),
]

PREFILL_SEQ_LENS = [512, 1024, 2048, 4096]
DECODE_KV_LENS = [512, 1024, 2048, 4096]

# ViT models: (name, config_path, seq_len)
# seq_len = (image_size / patch_size)^2 + 1 (class token)
VIT_MODELS = [
    ("vit_b16", os.path.join(CONFIGS, "vit_b16", "config.json"), 197),   # 224/16=14, 14^2+1=197
    ("vit_l16", os.path.join(CONFIGS, "vit_l16", "config.json"), 197),
    ("vit_h14", os.path.join(CONFIGS, "vit_h14", "config.json"), 257),   # 224/14=16, 16^2+1=257
]


def _write_vit_network_yaml(config_path, model_name, seq_len, outdir):
    """Overwrite NETWORK.yaml with ViT-specific description.

    ViT differs from LLM in:
    - MLP has 2 ops (fc1+fc2 with GELU), not 3 (gate+up+down SwiGLU).
      gate_proj.yaml == fc1, down_proj.yaml == fc2, no up_proj.
    - Standard MHA (no GQA): q_proj == k_proj == v_proj in dimensions.
    - cls_head instead of lm_head (same YAML, just different semantics).
    - No prefill/decode distinction.
    """
    with open(config_path) as f:
        cfg = json.load(f)

    H = cfg['hidden_size']
    n_h = cfg['num_attention_heads']
    d = H // n_h
    I = cfg['intermediate_size']
    L = cfg['num_hidden_layers']
    num_classes = cfg['vocab_size']

    # 12 ops per layer: 10 attention (q/k/v proj + qk + 4 softmax + v + o_proj) + 2 MLP (fc1 + fc2)
    ops_per_layer = 12
    total_ops = ops_per_layer * L + 1  # + cls_head

    desc = {
        'network': {
            'model_name': model_name,
            'model_type': 'Vision Transformer (ViT)',
            'batch_size': 1,
            'seq_len': f'{seq_len} (num_patches + 1 class token)',
            'attention': f'Q@K^T: ({seq_len} x {seq_len})',
        },
        'architecture': {
            'hidden_size': H,
            'num_attention_heads': n_h,
            'head_dim': d,
            'mlp_dim': I,
            'num_layers': L,
            'num_classes': num_classes,
            'attention_type': 'standard MHA (no GQA)',
            'mlp_type': 'GELU (fc1 + fc2), not SwiGLU',
        },
        'structure': {
            'description': (
                f"Only unique operator YAMLs are stored. "
                f"k_proj and v_proj have the same dimensions as q_proj (standard MHA). "
                f"gate_proj.yaml is fc1, down_proj.yaml is fc2 (no up_proj in ViT GELU MLP). "
                f"lm_head.yaml is the classification head. "
                f"All {L} encoder layers are identical."
            ),
            'unique_yamls': [
                'layer0_q_proj',
                'layer0_attn_qk',
                'layer0_softmax_max', 'layer0_softmax_sub_exp',
                'layer0_softmax_sum', 'layer0_softmax_div',
                'layer0_attn_v',
                'layer0_gate_proj', 'layer0_down_proj',
                'lm_head',
            ],
            'total_operators_in_model': total_ops,
            'operators_per_layer': ops_per_layer,
            'num_layers': L,
            'layer_operator_sequence': [
                {'yaml': 'layer0_q_proj.yaml', 'count': 1,
                 'description': f'Query projection: (B*S, {H}) @ ({H}, {n_h * d})'},
                {'yaml': 'layer0_q_proj.yaml', 'count': 1, 'alias': 'k_proj',
                 'description': 'Key projection: same dims as q_proj (standard MHA)'},
                {'yaml': 'layer0_q_proj.yaml', 'count': 1, 'alias': 'v_proj',
                 'description': 'Value projection: same dims as q_proj (standard MHA)'},
                {'yaml': 'layer0_attn_qk.yaml', 'count': 1,
                 'description': f'Attention scores: Q @ K^T, batched over {n_h} heads'},
                {'yaml': 'layer0_softmax_max.yaml', 'count': 1,
                 'description': 'Softmax step 1: row-wise max'},
                {'yaml': 'layer0_softmax_sub_exp.yaml', 'count': 1,
                 'description': 'Softmax step 2: subtract max and exponentiate'},
                {'yaml': 'layer0_softmax_sum.yaml', 'count': 1,
                 'description': 'Softmax step 3: row-wise sum'},
                {'yaml': 'layer0_softmax_div.yaml', 'count': 1,
                 'description': 'Softmax step 4: divide by sum'},
                {'yaml': 'layer0_attn_v.yaml', 'count': 1,
                 'description': 'Attention context: A @ V'},
                {'yaml': 'layer0_q_proj.yaml', 'count': 1, 'alias': 'o_proj',
                 'description': f'Output projection: same dims as q_proj ({n_h * d}) -> ({H})'},
                {'yaml': 'layer0_gate_proj.yaml', 'count': 1,
                 'description': f'MLP fc1: ({H}) -> ({I}) with GELU'},
                {'yaml': 'layer0_down_proj.yaml', 'count': 1,
                 'description': f'MLP fc2: ({I}) -> ({H})'},
            ],
            'global_operators': [
                {'yaml': 'lm_head.yaml', 'count': 1,
                 'description': f'Classification head: ({H}) -> ({num_classes})'},
            ],
        },
    }

    fpath = os.path.join(outdir, 'NETWORK.yaml')
    with open(fpath, 'w') as f:
        yaml.dump(desc, f, default_flow_style=False, sort_keys=False)
    print(f"  -> Overwrote {fpath} with ViT network description")


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

    # ViT models: reuse convert_llm (same transformer block), then fix NETWORK.yaml
    for model_name, config_path, seq_len in VIT_MODELS:
        if not os.path.exists(config_path):
            print(f"SKIP: {model_name} (config not found at {config_path})")
            continue

        print(f"\n{'#' * 60}")
        print(f"# {model_name} (ViT, seq_len={seq_len})")
        print(f"{'#' * 60}")

        vit_dir = f"{model_name}_s{seq_len}"
        convert_llm(config_path=config_path, save_dir=SAVE_DIR,
                     model_name=model_name, batch_size=1, seq_len=seq_len,
                     phase='prefill', output_name=vit_dir)

        # Overwrite NETWORK.yaml with ViT-specific description
        outdir = os.path.join(SAVE_DIR, vit_dir)
        _write_vit_network_yaml(config_path, model_name, seq_len, outdir)

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
