#!/usr/bin/env python3
"""
Generate DAG representations for neural networks.

Produces DAG.yaml files alongside existing Timeloop operator YAMLs,
capturing the true computation graph structure (parallel branches,
skip connections, residual adds) that Mozart needs for DAG-aware
pipeline stage mapping.

Supported networks:
  - LLaMA 3.1 (8B, 70B) — dense decoder, SwiGLU MLP
  - Qwen3 MoE (235B-A22B, 30B-A3B) — MoE decoder
  - MobileNetV3-Small — inverted residuals + squeeze-excitation
  - RepLKNet-31B — large kernel (31×31) + small kernel parallel branches

Usage:
    python convert_dag.py                      # all networks
    python convert_dag.py --network llama      # just LLMs
    python convert_dag.py --network mobilenet  # just MobileNet
    python convert_dag.py --network replknet   # just RepLKNet
"""

import argparse
import os
import sys
import json
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fx as fx
import torchvision.models as models

from pytorch2timeloop.dag import (
    NetworkDAG, DAGNode, DAGEdge,
    build_full_llm_dag,
)
from pytorch2timeloop.convert_llm import LLMConfig

logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIGS_DIR = os.path.join(SCRIPT_DIR, "test", "configs")
DAG_OUT_DIR = os.path.join(SCRIPT_DIR, "dag_workloads")


# ============================================================
# LLM DAGs
# ============================================================

def generate_llm_dags():
    """Generate DAGs for LLaMA and Qwen models."""
    models_list = [
        ("llama3.1_8b", os.path.join(CONFIGS_DIR, "llama3.1_8b", "config.json")),
        ("llama3.1_70b", os.path.join(CONFIGS_DIR, "llama3.1_70b", "config.json")),
        ("qwen3_235b_a22b", os.path.join(CONFIGS_DIR, "qwen3_235b_a22b", "config.json")),
        ("qwen3_30b_a3b", os.path.join(CONFIGS_DIR, "qwen3_30b_a3b", "config.json")),
    ]

    seq_lens = [2048]
    batch_size = 1

    for model_name, config_path in models_list:
        if not os.path.exists(config_path):
            print(f"SKIP: {model_name} (config not found at {config_path})")
            continue

        cfg = LLMConfig.from_json(config_path, model_name)

        for seq_len in seq_lens:
            # Prefill DAG
            dag = build_full_llm_dag(cfg, batch_size, seq_len,
                                     kv_cache_len=None)
            outdir = os.path.join(DAG_OUT_DIR, f"{model_name}_prefill_s{seq_len}")
            os.makedirs(outdir, exist_ok=True)
            dag_path = os.path.join(outdir, "DAG.yaml")
            dag.save_yaml(dag_path)
            _print_dag_summary(dag, dag_path)

            # Decode DAG
            dag = build_full_llm_dag(cfg, batch_size, seq_len=1,
                                     kv_cache_len=seq_len)
            outdir = os.path.join(DAG_OUT_DIR, f"{model_name}_decode_kv{seq_len}")
            os.makedirs(outdir, exist_ok=True)
            dag_path = os.path.join(outdir, "DAG.yaml")
            dag.save_yaml(dag_path)
            _print_dag_summary(dag, dag_path)


# ============================================================
# MobileNetV3 DAG (manually constructed from known architecture)
# ============================================================

def _build_mobilenetv3_small_dag(batch_size: int = 1) -> NetworkDAG:
    """
    Build DAG for MobileNetV3-Small.

    Architecture: stem → 11 inverted residual blocks → final conv → classifier

    Each inverted residual block:
        input
        └→ expand_conv (1×1, if expand_ratio != 1)
           └→ depthwise_conv (3×3 or 5×5)
              └→ [SE: avgpool → fc1 → fc2 → scale]  (optional)
                 └→ project_conv (1×1)
                    └→ [+ input]  (skip if stride==1 and in==out channels)
                       └→ output

    SE (squeeze-excitation) branches create a DAG fork-join:
        depthwise_out
        ├→ se_avgpool → se_fc1 → se_fc2 → se_scale
        └→ ────────────────────────────→ multiply ← se_scale
    """
    # MobileNetV3-Small configuration from torchvision
    # (expand, channels, kernel, stride, use_se, activation)
    inverted_residual_configs = [
        # expand, out_ch, kernel, stride, se, act
        (1,    16,  3, 2, True,  'RE'),   # block 0
        (72/16, 24,  3, 2, False, 'RE'),  # block 1
        (88/24, 24,  3, 1, False, 'RE'),  # block 2
        (4,    40,  5, 2, True,  'HS'),   # block 3
        (6,    40,  5, 1, True,  'HS'),   # block 4
        (6,    40,  5, 1, True,  'HS'),   # block 5
        (3,    48,  5, 1, True,  'HS'),   # block 6
        (3,    48,  5, 1, True,  'HS'),   # block 7
        (6,    96,  5, 2, True,  'HS'),   # block 8
        (6,    96,  5, 1, True,  'HS'),   # block 9
        (6,    96,  5, 1, True,  'HS'),   # block 10
    ]

    dag = NetworkDAG(model_name='mobilenet_v3_small')
    N = batch_size
    H, W = 224, 224

    # Stem: Conv2d(3, 16, 3, stride=2)
    stem_id = "stem_conv"
    C_in, C_out = 3, 16
    H_out, W_out = H // 2, W // 2
    dag.add_node(DAGNode(stem_id, 'conv2d', 'layer1_features_0_0.yaml',
                         {'N': N, 'C': C_in, 'M': C_out, 'R': 3, 'S': 3,
                          'P': H_out, 'Q': W_out}))

    prev_id = stem_id
    C_prev = C_out
    layer_counter = 2  # layer1 is stem
    H_cur, W_cur = H_out, W_out

    for blk_idx, (expand_ratio, out_ch, kernel, stride, use_se, _) in enumerate(inverted_residual_configs):
        in_ch = C_prev
        mid_ch = int(round(in_ch * expand_ratio))
        has_skip = (stride == 1 and in_ch == out_ch)

        block_input = prev_id

        # Expand conv (1×1) — only if expand_ratio != 1
        if mid_ch != in_ch:
            exp_id = f"block{blk_idx}_expand"
            dag.add_node(DAGNode(exp_id, 'pointwise_conv2d',
                                 f'layer{layer_counter}_features_{blk_idx + 1}_block_0_0.yaml',
                                 {'N': N, 'C': in_ch, 'M': mid_ch,
                                  'R': 1, 'S': 1, 'P': H_cur, 'Q': W_cur}))
            dag.add_edge(prev_id, exp_id, 'expand_input',
                         [N, in_ch, H_cur, W_cur])
            prev_id = exp_id
            layer_counter += 1

        # Depthwise conv
        H_dw = H_cur // stride
        W_dw = W_cur // stride
        dw_id = f"block{blk_idx}_depthwise"
        dag.add_node(DAGNode(dw_id, 'depthwise_conv2d',
                             f'layer{layer_counter}_features_{blk_idx + 1}_block_1_0.yaml',
                             {'N': N, 'C': mid_ch, 'M': mid_ch, 'G': mid_ch,
                              'R': kernel, 'S': kernel,
                              'P': H_dw, 'Q': W_dw}))
        dag.add_edge(prev_id, dw_id, 'dw_input', [N, mid_ch, H_cur, W_cur])
        prev_id = dw_id
        layer_counter += 1
        H_cur, W_cur = H_dw, W_dw

        # Squeeze-Excitation (if present)
        if use_se:
            se_fc1_id = f"block{blk_idx}_se_fc1"
            se_fc2_id = f"block{blk_idx}_se_fc2"
            se_mul_id = f"block{blk_idx}_se_mul"
            se_reduced = max(1, mid_ch // 4)

            dag.add_node(DAGNode(se_fc1_id, 'pointwise_conv2d',
                                 f'layer{layer_counter}_features_{blk_idx + 1}_block_2_fc1.yaml',
                                 {'N': N, 'C': mid_ch, 'M': se_reduced,
                                  'R': 1, 'S': 1, 'P': 1, 'Q': 1}))
            layer_counter += 1

            dag.add_node(DAGNode(se_fc2_id, 'pointwise_conv2d',
                                 f'layer{layer_counter}_features_{blk_idx + 1}_block_2_fc2.yaml',
                                 {'N': N, 'C': se_reduced, 'M': mid_ch,
                                  'R': 1, 'S': 1, 'P': 1, 'Q': 1}))
            layer_counter += 1

            dag.add_node(DAGNode(se_mul_id, 'elementwise_mul', '',
                                 {'N': N, 'C': mid_ch, 'H': H_cur, 'W': W_cur}))

            # SE branch: depthwise → avgpool → fc1 → fc2
            dag.add_edge(dw_id, se_fc1_id, 'se_pool_input',
                         [N, mid_ch, 1, 1])
            dag.add_edge(se_fc1_id, se_fc2_id, 'se_reduced',
                         [N, se_reduced, 1, 1])

            # Multiply: depthwise_out * se_scale
            dag.add_edge(dw_id, se_mul_id, 'dw_features',
                         [N, mid_ch, H_cur, W_cur])
            dag.add_edge(se_fc2_id, se_mul_id, 'se_scale',
                         [N, mid_ch, 1, 1])

            prev_id = se_mul_id

        # Project conv (1×1)
        proj_id = f"block{blk_idx}_project"
        dag.add_node(DAGNode(proj_id, 'pointwise_conv2d',
                             f'layer{layer_counter}_features_{blk_idx + 1}_block_3_0.yaml',
                             {'N': N, 'C': mid_ch, 'M': out_ch,
                              'R': 1, 'S': 1, 'P': H_cur, 'Q': W_cur}))
        dag.add_edge(prev_id, proj_id, 'proj_input',
                     [N, mid_ch, H_cur, W_cur])
        layer_counter += 1

        # Skip connection (residual add)
        if has_skip:
            res_id = f"block{blk_idx}_residual_add"
            dag.add_node(DAGNode(res_id, 'residual_add', '',
                                 {'N': N, 'C': out_ch, 'H': H_cur, 'W': W_cur}))
            dag.add_edge(proj_id, res_id, 'proj_output',
                         [N, out_ch, H_cur, W_cur])
            dag.add_edge(block_input, res_id, 'residual',
                         [N, out_ch, H_cur, W_cur])
            prev_id = res_id
        else:
            prev_id = proj_id

        C_prev = out_ch

    # Final conv + classifier
    final_conv_id = "final_conv"
    dag.add_node(DAGNode(final_conv_id, 'pointwise_conv2d',
                         f'layer{layer_counter}_features_12_0.yaml',
                         {'N': N, 'C': C_prev, 'M': 576,
                          'R': 1, 'S': 1, 'P': H_cur, 'Q': W_cur}))
    dag.add_edge(prev_id, final_conv_id, 'final_features',
                 [N, C_prev, H_cur, W_cur])

    classifier_id = "classifier"
    dag.add_node(DAGNode(classifier_id, 'linear',
                         f'layer{layer_counter + 1}_classifier_0.yaml',
                         {'N': N, 'C': 576, 'M': 1024}))
    dag.add_edge(final_conv_id, classifier_id, 'pooled',
                 [N, 576])

    head_id = "head"
    dag.add_node(DAGNode(head_id, 'linear',
                         f'layer{layer_counter + 2}_classifier_3.yaml',
                         {'N': N, 'C': 1024, 'M': 1000}))
    dag.add_edge(classifier_id, head_id, 'features', [N, 1024])

    return dag


def generate_mobilenet_dag():
    """Generate DAG for MobileNetV3-Small."""
    for batch_size in [1]:
        dag = _build_mobilenetv3_small_dag(batch_size)
        outdir = os.path.join(DAG_OUT_DIR, f"mobilenet_v3_small_{batch_size}")
        os.makedirs(outdir, exist_ok=True)
        dag_path = os.path.join(outdir, "DAG.yaml")
        dag.save_yaml(dag_path)
        _print_dag_summary(dag, dag_path)


# ============================================================
# RepLKNet-31B DAG
# ============================================================

def _build_replknet31b_dag(batch_size: int = 1) -> NetworkDAG:
    """
    Build DAG for RepLKNet-31B.

    Architecture: stem → 4 stages → head
    Each stage has blocks. Each RepLKBlock:
        input → pw1 (1×1)
                 ├→ large_kernel (31×31/29×29/27×27/13×13 depthwise)
                 └→ small_kernel (5×5 depthwise)
                      └→ add (large + small)
                           └→ pw2 (1×1)
                                └→ [+ input]  (skip connection for every block)

    Some blocks lack the large+small kernel branch (just pw1 → pw2),
    these are the "non-LK" blocks that still have skip connections.

    Stage config: [3,3,3,3] blocks per stage with channels [128,256,512,1024]
    Large kernels: [31,29,27,13] per stage
    Block counts: [2,2,18,2] blocks per stage
    Blocks with LK: every other block has large kernel convs
    """
    # RepLKNet-31B configuration
    channels = [128, 256, 512, 1024]
    num_blocks = [2, 2, 18, 2]
    large_kernels = [31, 29, 27, 13]
    small_kernel = 5

    # From the actual workload files, identify which blocks have LK
    # Pattern: blocks 0, 2, 4, ... have LK; blocks 1, 3, 5, ... don't
    # (this matches RepLKNet's alternating pattern)

    dag = NetworkDAG(model_name='replknet31b')
    N = batch_size
    H, W = 224, 224

    # Stem (simplified as a single conv entry point — stem is multiple convs
    # but we treat it as a black box feeding into stage 0)
    stem_id = "stem"
    C_stem = channels[0]
    H_cur, W_cur = H // 4, W // 4  # stem has stride 4 total
    dag.add_node(DAGNode(stem_id, 'conv2d', '',
                         {'N': N, 'C': 3, 'M': C_stem,
                          'P': H_cur, 'Q': W_cur}))

    prev_id = stem_id
    layer_counter = 1
    C_prev = C_stem

    for stage_idx in range(4):
        C_out = channels[stage_idx]
        lk_size = large_kernels[stage_idx]
        n_blocks = num_blocks[stage_idx]

        # Transition downsample between stages (except stage 0)
        if stage_idx > 0:
            H_cur //= 2
            W_cur //= 2
            # Transition is 1×1 conv + pooling — simplified
            trans_id = f"transition_{stage_idx}"
            dag.add_node(DAGNode(trans_id, 'conv2d', '',
                                 {'N': N, 'C': C_prev, 'M': C_out,
                                  'P': H_cur, 'Q': W_cur}))
            dag.add_edge(prev_id, trans_id, 'stage_transition',
                         [N, C_prev, H_cur * 2, W_cur * 2])
            prev_id = trans_id
            C_prev = C_out

        for blk_idx in range(n_blocks):
            block_input = prev_id
            has_lk = (blk_idx % 2 == 0)  # alternating LK pattern

            # pw1 (1×1 pointwise conv)
            pw1_id = f"stages_{stage_idx}_blocks_{blk_idx}_pw1"
            dag.add_node(DAGNode(pw1_id, 'pointwise_conv2d',
                                 f'layer{layer_counter}_stages_{stage_idx}_blocks_{blk_idx}_pw1_conv.yaml',
                                 {'N': N, 'C': C_out, 'M': C_out,
                                  'R': 1, 'S': 1, 'P': H_cur, 'Q': W_cur}))
            dag.add_edge(prev_id, pw1_id, 'pw1_input',
                         [N, C_out, H_cur, W_cur])
            layer_counter += 1

            if has_lk:
                # Large kernel (depthwise)
                lk_id = f"stages_{stage_idx}_blocks_{blk_idx}_large_kernel"
                dag.add_node(DAGNode(lk_id, 'depthwise_conv2d',
                                     f'layer{layer_counter}_stages_{stage_idx}_blocks_{blk_idx}_large_kernel_lkb_origin_conv.yaml',
                                     {'N': N, 'C': C_out, 'M': C_out, 'G': C_out,
                                      'R': lk_size, 'S': lk_size,
                                      'P': H_cur, 'Q': W_cur}))
                dag.add_edge(pw1_id, lk_id, 'to_large_kernel',
                             [N, C_out, H_cur, W_cur])
                layer_counter += 1

                # Small kernel (depthwise) — parallel with large kernel
                sk_id = f"stages_{stage_idx}_blocks_{blk_idx}_small_kernel"
                dag.add_node(DAGNode(sk_id, 'depthwise_conv2d',
                                     f'layer{layer_counter}_stages_{stage_idx}_blocks_{blk_idx}_large_kernel_small_conv_conv.yaml',
                                     {'N': N, 'C': C_out, 'M': C_out, 'G': C_out,
                                      'R': small_kernel, 'S': small_kernel,
                                      'P': H_cur, 'Q': W_cur}))
                dag.add_edge(pw1_id, sk_id, 'to_small_kernel',
                             [N, C_out, H_cur, W_cur])
                layer_counter += 1

                # Add large + small kernel outputs
                kernel_add_id = f"stages_{stage_idx}_blocks_{blk_idx}_kernel_add"
                dag.add_node(DAGNode(kernel_add_id, 'elementwise_add', '',
                                     {'N': N, 'C': C_out,
                                      'H': H_cur, 'W': W_cur}))
                dag.add_edge(lk_id, kernel_add_id, 'large_kernel_out',
                             [N, C_out, H_cur, W_cur])
                dag.add_edge(sk_id, kernel_add_id, 'small_kernel_out',
                             [N, C_out, H_cur, W_cur])

                prev_for_pw2 = kernel_add_id
            else:
                prev_for_pw2 = pw1_id

            # pw2 (1×1 pointwise conv)
            pw2_id = f"stages_{stage_idx}_blocks_{blk_idx}_pw2"
            dag.add_node(DAGNode(pw2_id, 'pointwise_conv2d',
                                 f'layer{layer_counter}_stages_{stage_idx}_blocks_{blk_idx}_pw2_conv.yaml',
                                 {'N': N, 'C': C_out, 'M': C_out,
                                  'R': 1, 'S': 1, 'P': H_cur, 'Q': W_cur}))
            dag.add_edge(prev_for_pw2, pw2_id, 'pw2_input',
                         [N, C_out, H_cur, W_cur])
            layer_counter += 1

            # Residual add (every block has skip connection)
            res_id = f"stages_{stage_idx}_blocks_{blk_idx}_residual_add"
            dag.add_node(DAGNode(res_id, 'residual_add', '',
                                 {'N': N, 'C': C_out,
                                  'H': H_cur, 'W': W_cur}))
            dag.add_edge(pw2_id, res_id, 'pw2_output',
                         [N, C_out, H_cur, W_cur])
            dag.add_edge(block_input, res_id, 'residual',
                         [N, C_out, H_cur, W_cur])

            prev_id = res_id

        C_prev = C_out

    # Classification head
    head_id = "head"
    dag.add_node(DAGNode(head_id, 'linear',
                         f'layer{layer_counter}_head.yaml',
                         {'N': N, 'C': C_out, 'M': 1000}))
    dag.add_edge(prev_id, head_id, 'pooled_features', [N, C_out])

    return dag


def generate_replknet_dag():
    """Generate DAG for RepLKNet-31B."""
    for batch_size in [1]:
        dag = _build_replknet31b_dag(batch_size)
        outdir = os.path.join(DAG_OUT_DIR, f"replknet31b_{batch_size}")
        os.makedirs(outdir, exist_ok=True)
        dag_path = os.path.join(outdir, "DAG.yaml")
        dag.save_yaml(dag_path)
        _print_dag_summary(dag, dag_path)


# ============================================================
# Helpers
# ============================================================

def _print_dag_summary(dag: NetworkDAG, path: str):
    """Print a compact summary of the DAG."""
    levels = dag.parallel_groups()
    topo = dag.topological_order()

    # Count skip/residual edges
    skip_edges = sum(1 for e in dag.edges if 'residual' in e.tensor_name)
    parallel_nodes = sum(1 for lvl in levels if len(lvl) > 1)

    print(f"\n  DAG: {dag.model_name}")
    print(f"    Nodes: {len(dag.nodes)}, Edges: {len(dag.edges)}")
    print(f"    Parallel levels: {len(levels)} "
          f"(max width: {max(len(l) for l in levels)})")
    print(f"    Skip/residual edges: {skip_edges}")
    print(f"    Levels with parallelism: {parallel_nodes}")
    if dag.repeat_blocks:
        for name, info in dag.repeat_blocks.items():
            print(f"    Repeat block '{name}': ×{info['repeat_count']}")
    print(f"    Saved to: {path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate DAG representations for neural networks"
    )
    parser.add_argument(
        '--network', type=str, default='all',
        choices=['all', 'llama', 'llm', 'mobilenet', 'replknet'],
        help='Which network(s) to convert'
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Output directory (default: dag_workloads/)'
    )
    args = parser.parse_args()

    global DAG_OUT_DIR
    if args.output_dir:
        DAG_OUT_DIR = args.output_dir
    os.makedirs(DAG_OUT_DIR, exist_ok=True)

    logging.basicConfig(level=logging.INFO)

    targets = args.network.lower()

    if targets in ('all', 'llama', 'llm'):
        print("=" * 60)
        print("Generating LLM DAGs (LLaMA, Qwen)...")
        print("=" * 60)
        generate_llm_dags()

    if targets in ('all', 'mobilenet'):
        print("\n" + "=" * 60)
        print("Generating MobileNetV3-Small DAG...")
        print("=" * 60)
        generate_mobilenet_dag()

    if targets in ('all', 'replknet'):
        print("\n" + "=" * 60)
        print("Generating RepLKNet-31B DAG...")
        print("=" * 60)
        generate_replknet_dag()

    print("\n" + "=" * 60)
    print("DAG generation complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
