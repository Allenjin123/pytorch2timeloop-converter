#!/usr/bin/env python3
"""
Demo: Generate Timeloop workloads from StableHLO MLIR.

This script demonstrates three usage modes:
  1. Sample MLIR (no framework deps) — generate synthetic transformer/CNN MLIR
  2. From .mlir file — parse an existing StableHLO file
  3. From PyTorch model — export via torch_xla then convert (requires torch_xla)

Usage:
    # Mode 1: Built-in sample models (no extra deps needed)
    python convert_stablehlo_models.py --sample

    # Mode 2: From existing .mlir file
    python convert_stablehlo_models.py --mlir path/to/model.mlir --model-name my_model

    # Mode 3: From PyTorch (requires torch_xla)
    python convert_stablehlo_models.py --pytorch resnet18
"""

import argparse
import os
import sys
import logging

# Add parent to path for development
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pytorch2timeloop.convert_stablehlo import convert_stablehlo  # noqa: E402
from pytorch2timeloop.export_stablehlo import generate_sample_mlir  # noqa: E402


logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def run_sample_models(save_base: str):
    """Generate and convert sample StableHLO MLIR for testing."""

    configs = [
        # Transformer block: LLaMA-8B-like single layer
        {
            'model_type': 'transformer_block',
            'model_name': 'stablehlo_llama8b_layer',
            'batch_size': 1,
            'seq_len': 2048,
            'hidden_size': 4096,
            'num_heads': 32,
            'intermediate_size': 14336,
        },
        # Transformer block: GPT-2 scale
        {
            'model_type': 'transformer_block',
            'model_name': 'stablehlo_gpt2_layer',
            'batch_size': 1,
            'seq_len': 512,
            'hidden_size': 768,
            'num_heads': 12,
            'intermediate_size': 3072,
        },
        # CNN block
        {
            'model_type': 'cnn_block',
            'model_name': 'stablehlo_cnn_block',
            'batch_size': 1,
        },
        # MLP
        {
            'model_type': 'mlp',
            'model_name': 'stablehlo_mlp',
            'batch_size': 1,
            'seq_len': 2048,
            'hidden_size': 4096,
            'intermediate_size': 14336,
        },
    ]

    for cfg in configs:
        model_name = cfg.pop('model_name')
        model_type = cfg.pop('model_type')

        logger.info(f"\n{'='*60}")
        logger.info(f"Generating: {model_name} ({model_type})")
        logger.info(f"{'='*60}")

        # Generate sample MLIR
        mlir_path = os.path.join('/tmp', f'{model_name}.mlir')
        generate_sample_mlir(
            model_type=model_type,
            output_path=mlir_path,
            **cfg,
        )

        # Convert to Timeloop workloads
        save_dir = os.path.join(save_base, model_name)
        ops, dag = convert_stablehlo(
            mlir_path=mlir_path,
            save_dir=save_dir,
            model_name=model_name,
            include_reduces=True,
            include_elementwise=False,
            build_dag=True,
            dedup=True,
        )

        # Print summary
        print(f"\n  Model: {model_name}")
        print(f"  Operators: {len(ops)}")
        if dag:
            levels = dag.parallel_groups()
            print(f"  DAG: {len(dag.nodes)} nodes, {len(dag.edges)} edges, "
                  f"{len(levels)} parallel levels")
            for i, level in enumerate(levels):
                print(f"    Level {i}: {level}")
        print(f"  Output: {save_dir}/")

        # List generated files
        for fname in sorted(os.listdir(save_dir)):
            fpath = os.path.join(save_dir, fname)
            size = os.path.getsize(fpath)
            print(f"    {fname} ({size} bytes)")


def run_from_mlir(mlir_path: str, save_dir: str, model_name: str):
    """Convert an existing .mlir file."""
    ops, dag = convert_stablehlo(
        mlir_path=mlir_path,
        save_dir=save_dir,
        model_name=model_name,
        include_reduces=True,
        build_dag=True,
    )

    print(f"\nConverted {len(ops)} operators from {mlir_path}")
    if dag:
        print(f"DAG: {len(dag.nodes)} nodes, {len(dag.edges)} edges")
    print(f"Output: {save_dir}/")


def main():
    parser = argparse.ArgumentParser(description='StableHLO → Timeloop demo')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--sample', action='store_true',
                       help='Run with built-in sample models')
    group.add_argument('--mlir', type=str, help='Path to .mlir file')

    parser.add_argument('--save-dir', type=str,
                        default='stablehlo_workloads',
                        help='Output directory (default: stablehlo_workloads)')
    parser.add_argument('--model-name', type=str, default=None,
                        help='Model name (default: filename stem)')

    args = parser.parse_args()

    if args.sample:
        run_sample_models(args.save_dir)
    elif args.mlir:
        model_name = args.model_name or os.path.splitext(
            os.path.basename(args.mlir))[0]
        run_from_mlir(args.mlir, args.save_dir, model_name)


if __name__ == '__main__':
    main()
