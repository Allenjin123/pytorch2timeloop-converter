"""
Export PyTorch or JAX models to StableHLO MLIR text format.

This module provides helpers to go from a PyTorch nn.Module or JAX function
to a .mlir file that can be consumed by convert_stablehlo.py.

Requirements:
  - PyTorch path: torch, torch_xla (pip install torch torch_xla)
  - JAX path: jax, jaxlib (pip install jax jaxlib)

Usage:
    # PyTorch
    from pytorch2timeloop.export_stablehlo import export_pytorch_to_stablehlo
    export_pytorch_to_stablehlo(model, sample_input, "model.mlir")

    # JAX
    from pytorch2timeloop.export_stablehlo import export_jax_to_stablehlo
    export_jax_to_stablehlo(jax_fn, abstract_args, "model.mlir")

    # HuggingFace (auto-downloads + exports)
    from pytorch2timeloop.export_stablehlo import export_hf_to_stablehlo
    export_hf_to_stablehlo("meta-llama/Llama-3.1-8B", "llama.mlir",
                           seq_len=512, batch_size=1)
"""

import logging
from pathlib import Path
from typing import Any, Optional, Tuple, Union

logger = logging.getLogger(__name__)


def export_pytorch_to_stablehlo(
    model,
    sample_input: Any,
    output_path: str,
    save_weights: bool = False,
) -> str:
    """
    Export a PyTorch model to StableHLO MLIR text format.

    Args:
        model: A torch.nn.Module instance.
        sample_input: Sample input tensor or tuple of tensors.
        output_path: Path to write the .mlir file.
        save_weights: If True, also save weight constants in the MLIR.

    Returns:
        Path to the written .mlir file.

    Requires: torch, torch_xla
    """
    try:
        import torch
        from torch.export import export as torch_export
    except ImportError:
        raise ImportError("torch is required. Install with: pip install torch")

    try:
        from torch_xla.stablehlo import exported_program_to_stablehlo
    except ImportError:
        raise ImportError(
            "torch_xla is required for StableHLO export. "
            "Install with: pip install torch_xla"
        )

    model.eval()

    # Ensure sample_input is a tuple
    if not isinstance(sample_input, tuple):
        sample_input = (sample_input,)

    logger.info("Tracing model with torch.export...")
    exported = torch_export(model, sample_input)

    logger.info("Converting to StableHLO...")
    from torch_xla.stablehlo import StableHLOExportOptions
    options = StableHLOExportOptions()
    options.include_human_readable_text = True
    options.save_weights = save_weights

    stablehlo_program = exported_program_to_stablehlo(exported, options=options)

    # Get the MLIR text
    mlir_text = stablehlo_program.get_stablehlo_text('forward')

    output_path = str(output_path)
    with open(output_path, 'w') as f:
        f.write(mlir_text)

    logger.info(f"Wrote StableHLO MLIR to {output_path}")
    return output_path


def export_jax_to_stablehlo(
    fn,
    abstract_args: Any,
    output_path: str,
) -> str:
    """
    Export a JAX function to StableHLO MLIR text format.

    Args:
        fn: A JAX function (will be jitted if not already).
        abstract_args: Abstract shape specs (jax.ShapeDtypeStruct instances).
        output_path: Path to write the .mlir file.

    Returns:
        Path to the written .mlir file.

    Requires: jax
    """
    try:
        import jax
        from jax import export as jax_export
    except ImportError:
        raise ImportError("jax is required. Install with: pip install jax jaxlib")

    try:
        from jax._src.interpreters import mlir as jax_mlir
        from jax._src.lib.mlir import ir
    except ImportError:
        raise ImportError("jax MLIR bindings not available in this jax version")

    # Ensure function is jitted
    if not hasattr(fn, 'lower'):
        fn = jax.jit(fn)

    logger.info("Exporting JAX function to StableHLO...")
    if not isinstance(abstract_args, (tuple, list)):
        abstract_args = (abstract_args,)

    exported = jax_export.export(fn)(*abstract_args)
    mlir_module = exported.mlir_module()

    # Convert to text
    with jax_mlir.make_ir_context() as ctx:
        module = ir.Module.parse(mlir_module, context=ctx)
        mlir_text = module.operation.get_asm(large_elements_limit=20)

    output_path = str(output_path)
    with open(output_path, 'w') as f:
        f.write(mlir_text)

    logger.info(f"Wrote StableHLO MLIR to {output_path}")
    return output_path


def export_hf_to_stablehlo(
    model_name_or_path: str,
    output_path: str,
    seq_len: int = 512,
    batch_size: int = 1,
    save_weights: bool = False,
) -> str:
    """
    Export a HuggingFace model to StableHLO MLIR text format.

    This auto-downloads the model and exports it via torch.export + torch_xla.

    Args:
        model_name_or_path: HuggingFace model name or local path.
        output_path: Path to write the .mlir file.
        seq_len: Sequence length for the sample input.
        batch_size: Batch size for the sample input.
        save_weights: If True, include weight constants in the MLIR.

    Returns:
        Path to the written .mlir file.

    Requires: torch, torch_xla, transformers
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        raise ImportError(
            "torch and transformers are required. "
            "Install with: pip install torch transformers"
        )

    logger.info(f"Loading model {model_name_or_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    model.eval()

    # Create sample input
    sample_input = torch.randint(0, 1000, (batch_size, seq_len))

    return export_pytorch_to_stablehlo(
        model, (sample_input,), output_path, save_weights=save_weights
    )


def generate_sample_mlir(
    model_type: str = 'transformer_block',
    batch_size: int = 1,
    seq_len: int = 512,
    hidden_size: int = 768,
    num_heads: int = 12,
    intermediate_size: int = 3072,
    output_path: Optional[str] = None,
) -> str:
    """
    Generate a sample StableHLO MLIR text for testing without any framework.

    This creates synthetic MLIR that represents common model patterns.
    Useful for testing the converter without needing PyTorch/JAX/torch_xla.

    Args:
        model_type: 'transformer_block', 'cnn_block', or 'mlp'.
        output_path: If given, write to file. Otherwise return as string.

    Returns:
        MLIR text string (also written to output_path if provided).
    """
    head_dim = hidden_size // num_heads

    if model_type == 'transformer_block':
        mlir = _generate_transformer_mlir(
            batch_size, seq_len, hidden_size, num_heads, head_dim,
            intermediate_size)
    elif model_type == 'cnn_block':
        mlir = _generate_cnn_mlir(batch_size)
    elif model_type == 'mlp':
        mlir = _generate_mlp_mlir(batch_size, seq_len, hidden_size,
                                   intermediate_size)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    if output_path:
        with open(output_path, 'w') as f:
            f.write(mlir)
        logger.info(f"Wrote sample MLIR to {output_path}")

    return mlir


def _generate_transformer_mlir(B, S, H, n_heads, head_dim, I):
    """Generate StableHLO MLIR for a single transformer decoder layer."""
    return f"""module {{
  func.func @main(%arg0: tensor<{B}x{S}x{H}xf32>) -> tensor<{B}x{S}x{H}xf32> {{
    // === Q/K/V Projections (parallel) ===
    %w_q = stablehlo.constant dense<0.01> : tensor<{H}x{H}xf32>
    %w_k = stablehlo.constant dense<0.01> : tensor<{H}x{n_heads * head_dim // (n_heads // max(1, n_heads // 8))}xf32>
    %w_v = stablehlo.constant dense<0.01> : tensor<{H}x{n_heads * head_dim // (n_heads // max(1, n_heads // 8))}xf32>
    %w_o = stablehlo.constant dense<0.01> : tensor<{H}x{H}xf32>
    %w_gate = stablehlo.constant dense<0.01> : tensor<{H}x{I}xf32>
    %w_up = stablehlo.constant dense<0.01> : tensor<{H}x{I}xf32>
    %w_down = stablehlo.constant dense<0.01> : tensor<{I}x{H}xf32>

    // Q projection: [B, S, H] @ [H, H] -> [B, S, H]
    %q = stablehlo.dot_general %arg0, %w_q, batching_dims = [] x [], contracting_dims = [2] x [0] : (tensor<{B}x{S}x{H}xf32>, tensor<{H}x{H}xf32>) -> tensor<{B}x{S}x{H}xf32>

    // K projection: [B, S, H] @ [H, Hkv] -> [B, S, Hkv]
    %k = stablehlo.dot_general %arg0, %w_k, batching_dims = [] x [], contracting_dims = [2] x [0] : (tensor<{B}x{S}x{H}xf32>, tensor<{H}x{n_heads * head_dim // (n_heads // max(1, n_heads // 8))}xf32>) -> tensor<{B}x{S}x{n_heads * head_dim // (n_heads // max(1, n_heads // 8))}xf32>

    // V projection (same dims as K for GQA)
    %v = stablehlo.dot_general %arg0, %w_v, batching_dims = [] x [], contracting_dims = [2] x [0] : (tensor<{B}x{S}x{H}xf32>, tensor<{H}x{n_heads * head_dim // (n_heads // max(1, n_heads // 8))}xf32>) -> tensor<{B}x{S}x{n_heads * head_dim // (n_heads // max(1, n_heads // 8))}xf32>

    // Reshape Q to [B, n_heads, S, head_dim]
    %q_reshaped = stablehlo.reshape %q : (tensor<{B}x{S}x{H}xf32>) -> tensor<{B}x{n_heads}x{S}x{head_dim}xf32>
    %k_reshaped = stablehlo.reshape %k : (tensor<{B}x{S}x{n_heads * head_dim // (n_heads // max(1, n_heads // 8))}xf32>) -> tensor<{B}x{n_heads}x{S}x{head_dim}xf32>
    %v_reshaped = stablehlo.reshape %v : (tensor<{B}x{S}x{n_heads * head_dim // (n_heads // max(1, n_heads // 8))}xf32>) -> tensor<{B}x{n_heads}x{S}x{head_dim}xf32>

    // Attention: Q @ K^T = [B, H, S, D] @ [B, H, S, D]^T -> [B, H, S, S]
    %scores = stablehlo.dot_general %q_reshaped, %k_reshaped, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3] : (tensor<{B}x{n_heads}x{S}x{head_dim}xf32>, tensor<{B}x{n_heads}x{S}x{head_dim}xf32>) -> tensor<{B}x{n_heads}x{S}x{S}xf32>

    // Softmax: max -> sub -> exp -> sum -> div
    %neg_inf = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %softmax_max = stablehlo.reduce(%scores init: %neg_inf) across dimensions = [3] : (tensor<{B}x{n_heads}x{S}x{S}xf32>, tensor<f32>) -> tensor<{B}x{n_heads}x{S}xf32>
     reducer(%a0: tensor<f32>, %a1: tensor<f32>) {{
      %m = stablehlo.maximum %a0, %a1 : tensor<f32>
      stablehlo.return %m : tensor<f32>
    }}
    %max_broadcast = stablehlo.broadcast_in_dim %softmax_max, dims = [0, 1, 2] : (tensor<{B}x{n_heads}x{S}xf32>) -> tensor<{B}x{n_heads}x{S}x{S}xf32>
    %shifted = stablehlo.subtract %scores, %max_broadcast : tensor<{B}x{n_heads}x{S}x{S}xf32>
    %exp_scores = stablehlo.exponential %shifted : tensor<{B}x{n_heads}x{S}x{S}xf32>
    %zero = stablehlo.constant dense<0.0> : tensor<f32>
    %softmax_sum = stablehlo.reduce(%exp_scores init: %zero) across dimensions = [3] : (tensor<{B}x{n_heads}x{S}x{S}xf32>, tensor<f32>) -> tensor<{B}x{n_heads}x{S}xf32>
     reducer(%a2: tensor<f32>, %a3: tensor<f32>) {{
      %s = stablehlo.add %a2, %a3 : tensor<f32>
      stablehlo.return %s : tensor<f32>
    }}
    %sum_broadcast = stablehlo.broadcast_in_dim %softmax_sum, dims = [0, 1, 2] : (tensor<{B}x{n_heads}x{S}xf32>) -> tensor<{B}x{n_heads}x{S}x{S}xf32>
    %attn_weights = stablehlo.divide %exp_scores, %sum_broadcast : tensor<{B}x{n_heads}x{S}x{S}xf32>

    // Attention @ V: [B, H, S, S] @ [B, H, S, D] -> [B, H, S, D]
    %context = stablehlo.dot_general %attn_weights, %v_reshaped, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2] : (tensor<{B}x{n_heads}x{S}x{S}xf32>, tensor<{B}x{n_heads}x{S}x{head_dim}xf32>) -> tensor<{B}x{n_heads}x{S}x{head_dim}xf32>

    // Reshape back to [B, S, H]
    %context_flat = stablehlo.reshape %context : (tensor<{B}x{n_heads}x{S}x{head_dim}xf32>) -> tensor<{B}x{S}x{H}xf32>

    // O projection: [B, S, H] @ [H, H] -> [B, S, H]
    %o_proj = stablehlo.dot_general %context_flat, %w_o, batching_dims = [] x [], contracting_dims = [2] x [0] : (tensor<{B}x{S}x{H}xf32>, tensor<{H}x{H}xf32>) -> tensor<{B}x{S}x{H}xf32>

    // Residual add (attention)
    %attn_out = stablehlo.add %arg0, %o_proj : tensor<{B}x{S}x{H}xf32>

    // === MLP (SwiGLU) ===
    // Gate projection: [B, S, H] @ [H, I] -> [B, S, I]
    %gate = stablehlo.dot_general %attn_out, %w_gate, batching_dims = [] x [], contracting_dims = [2] x [0] : (tensor<{B}x{S}x{H}xf32>, tensor<{H}x{I}xf32>) -> tensor<{B}x{S}x{I}xf32>

    // Up projection: [B, S, H] @ [H, I] -> [B, S, I]
    %up = stablehlo.dot_general %attn_out, %w_up, batching_dims = [] x [], contracting_dims = [2] x [0] : (tensor<{B}x{S}x{H}xf32>, tensor<{H}x{I}xf32>) -> tensor<{B}x{S}x{I}xf32>

    // SwiGLU: gate * sigmoid(gate) * up (simplified as gate * up)
    %mlp_mid = stablehlo.multiply %gate, %up : tensor<{B}x{S}x{I}xf32>

    // Down projection: [B, S, I] @ [I, H] -> [B, S, H]
    %down = stablehlo.dot_general %mlp_mid, %w_down, batching_dims = [] x [], contracting_dims = [2] x [0] : (tensor<{B}x{S}x{I}xf32>, tensor<{I}x{H}xf32>) -> tensor<{B}x{S}x{H}xf32>

    // Residual add (MLP)
    %output = stablehlo.add %attn_out, %down : tensor<{B}x{S}x{H}xf32>

    return %output : tensor<{B}x{S}x{H}xf32>
  }}
}}
"""


def _generate_cnn_mlir(B):
    """Generate StableHLO MLIR for a simple CNN block."""
    return f"""module {{
  func.func @main(%arg0: tensor<{B}x3x224x224xf32>) -> tensor<{B}x256x14x14xf32> {{
    // Conv1: 3 -> 64, 7x7, stride 2
    %w1 = stablehlo.constant dense<0.01> : tensor<64x3x7x7xf32>
    %conv1 = stablehlo.convolution(%arg0, %w1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {{stride = [2, 2], pad = [[3, 3], [3, 3]]}} {{feature_group_count = 1 : i64, batch_group_count = 1 : i64}} : (tensor<{B}x3x224x224xf32>, tensor<64x3x7x7xf32>) -> tensor<{B}x64x112x112xf32>

    // Conv2: 64 -> 128, 3x3, stride 2
    %w2 = stablehlo.constant dense<0.01> : tensor<128x64x3x3xf32>
    %conv2 = stablehlo.convolution(%conv1, %w2) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {{stride = [2, 2], pad = [[1, 1], [1, 1]]}} {{feature_group_count = 1 : i64, batch_group_count = 1 : i64}} : (tensor<{B}x64x112x112xf32>, tensor<128x64x3x3xf32>) -> tensor<{B}x128x56x56xf32>

    // Conv3: 128 -> 256, 3x3, stride 2
    %w3 = stablehlo.constant dense<0.01> : tensor<256x128x3x3xf32>
    %conv3 = stablehlo.convolution(%conv2, %w3) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {{stride = [2, 2], pad = [[1, 1], [1, 1]]}} {{feature_group_count = 1 : i64, batch_group_count = 1 : i64}} : (tensor<{B}x128x56x56xf32>, tensor<256x128x3x3xf32>) -> tensor<{B}x256x28x28xf32>

    // Depthwise Conv: 256 -> 256, 3x3 (groups=256)
    %w4 = stablehlo.constant dense<0.01> : tensor<256x1x3x3xf32>
    %dw_conv = stablehlo.convolution(%conv3, %w4) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {{stride = [2, 2], pad = [[1, 1], [1, 1]]}} {{feature_group_count = 256 : i64, batch_group_count = 1 : i64}} : (tensor<{B}x256x28x28xf32>, tensor<256x1x3x3xf32>) -> tensor<{B}x256x14x14xf32>

    return %dw_conv : tensor<{B}x256x14x14xf32>
  }}
}}
"""


def _generate_mlp_mlir(B, S, H, I):
    """Generate StableHLO MLIR for a simple MLP."""
    return f"""module {{
  func.func @main(%arg0: tensor<{B}x{S}x{H}xf32>) -> tensor<{B}x{S}x{H}xf32> {{
    %w1 = stablehlo.constant dense<0.01> : tensor<{H}x{I}xf32>
    %w2 = stablehlo.constant dense<0.01> : tensor<{I}x{H}xf32>

    // Up projection: [B, S, H] @ [H, I] -> [B, S, I]
    %hidden = stablehlo.dot_general %arg0, %w1, batching_dims = [] x [], contracting_dims = [2] x [0] : (tensor<{B}x{S}x{H}xf32>, tensor<{H}x{I}xf32>) -> tensor<{B}x{S}x{I}xf32>

    // Down projection: [B, S, I] @ [I, H] -> [B, S, H]
    %output = stablehlo.dot_general %hidden, %w2, batching_dims = [] x [], contracting_dims = [2] x [0] : (tensor<{B}x{S}x{I}xf32>, tensor<{I}x{H}xf32>) -> tensor<{B}x{S}x{H}xf32>

    return %output : tensor<{B}x{S}x{H}xf32>
  }}
}}
"""
