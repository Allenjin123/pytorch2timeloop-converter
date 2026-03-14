"""
Test pytorch2timeloop conversion for LLaMA 3.1 70B.
Since torch.fx.symbolic_trace() fails on LLaMA (dynamic control flow),
we try multiple approaches and analyze what works/what's missing.
"""
import sys
import os
import logging
import traceback

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.fx as fx
from transformers import AutoConfig, AutoModelForCausalLM

# ============================================================
# Step 1: Load model
# ============================================================
config_path = "/home/allenjin/Projects/Mozart/Llama-3.1-70B-Instruct"
config = AutoConfig.from_pretrained(config_path)
config.num_hidden_layers = 1  # 1 layer for testing
model = AutoModelForCausalLM.from_config(config)
model.eval()

print("LLaMA 3.1 70B (1-layer) modules:")
for name, mod in model.named_modules():
    print(f"  {name}: {type(mod).__name__}")

# ============================================================
# Step 2: Test torch.fx.symbolic_trace
# ============================================================
print("\n" + "=" * 60)
print("Test 1: torch.fx.symbolic_trace()")
print("=" * 60)
try:
    traced = fx.symbolic_trace(model)
    print("SUCCESS")
except Exception as e:
    print(f"FAILED: {type(e).__name__}: {e}")

# ============================================================
# Step 3: Test on individual sub-modules
# ============================================================
print("\n" + "=" * 60)
print("Test 2: Trace individual sub-modules")
print("=" * 60)

sub_modules_to_test = {
    "model.layers.0.mlp": model.model.layers[0].mlp,
    "model.layers.0.self_attn": model.model.layers[0].self_attn,
    "model.layers.0.input_layernorm": model.model.layers[0].input_layernorm,
    "model.layers.0": model.model.layers[0],
    "model": model.model,
}

for name, mod in sub_modules_to_test.items():
    try:
        traced = fx.symbolic_trace(mod)
        print(f"  {name}: SUCCESS")
    except Exception as e:
        print(f"  {name}: FAILED ({type(e).__name__}: {e})")

# ============================================================
# Step 4: Test converter's generate_description on each module type
# ============================================================
print("\n" + "=" * 60)
print("Test 3: generate_description() on each module type")
print("=" * 60)

from pytorch2timeloop.utils.converter import generate_description

seq_len = 128
batch_size = 1
hidden_size = config.hidden_size
num_heads = config.num_attention_heads
num_kv_heads = config.num_key_value_heads
head_dim = hidden_size // num_heads
intermediate_size = config.intermediate_size

test_cases = [
    ("q_proj (Linear 8192->8192)", model.model.layers[0].self_attn.q_proj,
     torch.randn(batch_size, seq_len, hidden_size),
     torch.randn(batch_size, seq_len, hidden_size)),
    ("k_proj (Linear 8192->1024)", model.model.layers[0].self_attn.k_proj,
     torch.randn(batch_size, seq_len, hidden_size),
     torch.randn(batch_size, seq_len, num_kv_heads * head_dim)),
    ("v_proj (Linear 8192->1024)", model.model.layers[0].self_attn.v_proj,
     torch.randn(batch_size, seq_len, hidden_size),
     torch.randn(batch_size, seq_len, num_kv_heads * head_dim)),
    ("o_proj (Linear 8192->8192)", model.model.layers[0].self_attn.o_proj,
     torch.randn(batch_size, seq_len, hidden_size),
     torch.randn(batch_size, seq_len, hidden_size)),
    ("gate_proj (Linear 8192->28672)", model.model.layers[0].mlp.gate_proj,
     torch.randn(batch_size, seq_len, hidden_size),
     torch.randn(batch_size, seq_len, intermediate_size)),
    ("up_proj (Linear 8192->28672)", model.model.layers[0].mlp.up_proj,
     torch.randn(batch_size, seq_len, hidden_size),
     torch.randn(batch_size, seq_len, intermediate_size)),
    ("down_proj (Linear 28672->8192)", model.model.layers[0].mlp.down_proj,
     torch.randn(batch_size, seq_len, intermediate_size),
     torch.randn(batch_size, seq_len, hidden_size)),
    ("lm_head (Linear 8192->128256)", model.lm_head,
     torch.randn(batch_size, seq_len, hidden_size),
     torch.randn(batch_size, seq_len, config.vocab_size)),
    ("input_layernorm (LlamaRMSNorm)", model.model.layers[0].input_layernorm,
     torch.randn(batch_size, seq_len, hidden_size),
     torch.randn(batch_size, seq_len, hidden_size)),
    ("embed_tokens (Embedding)", model.model.embed_tokens,
     torch.randint(0, config.vocab_size, (batch_size, seq_len)),
     torch.randn(batch_size, seq_len, hidden_size)),
]

import yaml
save_dir = os.path.join(os.path.dirname(__file__), "llama31_workloads")
os.makedirs(os.path.join(save_dir, "llama3.1_70b"), exist_ok=True)

generated_files = []
failed_modules = []

for i, (desc, mod, inp, out) in enumerate(test_cases):
    try:
        layer_desc = generate_description(mod, inp, out, desc.split()[0], f"input_{i}")
        yaml_data = layer_desc.to_yaml()
        fname = f"layer{i+1}_{desc.split()[0]}.yaml"
        fpath = os.path.join(save_dir, "llama3.1_70b", fname)
        with open(fpath, 'w') as f:
            f.write(yaml.dump(yaml_data, default_flow_style=False))
        generated_files.append((desc, fname, yaml_data))
        print(f"  OK: {desc} -> {fname}")
    except Exception as e:
        failed_modules.append((desc, type(e).__name__, str(e)))
        print(f"  FAIL: {desc} -> {type(e).__name__}: {e}")

# ============================================================
# Step 5: Check what operations are MISSING
# ============================================================
print("\n" + "=" * 60)
print("Test 4: What operations are MISSING from converter")
print("=" * 60)

missing_ops = [
    "Embedding (nn.Embedding) - token embedding lookup",
    "LlamaRMSNorm - RMS normalization (different from LayerNorm)",
    "LlamaRotaryEmbedding (RoPE) - rotary position embeddings",
    "SiLUActivation - SiLU/Swish activation in MLP",
    "torch.matmul for Q@K^T attention scores (batched, GQA)",
    "torch.matmul for attn_weights@V (batched, GQA)",
    "Softmax for attention weights",
    "Element-wise multiply: gate_proj * up_proj in SwiGLU MLP",
    "Causal attention mask application",
    "KV repeat for GQA (repeat_kv)",
]

print("\nMissing operations that need converter support:")
for op in missing_ops:
    print(f"  - {op}")

# ============================================================
# Step 6: Validate generated YAML correctness
# ============================================================
print("\n" + "=" * 60)
print("Test 5: Validate generated YAML correctness")
print("=" * 60)

for desc, fname, yaml_data in generated_files:
    print(f"\n--- {desc} ({fname}) ---")
    problem = yaml_data.get('problem', {})
    instance = problem.get('instance', {})
    shape = problem.get('shape', {})

    print(f"  Shape name: {shape.get('name', 'N/A')}")
    print(f"  Dimensions: {shape.get('dimensions', 'N/A')}")
    print(f"  Instance: {instance}")

    # Validate specific dimensions
    if 'q_proj' in desc or 'o_proj' in desc:
        expected_c = hidden_size
        expected_m = hidden_size
        actual_c = instance.get('C', 0)
        actual_m = instance.get('M', 0)
        correct = (actual_c == expected_c and actual_m == expected_m)
        print(f"  Expected C={expected_c}, M={expected_m}")
        print(f"  Got C={actual_c}, M={actual_m}")
        print(f"  CORRECT: {correct}")

        # Check if sequence dimension is captured
        if instance.get('N') == batch_size:
            print(f"  WARNING: N={batch_size} only captures batch, NOT batch*seq_len={batch_size*seq_len}")
            print(f"           Linear layers in LLMs operate on (batch*seq_len, hidden_size)")
            print(f"           The converter models Linear as 1x1 conv, losing seq dimension")

    if 'k_proj' in desc or 'v_proj' in desc:
        expected_c = hidden_size
        expected_m = num_kv_heads * head_dim  # 8 * 128 = 1024
        actual_c = instance.get('C', 0)
        actual_m = instance.get('M', 0)
        correct = (actual_c == expected_c and actual_m == expected_m)
        print(f"  Expected C={expected_c}, M={expected_m}")
        print(f"  Got C={actual_c}, M={actual_m}")
        print(f"  CORRECT: {correct}")

    if 'gate_proj' in desc or 'up_proj' in desc:
        expected_c = hidden_size
        expected_m = intermediate_size
        actual_c = instance.get('C', 0)
        actual_m = instance.get('M', 0)
        correct = (actual_c == expected_c and actual_m == expected_m)
        print(f"  Expected C={expected_c}, M={expected_m}")
        print(f"  Got C={actual_c}, M={actual_m}")
        print(f"  CORRECT: {correct}")

    if 'down_proj' in desc:
        expected_c = intermediate_size
        expected_m = hidden_size
        actual_c = instance.get('C', 0)
        actual_m = instance.get('M', 0)
        correct = (actual_c == expected_c and actual_m == expected_m)
        print(f"  Expected C={expected_c}, M={expected_m}")
        print(f"  Got C={actual_c}, M={actual_m}")
        print(f"  CORRECT: {correct}")

# ============================================================
# Step 7: Print generated YAML for one example
# ============================================================
print("\n" + "=" * 60)
print("Example generated YAML (q_proj):")
print("=" * 60)
if generated_files:
    print(yaml.dump(generated_files[0][2], default_flow_style=False))

# ============================================================
# Step 8: Analyze attention matmuls that are NOT generated
# ============================================================
print("\n" + "=" * 60)
print("Test 6: Missing attention matmul dimensions")
print("=" * 60)

print(f"\nFor LLaMA 3.1 70B with seq_len={seq_len}:")
print(f"  GQA: {num_heads} query heads, {num_kv_heads} KV heads")
print(f"  GQA ratio: {num_heads // num_kv_heads} query heads per KV head")
print(f"  head_dim: {head_dim}")
print(f"\n  Q@K^T attention scores matmul:")
print(f"    Q shape: (batch, num_heads, seq_len, head_dim) = ({batch_size}, {num_heads}, {seq_len}, {head_dim})")
print(f"    K^T shape: (batch, num_heads, head_dim, seq_len) = ({batch_size}, {num_heads}, {head_dim}, {seq_len})")
print(f"    -> Matmul: M={seq_len}, K={head_dim}, N={seq_len}, batch={batch_size}*{num_heads}")
print(f"\n  attn_weights@V matmul:")
print(f"    attn shape: (batch, num_heads, seq_len, seq_len) = ({batch_size}, {num_heads}, {seq_len}, {seq_len})")
print(f"    V shape: (batch, num_heads, seq_len, head_dim) = ({batch_size}, {num_heads}, {seq_len}, {head_dim})")
print(f"    -> Matmul: M={seq_len}, K={seq_len}, N={head_dim}, batch={batch_size}*{num_heads}")
print(f"\n  NOTE: KV heads are repeated {num_heads // num_kv_heads}x for GQA before these matmuls")
print(f"  Actual unique KV computation: {num_kv_heads} heads, but used by {num_heads} query heads")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"\nGenerated: {len(generated_files)} layer YAML files")
print(f"Failed: {len(failed_modules)} modules")
for desc, etype, emsg in failed_modules:
    print(f"  - {desc}: {etype}: {emsg}")
