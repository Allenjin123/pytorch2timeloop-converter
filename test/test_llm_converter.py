"""
Test and verify the config-driven LLM converter.

Runs the converter on LLaMA 3.1 70B and Qwen 3 8B configs, then verifies
the generated YAML dimensions against actual PyTorch model instances.
"""
import os
import sys
import json

# Add parent to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pytorch2timeloop.convert_llm import (
    LLMConfig, convert_llm, verify_against_model
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)


def test_llama31_70b():
    """Test converter on LLaMA 3.1 70B."""
    config_path = os.path.join(
        os.path.dirname(PROJECT_DIR),
        "Llama-3.1-70B-Instruct", "config.json"
    )
    if not os.path.exists(config_path):
        print(f"SKIP: LLaMA config not found at {config_path}")
        return

    print("=" * 60)
    print("TEST: LLaMA 3.1 70B — Generate workloads")
    print("=" * 60)

    save_dir = os.path.join(PROJECT_DIR, "llm_workloads")
    operators = convert_llm(
        config_path=config_path,
        save_dir=save_dir,
        model_name="llama3.1_70b",
        batch_size=1,
        seq_len=2048,
    )

    # Basic sanity checks
    cfg = LLMConfig.from_json(config_path)
    expected_ops = 9 * cfg.num_hidden_layers + 1
    assert len(operators) == expected_ops, \
        f"Expected {expected_ops} operators, got {len(operators)}"
    print(f"  Operator count: {len(operators)} (expected {expected_ops}) — PASS")

    # Verify dimensions of first layer's q_proj
    q_proj = operators[0]
    assert q_proj.c == 8192, f"q_proj.c={q_proj.c}, expected 8192"
    assert q_proj.m == 8192, f"q_proj.m={q_proj.m}, expected 8192"
    assert q_proj.n == 2048, f"q_proj.n={q_proj.n}, expected 2048"
    print(f"  q_proj dims: C={q_proj.c}, M={q_proj.m}, N={q_proj.n} — PASS")

    # Verify k_proj (GQA: 8 KV heads * 128 head_dim = 1024)
    k_proj = operators[1]
    assert k_proj.c == 8192, f"k_proj.c={k_proj.c}, expected 8192"
    assert k_proj.m == 1024, f"k_proj.m={k_proj.m}, expected 1024"
    print(f"  k_proj dims: C={k_proj.c}, M={k_proj.m} — PASS")

    # Verify attention matmul dims
    attn_qk = operators[3]
    assert attn_qk.m == 2048, f"attn_qk.m={attn_qk.m}, expected 2048"
    assert attn_qk.k == 128, f"attn_qk.k={attn_qk.k}, expected 128"
    assert attn_qk.n == 2048, f"attn_qk.n={attn_qk.n}, expected 2048"
    assert attn_qk.extra_dims == (64,), \
        f"attn_qk.extra_dims={attn_qk.extra_dims}, expected (64,)"
    print(f"  attn_qk dims: M={attn_qk.m}, K={attn_qk.k}, N={attn_qk.n}, "
          f"batch={attn_qk.extra_dims} — PASS")

    # Verify MLP dims
    gate = operators[6]
    assert gate.c == 8192 and gate.m == 28672
    down = operators[8]
    assert down.c == 28672 and down.m == 8192
    print(f"  gate_proj: C={gate.c}, M={gate.m} — PASS")
    print(f"  down_proj: C={down.c}, M={down.m} — PASS")

    # Verify lm_head
    lm_head = operators[-1]
    assert lm_head.c == 8192 and lm_head.m == 128256
    print(f"  lm_head: C={lm_head.c}, M={lm_head.m} — PASS")

    print("\nLLaMA 3.1 70B: ALL CHECKS PASSED\n")


def test_qwen3_8b():
    """Test converter on Qwen 3 8B."""
    config_path = os.path.join(SCRIPT_DIR, "configs", "qwen3_8b", "config.json")

    print("=" * 60)
    print("TEST: Qwen 3 8B — Generate workloads")
    print("=" * 60)

    save_dir = os.path.join(PROJECT_DIR, "llm_workloads")
    operators = convert_llm(
        config_path=config_path,
        save_dir=save_dir,
        model_name="qwen3_8b",
        batch_size=1,
        seq_len=2048,
    )

    cfg = LLMConfig.from_json(config_path)
    expected_ops = 9 * cfg.num_hidden_layers + 1
    assert len(operators) == expected_ops, \
        f"Expected {expected_ops} operators, got {len(operators)}"
    print(f"  Operator count: {len(operators)} (expected {expected_ops}) — PASS")

    # Qwen3 8B: H=4096, n_h=32, n_kv=8, d=128, I=12288, V=151936
    q_proj = operators[0]
    assert q_proj.c == 4096 and q_proj.m == 4096
    print(f"  q_proj dims: C={q_proj.c}, M={q_proj.m} — PASS")

    k_proj = operators[1]
    assert k_proj.c == 4096 and k_proj.m == 1024  # 8 kv heads * 128
    print(f"  k_proj dims: C={k_proj.c}, M={k_proj.m} — PASS")

    attn_qk = operators[3]
    assert attn_qk.extra_dims == (32,)  # 32 query heads
    print(f"  attn_qk batch={attn_qk.extra_dims} (32 heads) — PASS")

    lm_head = operators[-1]
    assert lm_head.c == 4096 and lm_head.m == 151936
    print(f"  lm_head: C={lm_head.c}, M={lm_head.m} — PASS")

    print("\nQwen 3 8B: ALL CHECKS PASSED\n")


def test_verify_llama31():
    """Run full verification against actual LLaMA model."""
    config_path = os.path.join(
        os.path.dirname(PROJECT_DIR),
        "Llama-3.1-70B-Instruct"
    )
    if not os.path.exists(config_path):
        print("SKIP: LLaMA config not found")
        return

    print("=" * 60)
    print("TEST: LLaMA 3.1 70B — Verify against PyTorch model")
    print("=" * 60)

    passed = verify_against_model(
        config_path, batch_size=1, seq_len=128
    )
    assert passed, "LLaMA verification failed"
    print("\nLLaMA 3.1 70B VERIFICATION: PASSED\n")


def test_verify_qwen3():
    """Run full verification against actual Qwen 3 model."""
    config_path = os.path.join(SCRIPT_DIR, "configs", "qwen3_8b")

    print("=" * 60)
    print("TEST: Qwen 3 8B — Verify against PyTorch model")
    print("=" * 60)

    passed = verify_against_model(
        config_path, batch_size=1, seq_len=128
    )
    assert passed, "Qwen 3 verification failed"
    print("\nQwen 3 8B VERIFICATION: PASSED\n")


if __name__ == '__main__':
    test_llama31_70b()
    test_qwen3_8b()
    test_verify_llama31()
    test_verify_qwen3()

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
