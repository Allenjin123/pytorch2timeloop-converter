# LLM Operator Namespace for Timeloop

This document defines the dimension and data space naming conventions used in the generated Timeloop workload YAMLs for LLM operators.

## Dimensions

| Variable | Meaning | Example (LLaMA 3.1 70B) |
|----------|---------|--------------------------|
| `B` | Batch size | 1 |
| `S` | Number of tokens processed by this operator | 2048 (prefill), 1 (decode), 128 (expert) |
| `Cin` | Input features | 8192 (hidden_size) |
| `Cout` | Output features | 8192 (num_heads * head_dim) |
| `Nh` | Number of attention heads (query heads) | 64 |
| `Sq` | Query sequence length | 2048 (prefill), 1 (decode) |
| `Sk` | Key/value sequence length | 2048 (prefill), kv_cache_len (decode) |
| `D` | Head dimension | 128 |

## Operator Templates

### Linear Projection (q/k/v/o_proj, gate/up/down_proj, router, expert MLPs, lm_head)

```
Dimensions: [B, S, Cin, Cout]

Inputs1(Cout, Cin)     — weight matrix
Inputs2(B, S, Cin)     — input activations
Outputs(B, S, Cout)    — output activations (read_write)
```

Computation: `Outputs[b, s, cout] += Inputs1[cout, cin] * Inputs2[b, s, cin]`

### Attention Q @ K^T (attn_qk)

```
Dimensions: [B, Nh, Sq, Sk, D]

Inputs1(B, Nh, Sq, D)  — query tensor
Inputs2(B, Nh, Sk, D)  — key tensor
Outputs(B, Nh, Sq, Sk) — attention scores (read_write)
```

Computation: `Outputs[b, nh, sq, sk] += Inputs1[b, nh, sq, d] * Inputs2[b, nh, sk, d]`

### Attention Scores @ V (attn_v)

```
Dimensions: [B, Nh, Sq, Sk, D]

Inputs1(B, Nh, Sq, Sk) — attention scores (after softmax)
Inputs2(B, Nh, Sk, D)  — value tensor
Outputs(B, Nh, Sq, D)  — context vectors (read_write)
```

Computation: `Outputs[b, nh, sq, d] += Inputs1[b, nh, sq, sk] * Inputs2[b, nh, sk, d]`

Note: `attn_qk` and `attn_v` share the same dimension set `[B, Nh, Sq, Sk, D]`. They differ only in which dimensions are contracted (summed over): `D` for attn_qk, `Sk` for attn_v.

## Operator Dimensions per Model

### LLaMA 3.1 (Dense)

| Operator | B | S | Cin | Cout | Description |
|----------|---|---|-----|------|-------------|
| q_proj | 1 | seq_len | H | Nh * D | Query projection |
| k_proj | 1 | seq_len | H | Nkv * D | Key projection (GQA) |
| v_proj | 1 | seq_len | H | Nkv * D | = k_proj dimensions |
| o_proj | 1 | seq_len | Nh * D | H | Output projection |
| gate_proj | 1 | seq_len | H | I | MLP gate (SwiGLU) |
| up_proj | 1 | seq_len | H | I | = gate_proj dimensions |
| down_proj | 1 | seq_len | I | H | MLP down |
| lm_head | 1 | seq_len | H | V | Vocabulary logits |

| Operator | B | Nh | Sq | Sk | D | Description |
|----------|---|----|----|-----|---|-------------|
| attn_qk | 1 | Nh | seq_len | seq_len* | D | Q @ K^T |
| attn_v | 1 | Nh | seq_len | seq_len* | D | Scores @ V |

*Decode: Sq=1, Sk=kv_cache_len

### Qwen3 MoE (additional operators)

| Operator | B | S | Cin | Cout | Description |
|----------|---|---|-----|------|-------------|
| router | 1 | seq_len | H | num_experts | Expert selection scores |
| expert_gate_proj | 1 | tokens_per_expert | H | E_I | Per-expert gate (x128) |
| expert_up_proj | 1 | tokens_per_expert | H | E_I | = expert_gate_proj dims |
| expert_down_proj | 1 | tokens_per_expert | E_I | H | Per-expert down (x128) |

Where `tokens_per_expert = B * seq_len * top_k / num_experts` (uniform routing).

## Config Parameter Mapping

| Symbol | Config field | LLaMA 8B | LLaMA 70B | Qwen3-235B |
|--------|-------------|----------|-----------|------------|
| H | hidden_size | 4096 | 8192 | 4096 |
| Nh | num_attention_heads | 32 | 64 | 64 |
| Nkv | num_key_value_heads | 8 | 8 | 4 |
| D | head_dim | 128 | 128 | 128 |
| I | intermediate_size | 14336 | 28672 | 12288 |
| V | vocab_size | 128256 | 128256 | 151936 |
| E_I | moe_intermediate_size | — | — | 1536 |

## GQA (Grouped Query Attention)

Standard MHA has equal Q, K, V heads. GQA reduces K/V heads:

| Model | Q heads (Nh) | KV heads (Nkv) | GQA ratio |
|-------|-------------|----------------|-----------|
| LLaMA 8B | 32 | 8 | 4:1 |
| LLaMA 70B | 64 | 8 | 8:1 |
| Qwen3-235B | 64 | 4 | 16:1 |

Each group of `Nh/Nkv` query heads shares one KV head. This KV data reuse is a mapping optimization for the hardware, not encoded in the problem dimensions.

## Prefill vs Decode

| Dimension | Prefill | Decode |
|-----------|---------|--------|
| S (linear ops) | seq_len (e.g., 2048) | 1 |
| Sq (attention) | seq_len | 1 |
| Sk (attention) | seq_len | kv_cache_len |

Prefill is compute-bound (all tokens processed in parallel, attention is Sq x Sk = S^2).
Decode is memory-bandwidth-bound (1 token, but reads all weights and KV cache).
