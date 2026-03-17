# LLM Operator Namespace for Timeloop

Dimension and data space naming conventions for the generated Timeloop workload YAMLs. All dimension names are single-character for Timeloop compatibility.

## Dimensions

### Linear Operators: `[B, N, C, M]`

| Var | Meaning | Example (LLaMA 3.1 70B, prefill s=2048) |
|-----|---------|------------------------------------------|
| `B` | Batch size | 1 |
| `N` | Number of tokens processed | 2048 (prefill), 1 (decode), 128 (expert) |
| `C` | Input features | 8192 (hidden_size) |
| `M` | Output features | 8192 (num_heads * head_dim) |

### Attention Operators: `[B, H, Q, K, D]`

| Var | Meaning | Example (LLaMA 3.1 70B, prefill s=2048) |
|-----|---------|------------------------------------------|
| `B` | Batch size | 1 |
| `H` | Number of attention heads | 64 |
| `Q` | Query sequence length | 2048 (prefill), 1 (decode) |
| `K` | Key/value sequence length | 2048 (prefill), kv_cache_len (decode) |
| `D` | Head dimension | 128 |

## Operator Templates

### Linear Projection (q/k/v/o_proj, gate/up/down_proj, router, expert MLPs, lm_head)

```
Dimensions: [B, N, C, M]

Inputs1(M, C)     — weight matrix
Inputs2(B, N, C)  — input activations
Outputs(B, N, M)  — output activations (read_write)
```

Computation: `Outputs[b, n, m] += Inputs1[m, c] * Inputs2[b, n, c]`

### Attention Q @ K^T (attn_qk)

```
Dimensions: [B, H, Q, K, D]

Inputs1(B, H, Q, D)  — query tensor
Inputs2(B, H, K, D)  — key tensor
Outputs(B, H, Q, K)  — attention scores (read_write)
```

Computation: `Outputs[b, h, q, k] += Inputs1[b, h, q, d] * Inputs2[b, h, k, d]`

### Attention Scores @ V (attn_v)

```
Dimensions: [B, H, Q, K, D]

Inputs1(B, H, Q, K)  — attention scores (after softmax)
Inputs2(B, H, K, D)  — value tensor
Outputs(B, H, Q, D)  — context vectors (read_write)
```

Computation: `Outputs[b, h, q, d] += Inputs1[b, h, q, k] * Inputs2[b, h, k, d]`

Note: `attn_qk` and `attn_v` share the same dimension set `[B, H, Q, K, D]`. They differ only in which dimension is contracted: `D` for attn_qk, `K` for attn_v.

## Operator Dimensions per Model

### LLaMA 3.1 (Dense)

| Operator | B | N | C | M | Description |
|----------|---|---|---|---|-------------|
| q_proj | 1 | seq_len | hidden_size | num_heads * head_dim | Query projection |
| k_proj | 1 | seq_len | hidden_size | num_kv_heads * head_dim | Key projection (GQA) |
| v_proj | 1 | seq_len | hidden_size | num_kv_heads * head_dim | = k_proj dimensions |
| o_proj | 1 | seq_len | num_heads * head_dim | hidden_size | Output projection |
| gate_proj | 1 | seq_len | hidden_size | intermediate_size | MLP gate (SwiGLU) |
| up_proj | 1 | seq_len | hidden_size | intermediate_size | = gate_proj dimensions |
| down_proj | 1 | seq_len | intermediate_size | hidden_size | MLP down |
| lm_head | 1 | seq_len | hidden_size | vocab_size | Vocabulary logits |

| Operator | B | H | Q | K | D | Description |
|----------|---|---|---|---|---|-------------|
| attn_qk | 1 | num_heads | seq_len | seq_len* | head_dim | Q @ K^T |
| attn_v | 1 | num_heads | seq_len | seq_len* | head_dim | Scores @ V |

\*Decode: Q=1, K=kv_cache_len

### Qwen3 MoE (additional operators)

| Operator | B | N | C | M | Description |
|----------|---|---|---|---|-------------|
| router | 1 | seq_len | hidden_size | num_experts | Expert selection scores |
| expert_gate_proj | 1 | tokens_per_expert | hidden_size | moe_intermediate_size | Per-expert gate (x128) |
| expert_up_proj | 1 | tokens_per_expert | hidden_size | moe_intermediate_size | = expert_gate_proj dims |
| expert_down_proj | 1 | tokens_per_expert | moe_intermediate_size | hidden_size | Per-expert down (x128) |

Where `tokens_per_expert = B * seq_len * top_k / num_experts` (uniform routing).

## Config Parameter Mapping

| Symbol | Config field | LLaMA 8B | LLaMA 70B | Qwen3-235B |
|--------|-------------|----------|-----------|------------|
| hidden_size | hidden_size | 4096 | 8192 | 4096 |
| num_heads | num_attention_heads | 32 | 64 | 64 |
| num_kv_heads | num_key_value_heads | 8 | 8 | 4 |
| head_dim | head_dim | 128 | 128 | 128 |
| intermediate_size | intermediate_size | 14336 | 28672 | 12288 |
| vocab_size | vocab_size | 128256 | 128256 | 151936 |
| moe_intermediate_size | moe_intermediate_size | — | — | 1536 |

## GQA (Grouped Query Attention)

Standard MHA has equal Q, K, V heads. GQA reduces K/V heads:

| Model | Q heads | KV heads | GQA ratio |
|-------|---------|----------|-----------|
| LLaMA 8B | 32 | 8 | 4:1 |
| LLaMA 70B | 64 | 8 | 8:1 |
| Qwen3-235B | 64 | 4 | 16:1 |

Each group of `num_heads / num_kv_heads` query heads shares one KV head. This KV data reuse is a mapping optimization for the hardware, not encoded in the problem dimensions.

## Prefill vs Decode

| Dimension | Prefill | Decode |
|-----------|---------|--------|
| N (linear ops) | seq_len (e.g., 2048) | 1 |
| Q (attention) | seq_len | 1 |
| K (attention) | seq_len | kv_cache_len |

Prefill is compute-bound (all tokens in parallel, attention is Q x K = S^2).
Decode is memory-bandwidth-bound (1 token, but reads all weights and KV cache).
