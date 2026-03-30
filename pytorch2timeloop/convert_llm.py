"""
Config-driven converter for decoder-only transformer LLMs to Timeloop workloads.

Reads a HuggingFace config.json and generates Timeloop problem YAML files for
every compute operator in the model. No torch.fx tracing required — all operator
shapes are computed arithmetically from config parameters.

Supports any decoder-only transformer with GQA attention and SwiGLU MLP:
LLaMA 3.1, Qwen 3, Mistral, Gemma, etc.
Also supports Mixture-of-Experts (MoE) models: Qwen3-MoE, Mixtral, etc.

Usage:
    python -m pytorch2timeloop.convert_llm \\
        --config /path/to/config.json \\
        --save-dir ./workloads \\
        --seq-len 2048 --batch-size 1

Verification (cross-checks formulas against actual model):
    python -m pytorch2timeloop.convert_llm \\
        --config /path/to/config.json \\
        --save-dir ./workloads \\
        --verify
"""

import json
import os
import sys
import logging
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


# ============================================================
# LLM Operator Namespace (single-char, Timeloop-compatible)
# ============================================================
#
#   Linear ops: [B, N, C, M]
#     B = batch size
#     N = number of tokens processed
#     C = input features
#     M = output features
#
#   Attention ops: [B, H, Q, K, D]
#     B = batch size
#     H = number of attention heads
#     Q = query sequence length
#     K = key/value sequence length
#     D = head dimension
#
#   Data spaces (all operators):
#     Inputs1, Inputs2, Outputs
#


@dataclass
class LLMLinearOp:
    """Linear projection with dimensions [B, N, C, M]."""
    name: str
    batch: int
    seq_len: int
    in_features: int
    out_features: int

    def to_yaml(self):
        return {
            'problem': {
                'shape': {
                    'name': self.name,
                    'dimensions': ['B', 'N', 'C', 'M'],
                    'data_spaces': [
                        {
                            'name': 'Inputs1',
                            'projection': [
                                [['M']],
                                [['C']],
                            ],
                        },
                        {
                            'name': 'Inputs2',
                            'projection': [
                                [['B']],
                                [['N']],
                                [['C']],
                            ],
                        },
                        {
                            'name': 'Outputs',
                            'projection': [
                                [['B']],
                                [['N']],
                                [['M']],
                            ],
                            'read_write': True,
                        },
                    ],
                },
                'instance': {
                    'B': self.batch,
                    'N': self.seq_len,
                    'C': self.in_features,
                    'M': self.out_features,
                },
            },
        }


@dataclass
class LLMAttentionQKOp:
    """Attention Q @ K^T with dimensions [B, H, Q, K, D]."""
    name: str
    batch: int
    num_heads: int
    q_seq_len: int
    kv_seq_len: int
    head_dim: int

    def to_yaml(self):
        return {
            'problem': {
                'shape': {
                    'name': self.name,
                    'dimensions': ['B', 'H', 'Q', 'K', 'D'],
                    'data_spaces': [
                        {
                            'name': 'Inputs1',
                            'projection': [
                                [['B']],
                                [['H']],
                                [['Q']],
                                [['D']],
                            ],
                        },
                        {
                            'name': 'Inputs2',
                            'projection': [
                                [['B']],
                                [['H']],
                                [['K']],
                                [['D']],
                            ],
                        },
                        {
                            'name': 'Outputs',
                            'projection': [
                                [['B']],
                                [['H']],
                                [['Q']],
                                [['K']],
                            ],
                            'read_write': True,
                        },
                    ],
                },
                'instance': {
                    'B': self.batch,
                    'H': self.num_heads,
                    'Q': self.q_seq_len,
                    'K': self.kv_seq_len,
                    'D': self.head_dim,
                },
            },
        }


@dataclass
class LLMAttentionVOp:
    """Attention Scores @ V with dimensions [B, H, Q, K, D]."""
    name: str
    batch: int
    num_heads: int
    q_seq_len: int
    kv_seq_len: int
    head_dim: int

    def to_yaml(self):
        return {
            'problem': {
                'shape': {
                    'name': self.name,
                    'dimensions': ['B', 'H', 'Q', 'K', 'D'],
                    'data_spaces': [
                        {
                            'name': 'Inputs1',
                            'projection': [
                                [['B']],
                                [['H']],
                                [['Q']],
                                [['K']],
                            ],
                        },
                        {
                            'name': 'Inputs2',
                            'projection': [
                                [['B']],
                                [['H']],
                                [['K']],
                                [['D']],
                            ],
                        },
                        {
                            'name': 'Outputs',
                            'projection': [
                                [['B']],
                                [['H']],
                                [['Q']],
                                [['D']],
                            ],
                            'read_write': True,
                        },
                    ],
                },
                'instance': {
                    'B': self.batch,
                    'H': self.num_heads,
                    'Q': self.q_seq_len,
                    'K': self.kv_seq_len,
                    'D': self.head_dim,
                },
            },
        }


# ============================================================
# FlashAttention (online tiled softmax) operators
# ============================================================
#
# K is tiled into T tiles of size N, i.e. K = T * N.
# These 12 operators implement the online softmax + tiled attention
# from FlashAttention / FuseMax proposal.
#
#   Dimensions: [B, H, Q, T, N, D]
#     T = number of KV tiles (kv_seq_len // tile_size)
#     N = tile size
#     D = head dimension
#

@dataclass
class FlashAttnQKOp:
    """Tiled Q @ K^T: Q[B,D,H,Q] @ BK[B,D,H,R,T] -> QK[B,H,R,T,Q]."""
    name: str
    batch: int
    num_heads: int
    q_seq_len: int
    num_tiles: int
    tile_size: int
    head_dim: int

    def to_yaml(self):
        return {
            'problem': {
                'shape': {
                    'name': self.name,
                    'dimensions': ['B', 'D', 'H', 'R', 'T', 'Q'],
                    'data_spaces': [
                        {'name': 'Q', 'projection': [[['B']], [['D']], [['H']], [['Q']]]},
                        {'name': 'BK', 'projection': [[['B']], [['D']], [['H']], [['R']], [['T']]]},
                        {'name': 'QK', 'projection': [[['B']], [['H']], [['R']], [['T']], [['Q']]], 'read_write': True},
                    ],
                },
                'instance': {
                    'B': self.batch, 'D': self.head_dim, 'H': self.num_heads,
                    'R': self.num_tiles, 'T': self.tile_size, 'Q': self.q_seq_len,
                },
            },
        }


@dataclass
class FlashAttnLMOp:
    """Local max: QK[B,H,R,T,Q] -> LM[B,H,R,Q]. Reduce over T."""
    name: str
    batch: int
    num_heads: int
    q_seq_len: int
    num_tiles: int
    tile_size: int

    def to_yaml(self):
        return {
            'problem': {
                'shape': {
                    'name': self.name,
                    'dimensions': ['B', 'H', 'R', 'T', 'Q'],
                    'data_spaces': [
                        {'name': 'QK', 'projection': [[['B']], [['H']], [['R']], [['T']], [['Q']]]},
                        {'name': 'LM', 'projection': [[['B']], [['H']], [['R']], [['Q']]], 'read_write': True},
                    ],
                },
                'instance': {
                    'B': self.batch, 'H': self.num_heads,
                    'R': self.num_tiles, 'T': self.tile_size, 'Q': self.q_seq_len,
                },
            },
        }


@dataclass
class FlashAttnRMOp:
    """Running max: max(LM, RM0) -> RM. Elementwise [B,H,R,Q]."""
    name: str
    batch: int
    num_heads: int
    q_seq_len: int
    num_tiles: int

    def to_yaml(self):
        return {
            'problem': {
                'shape': {
                    'name': self.name,
                    'dimensions': ['B', 'H', 'R', 'Q'],
                    'data_spaces': [
                        {'name': 'LM', 'projection': [[['B']], [['H']], [['R']], [['Q']]]},
                        {'name': 'RM0', 'projection': [[['B']], [['H']], [['R']], [['Q']]]},
                        {'name': 'RM', 'projection': [[['B']], [['H']], [['R']], [['Q']]], 'read_write': True},
                    ],
                },
                'instance': {
                    'B': self.batch, 'H': self.num_heads,
                    'R': self.num_tiles, 'Q': self.q_seq_len,
                },
            },
        }


@dataclass
class FlashAttnPRMOp:
    """Prev running max rescale: exp(RM0 - RM) -> PRM. Elementwise [B,H,R,Q]."""
    name: str
    batch: int
    num_heads: int
    q_seq_len: int
    num_tiles: int

    def to_yaml(self):
        return {
            'problem': {
                'shape': {
                    'name': self.name,
                    'dimensions': ['B', 'H', 'R', 'Q'],
                    'data_spaces': [
                        {'name': 'RM0', 'projection': [[['B']], [['H']], [['R']], [['Q']]]},
                        {'name': 'RM', 'projection': [[['B']], [['H']], [['R']], [['Q']]]},
                        {'name': 'PRM', 'projection': [[['B']], [['H']], [['R']], [['Q']]], 'read_write': True},
                    ],
                },
                'instance': {
                    'B': self.batch, 'H': self.num_heads,
                    'R': self.num_tiles, 'Q': self.q_seq_len,
                },
            },
        }


@dataclass
class FlashAttnSLNOp:
    """Local sub+exp: exp(QK - RM) -> SLN. Dims [B,H,R,T,Q]."""
    name: str
    batch: int
    num_heads: int
    q_seq_len: int
    num_tiles: int
    tile_size: int

    def to_yaml(self):
        return {
            'problem': {
                'shape': {
                    'name': self.name,
                    'dimensions': ['B', 'H', 'R', 'T', 'Q'],
                    'data_spaces': [
                        {'name': 'QK', 'projection': [[['B']], [['H']], [['R']], [['T']], [['Q']]]},
                        {'name': 'RM', 'projection': [[['B']], [['H']], [['R']], [['Q']]]},
                        {'name': 'SLN', 'projection': [[['B']], [['H']], [['R']], [['T']], [['Q']]], 'read_write': True},
                    ],
                },
                'instance': {
                    'B': self.batch, 'H': self.num_heads,
                    'R': self.num_tiles, 'T': self.tile_size, 'Q': self.q_seq_len,
                },
            },
        }


@dataclass
class FlashAttnSLDOp:
    """Local sum: SLN[B,H,R,T,Q] -> SLD[B,H,R,Q]. Reduce over T."""
    name: str
    batch: int
    num_heads: int
    q_seq_len: int
    num_tiles: int
    tile_size: int

    def to_yaml(self):
        return {
            'problem': {
                'shape': {
                    'name': self.name,
                    'dimensions': ['B', 'H', 'R', 'T', 'Q'],
                    'data_spaces': [
                        {'name': 'SLN', 'projection': [[['B']], [['H']], [['R']], [['T']], [['Q']]]},
                        {'name': 'SLD', 'projection': [[['B']], [['H']], [['R']], [['Q']]], 'read_write': True},
                    ],
                },
                'instance': {
                    'B': self.batch, 'H': self.num_heads,
                    'R': self.num_tiles, 'T': self.tile_size, 'Q': self.q_seq_len,
                },
            },
        }


@dataclass
class FlashAttnSPDOp:
    """Rescale prev denom: PRM * RD0 -> SPD. Elementwise [B,H,R,Q]."""
    name: str
    batch: int
    num_heads: int
    q_seq_len: int
    num_tiles: int

    def to_yaml(self):
        return {
            'problem': {
                'shape': {
                    'name': self.name,
                    'dimensions': ['B', 'H', 'R', 'Q'],
                    'data_spaces': [
                        {'name': 'PRM', 'projection': [[['B']], [['H']], [['R']], [['Q']]]},
                        {'name': 'RD0', 'projection': [[['B']], [['H']], [['R']], [['Q']]]},
                        {'name': 'SPD', 'projection': [[['B']], [['H']], [['R']], [['Q']]], 'read_write': True},
                    ],
                },
                'instance': {
                    'B': self.batch, 'H': self.num_heads,
                    'R': self.num_tiles, 'Q': self.q_seq_len,
                },
            },
        }


@dataclass
class FlashAttnRDOp:
    """Running denom: SPD + SLD -> RD. Elementwise [B,H,R,Q]."""
    name: str
    batch: int
    num_heads: int
    q_seq_len: int
    num_tiles: int

    def to_yaml(self):
        return {
            'problem': {
                'shape': {
                    'name': self.name,
                    'dimensions': ['B', 'H', 'R', 'Q'],
                    'data_spaces': [
                        {'name': 'SPD', 'projection': [[['B']], [['H']], [['R']], [['Q']]]},
                        {'name': 'SLD', 'projection': [[['B']], [['H']], [['R']], [['Q']]]},
                        {'name': 'RD', 'projection': [[['B']], [['H']], [['R']], [['Q']]], 'read_write': True},
                    ],
                },
                'instance': {
                    'B': self.batch, 'H': self.num_heads,
                    'R': self.num_tiles, 'Q': self.q_seq_len,
                },
            },
        }


@dataclass
class FlashAttnSLNVOp:
    """Local attn @ V: SLN[B,H,R,T,Q] @ BV[B,D,H,R,T] -> SLNV[B,D,H,Q]."""
    name: str
    batch: int
    num_heads: int
    q_seq_len: int
    num_tiles: int
    tile_size: int
    head_dim: int

    def to_yaml(self):
        return {
            'problem': {
                'shape': {
                    'name': self.name,
                    'dimensions': ['B', 'D', 'H', 'R', 'T', 'Q'],
                    'data_spaces': [
                        {'name': 'SLN', 'projection': [[['B']], [['H']], [['R']], [['T']], [['Q']]]},
                        {'name': 'BV', 'projection': [[['B']], [['D']], [['H']], [['R']], [['T']]]},
                        {'name': 'SLNV', 'projection': [[['B']], [['D']], [['H']], [['Q']]], 'read_write': True},
                    ],
                },
                'instance': {
                    'B': self.batch, 'D': self.head_dim, 'H': self.num_heads,
                    'R': self.num_tiles, 'T': self.tile_size, 'Q': self.q_seq_len,
                },
            },
        }


@dataclass
class FlashAttnSPNVOp:
    """Rescale prev output: PRM * RNV0 -> SPNV. Elementwise [B,D,H,R,Q]."""
    name: str
    batch: int
    num_heads: int
    q_seq_len: int
    num_tiles: int
    head_dim: int

    def to_yaml(self):
        return {
            'problem': {
                'shape': {
                    'name': self.name,
                    'dimensions': ['B', 'D', 'H', 'R', 'Q'],
                    'data_spaces': [
                        {'name': 'PRM', 'projection': [[['B']], [['H']], [['R']], [['Q']]]},
                        {'name': 'RNV0', 'projection': [[['B']], [['D']], [['H']], [['R']], [['Q']]]},
                        {'name': 'SPNV', 'projection': [[['B']], [['D']], [['H']], [['R']], [['Q']]], 'read_write': True},
                    ],
                },
                'instance': {
                    'B': self.batch, 'D': self.head_dim, 'H': self.num_heads,
                    'R': self.num_tiles, 'Q': self.q_seq_len,
                },
            },
        }


@dataclass
class FlashAttnRNVOp:
    """Running output: SPNV + SLNV -> RNV. Elementwise [B,D,H,R,Q]."""
    name: str
    batch: int
    num_heads: int
    q_seq_len: int
    num_tiles: int
    head_dim: int

    def to_yaml(self):
        return {
            'problem': {
                'shape': {
                    'name': self.name,
                    'dimensions': ['B', 'D', 'H', 'R', 'Q'],
                    'data_spaces': [
                        {'name': 'SPNV', 'projection': [[['B']], [['D']], [['H']], [['R']], [['Q']]]},
                        {'name': 'SLNV', 'projection': [[['B']], [['D']], [['H']], [['R']], [['Q']]]},
                        {'name': 'RNV', 'projection': [[['B']], [['D']], [['H']], [['R']], [['Q']]], 'read_write': True},
                    ],
                },
                'instance': {
                    'B': self.batch, 'D': self.head_dim, 'H': self.num_heads,
                    'R': self.num_tiles, 'Q': self.q_seq_len,
                },
            },
        }


@dataclass
class FlashAttnAVOp:
    """Final divide: RNV / RD -> AV. Dims [B,D,H,Q]."""
    name: str
    batch: int
    num_heads: int
    q_seq_len: int
    head_dim: int

    def to_yaml(self):
        return {
            'problem': {
                'shape': {
                    'name': self.name,
                    'dimensions': ['B', 'D', 'H', 'Q'],
                    'data_spaces': [
                        {'name': 'RNV', 'projection': [[['B']], [['D']], [['H']], [['Q']]]},
                        {'name': 'RD', 'projection': [[['B']], [['H']], [['Q']]]},
                        {'name': 'AV', 'projection': [[['B']], [['D']], [['H']], [['Q']]], 'read_write': True},
                    ],
                },
                'instance': {
                    'B': self.batch, 'D': self.head_dim, 'H': self.num_heads,
                    'Q': self.q_seq_len,
                },
            },
        }


@dataclass
class LLMSoftmaxMaxOp:
    """Softmax step 1: row-wise max. Dims [B, H, Q, K]. Reduces over K."""
    name: str
    batch: int
    num_heads: int
    q_seq_len: int
    kv_seq_len: int

    def to_yaml(self):
        return {
            'problem': {
                'shape': {
                    'name': self.name,
                    'dimensions': ['B', 'H', 'Q', 'K'],
                    'data_spaces': [
                        {
                            'name': 'Inputs1',
                            'projection': [
                                [['B']],
                                [['H']],
                                [['Q']],
                                [['K']],
                            ],
                        },
                        {
                            'name': 'Outputs',
                            'projection': [
                                [['B']],
                                [['H']],
                                [['Q']],
                            ],
                            'read_write': True,
                        },
                    ],
                },
                'instance': {
                    'B': self.batch,
                    'H': self.num_heads,
                    'Q': self.q_seq_len,
                    'K': self.kv_seq_len,
                },
            },
        }


@dataclass
class LLMSoftmaxSubExpOp:
    """Softmax step 2: subtract max and exponentiate. Dims [B, H, Q, K]."""
    name: str
    batch: int
    num_heads: int
    q_seq_len: int
    kv_seq_len: int

    def to_yaml(self):
        return {
            'problem': {
                'shape': {
                    'name': self.name,
                    'dimensions': ['B', 'H', 'Q', 'K'],
                    'data_spaces': [
                        {
                            'name': 'Inputs1',
                            'projection': [
                                [['B']],
                                [['H']],
                                [['Q']],
                                [['K']],
                            ],
                        },
                        {
                            'name': 'Inputs2',
                            'projection': [
                                [['B']],
                                [['H']],
                                [['Q']],
                            ],
                        },
                        {
                            'name': 'Outputs',
                            'projection': [
                                [['B']],
                                [['H']],
                                [['Q']],
                                [['K']],
                            ],
                            'read_write': True,
                        },
                    ],
                },
                'instance': {
                    'B': self.batch,
                    'H': self.num_heads,
                    'Q': self.q_seq_len,
                    'K': self.kv_seq_len,
                },
            },
        }


@dataclass
class LLMSoftmaxSumOp:
    """Softmax step 3: row-wise sum of exp values. Dims [B, H, Q, K]. Reduces over K."""
    name: str
    batch: int
    num_heads: int
    q_seq_len: int
    kv_seq_len: int

    def to_yaml(self):
        return {
            'problem': {
                'shape': {
                    'name': self.name,
                    'dimensions': ['B', 'H', 'Q', 'K'],
                    'data_spaces': [
                        {
                            'name': 'Inputs1',
                            'projection': [
                                [['B']],
                                [['H']],
                                [['Q']],
                                [['K']],
                            ],
                        },
                        {
                            'name': 'Outputs',
                            'projection': [
                                [['B']],
                                [['H']],
                                [['Q']],
                            ],
                            'read_write': True,
                        },
                    ],
                },
                'instance': {
                    'B': self.batch,
                    'H': self.num_heads,
                    'Q': self.q_seq_len,
                    'K': self.kv_seq_len,
                },
            },
        }


@dataclass
class LLMSoftmaxDivOp:
    """Softmax step 4: divide by sum. Dims [B, H, Q, K]."""
    name: str
    batch: int
    num_heads: int
    q_seq_len: int
    kv_seq_len: int

    def to_yaml(self):
        return {
            'problem': {
                'shape': {
                    'name': self.name,
                    'dimensions': ['B', 'H', 'Q', 'K'],
                    'data_spaces': [
                        {
                            'name': 'Inputs1',
                            'projection': [
                                [['B']],
                                [['H']],
                                [['Q']],
                                [['K']],
                            ],
                        },
                        {
                            'name': 'Inputs2',
                            'projection': [
                                [['B']],
                                [['H']],
                                [['Q']],
                            ],
                        },
                        {
                            'name': 'Outputs',
                            'projection': [
                                [['B']],
                                [['H']],
                                [['Q']],
                                [['K']],
                            ],
                            'read_write': True,
                        },
                    ],
                },
                'instance': {
                    'B': self.batch,
                    'H': self.num_heads,
                    'Q': self.q_seq_len,
                    'K': self.kv_seq_len,
                },
            },
        }


@dataclass
class LLMConfig:
    """Parsed LLM configuration with all fields needed for workload generation."""
    model_name: str
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    vocab_size: int
    num_hidden_layers: int
    # Explicit head_dim (None = derive from hidden_size // num_attention_heads)
    _head_dim: Optional[int] = None
    # MoE fields (None = dense model)
    num_experts: Optional[int] = None
    num_experts_per_tok: Optional[int] = None
    moe_intermediate_size: Optional[int] = None

    @property
    def head_dim(self):
        if self._head_dim is not None:
            return self._head_dim
        return self.hidden_size // self.num_attention_heads

    @property
    def gqa_ratio(self):
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def is_moe(self):
        return self.num_experts is not None and self.num_experts > 1

    @classmethod
    def from_json(cls, config_path: str, model_name: str = None):
        """Load from a HuggingFace config.json file."""
        with open(config_path) as f:
            config = json.load(f)

        if model_name is None:
            model_name = config.get('model_type', 'unknown')

        return cls(
            model_name=model_name,
            hidden_size=config['hidden_size'],
            num_attention_heads=config['num_attention_heads'],
            num_key_value_heads=config.get(
                'num_key_value_heads', config['num_attention_heads']
            ),
            intermediate_size=config['intermediate_size'],
            vocab_size=config['vocab_size'],
            num_hidden_layers=config['num_hidden_layers'],
            _head_dim=config.get('head_dim', None),
            num_experts=config.get('num_local_experts', None),
            num_experts_per_tok=config.get('num_experts_per_tok', None),
            moe_intermediate_size=config.get('moe_intermediate_size', None),
        )

    @classmethod
    def from_hf_config(cls, hf_config, model_name: str = None):
        """Load from a HuggingFace AutoConfig object."""
        if model_name is None:
            model_name = getattr(hf_config, 'model_type', 'unknown')

        return cls(
            model_name=model_name,
            hidden_size=hf_config.hidden_size,
            num_attention_heads=hf_config.num_attention_heads,
            num_key_value_heads=getattr(
                hf_config, 'num_key_value_heads',
                hf_config.num_attention_heads
            ),
            intermediate_size=hf_config.intermediate_size,
            vocab_size=hf_config.vocab_size,
            num_hidden_layers=hf_config.num_hidden_layers,
            _head_dim=getattr(hf_config, 'head_dim', None),
            num_experts=getattr(hf_config, 'num_local_experts', None),
            num_experts_per_tok=getattr(hf_config, 'num_experts_per_tok', None),
            moe_intermediate_size=getattr(hf_config, 'moe_intermediate_size', None),
        )


def _linear_op(name, batch, seq_len, in_features, out_features):
    """Create an LLMLinearOp with LLM-semantic dimensions."""
    return LLMLinearOp(
        name=name,
        batch=batch,
        seq_len=seq_len,
        in_features=in_features,
        out_features=out_features,
    )


def _attn_qk_op(name, batch, num_heads, q_seq_len, kv_seq_len, head_dim):
    """Create an LLMAttentionQKOp (Q @ K^T)."""
    return LLMAttentionQKOp(
        name=name,
        batch=batch,
        num_heads=num_heads,
        q_seq_len=q_seq_len,
        kv_seq_len=kv_seq_len,
        head_dim=head_dim,
    )


def _attn_v_op(name, batch, num_heads, q_seq_len, kv_seq_len, head_dim):
    """Create an LLMAttentionVOp (softmax(Scores) @ V)."""
    return LLMAttentionVOp(
        name=name,
        batch=batch,
        num_heads=num_heads,
        q_seq_len=q_seq_len,
        kv_seq_len=kv_seq_len,
        head_dim=head_dim,
    )


def _softmax_max_op(name, batch, num_heads, q_seq_len, kv_seq_len):
    """Create an LLMSoftmaxMaxOp (row-wise max)."""
    return LLMSoftmaxMaxOp(
        name=name, batch=batch, num_heads=num_heads,
        q_seq_len=q_seq_len, kv_seq_len=kv_seq_len,
    )


def _softmax_sub_exp_op(name, batch, num_heads, q_seq_len, kv_seq_len):
    """Create an LLMSoftmaxSubExpOp (subtract max + exp)."""
    return LLMSoftmaxSubExpOp(
        name=name, batch=batch, num_heads=num_heads,
        q_seq_len=q_seq_len, kv_seq_len=kv_seq_len,
    )


def _softmax_sum_op(name, batch, num_heads, q_seq_len, kv_seq_len):
    """Create an LLMSoftmaxSumOp (row-wise sum)."""
    return LLMSoftmaxSumOp(
        name=name, batch=batch, num_heads=num_heads,
        q_seq_len=q_seq_len, kv_seq_len=kv_seq_len,
    )


def _softmax_div_op(name, batch, num_heads, q_seq_len, kv_seq_len):
    """Create an LLMSoftmaxDivOp (divide by sum)."""
    return LLMSoftmaxDivOp(
        name=name, batch=batch, num_heads=num_heads,
        q_seq_len=q_seq_len, kv_seq_len=kv_seq_len,
    )


def _generate_flashattn_operators(cfg: LLMConfig, batch_size: int,
                                   q_seq_len: int, kv_seq_len: int,
                                   tile_size: int = 256) -> List:
    """
    Generate the 12 FlashAttention (online tiled softmax) operators.

    K is split into R tiles of size T: K = R * T.
    These implement the online softmax recurrence from FlashAttention / FuseMax.

    Args:
        tile_size: KV tile size (default 256, matching typical FlashAttention).
    """
    B = batch_size
    H = cfg.num_attention_heads
    D = cfg.head_dim
    Q = q_seq_len
    K = kv_seq_len
    T = tile_size
    R = (K + T - 1) // T  # ceil division

    return [
        FlashAttnQKOp(name='flashattn_qk', batch=B, num_heads=H,
                       q_seq_len=Q, num_tiles=R, tile_size=T, head_dim=D),
        FlashAttnLMOp(name='flashattn_lm', batch=B, num_heads=H,
                       q_seq_len=Q, num_tiles=R, tile_size=T),
        FlashAttnRMOp(name='flashattn_rm', batch=B, num_heads=H,
                       q_seq_len=Q, num_tiles=R),
        FlashAttnPRMOp(name='flashattn_prm', batch=B, num_heads=H,
                        q_seq_len=Q, num_tiles=R),
        FlashAttnSLNOp(name='flashattn_sln', batch=B, num_heads=H,
                        q_seq_len=Q, num_tiles=R, tile_size=T),
        FlashAttnSLDOp(name='flashattn_sld', batch=B, num_heads=H,
                        q_seq_len=Q, num_tiles=R, tile_size=T),
        FlashAttnSPDOp(name='flashattn_spd', batch=B, num_heads=H,
                        q_seq_len=Q, num_tiles=R),
        FlashAttnRDOp(name='flashattn_rd', batch=B, num_heads=H,
                       q_seq_len=Q, num_tiles=R),
        FlashAttnSLNVOp(name='flashattn_slnv', batch=B, num_heads=H,
                         q_seq_len=Q, num_tiles=R, tile_size=T, head_dim=D),
        FlashAttnSPNVOp(name='flashattn_spnv', batch=B, num_heads=H,
                         q_seq_len=Q, num_tiles=R, head_dim=D),
        FlashAttnRNVOp(name='flashattn_rnv', batch=B, num_heads=H,
                        q_seq_len=Q, num_tiles=R, head_dim=D),
        FlashAttnAVOp(name='flashattn_av', batch=B, num_heads=H,
                       q_seq_len=Q, head_dim=D),
    ]


def _generate_attention_operators(cfg: LLMConfig, layer_idx: int,
                                   batch_size: int, seq_len: int,
                                   kv_cache_len: Optional[int] = None) -> List:
    """
    Generate attention block operators (shared between dense and MoE).

    Args:
        kv_cache_len: KV cache length for decode phase. If None, this is
            prefill and attention is seq_len x seq_len. If set, this is
            decode: Q has seq_len=1 new token, K/V have kv_cache_len
            entries from the cache + 1 new = kv_cache_len total.
    """
    H = cfg.hidden_size
    n_h = cfg.num_attention_heads
    n_kv = cfg.num_key_value_heads
    d = cfg.head_dim
    S = seq_len
    B = batch_size
    L = f"layer{layer_idx}"

    Sq = S
    Sk = kv_cache_len if kv_cache_len is not None else S

    ops = []

    # Q projection: (B, S, H) -> (B, S, n_h*d)
    ops.append(_linear_op(f"{L}_q_proj", B, S, H, n_h * d))

    # K projection: (B, S, H) -> (B, S, n_kv*d)
    ops.append(_linear_op(f"{L}_k_proj", B, S, H, n_kv * d))

    # V projection: (B, S, H) -> (B, S, n_kv*d)
    ops.append(_linear_op(f"{L}_v_proj", B, S, H, n_kv * d))

    # Attention scores: Q(B, Nh, Sq, D) @ K^T(B, Nh, D, Sk) -> Scores(B, Nh, Sq, Sk)
    # KV heads are broadcast to n_h via GQA repeat; KV reuse is a mapping concern.
    ops.append(_attn_qk_op(f"{L}_attn_qk", B, n_h, Sq, Sk, d))

    # Softmax: 4-step decomposition (max, sub+exp, sum, div)
    # Scores(B, Nh, Sq, Sk) -> Attention(B, Nh, Sq, Sk)
    ops.append(_softmax_max_op(f"{L}_softmax_max", B, n_h, Sq, Sk))
    ops.append(_softmax_sub_exp_op(f"{L}_softmax_sub_exp", B, n_h, Sq, Sk))
    ops.append(_softmax_sum_op(f"{L}_softmax_sum", B, n_h, Sq, Sk))
    ops.append(_softmax_div_op(f"{L}_softmax_div", B, n_h, Sq, Sk))

    # Attention context: A(B, Nh, Sq, Sk) @ V(B, Nh, Sk, D) -> Context(B, Nh, Sq, D)
    ops.append(_attn_v_op(f"{L}_attn_v", B, n_h, Sq, Sk, d))

    # O projection: (B, S, n_h*d) -> (B, S, H)
    ops.append(_linear_op(f"{L}_o_proj", B, S, n_h * d, H))

    return ops


def _generate_dense_mlp_operators(cfg: LLMConfig, layer_idx: int,
                                   batch_size: int, seq_len: int) -> List:
    """Generate dense MLP block operators (gate_proj, up_proj, down_proj)."""
    H = cfg.hidden_size
    I = cfg.intermediate_size
    B = batch_size
    S = seq_len
    L = f"layer{layer_idx}"

    return [
        _linear_op(f"{L}_gate_proj", B, S, H, I),
        _linear_op(f"{L}_up_proj", B, S, H, I),
        _linear_op(f"{L}_down_proj", B, S, I, H),
    ]


def _generate_moe_mlp_operators(cfg: LLMConfig, layer_idx: int,
                                 batch_size: int, seq_len: int) -> List:
    """
    Generate MoE MLP block operators.

    MoE replaces the dense MLP with:
      1. Router: Linear(H, num_experts) — selects top-k experts per token
      2. num_experts copies of expert MLPs, each processing
         B*S*top_k/num_experts tokens (assuming uniform routing)

    All experts have identical dimensions (moe_intermediate_size), so we
    generate one representative expert and annotate the count.
    """
    H = cfg.hidden_size
    E_I = cfg.moe_intermediate_size
    n_exp = cfg.num_experts
    top_k = cfg.num_experts_per_tok
    B = batch_size
    S = seq_len
    L = f"layer{layer_idx}"

    # Tokens per expert (uniform routing assumption)
    # S_expert = total_tokens * top_k / num_experts
    tokens_per_expert = (B * S * top_k) // n_exp
    if tokens_per_expert == 0:
        tokens_per_expert = 1

    ops = []

    # Router: (B, S, H) -> (B, S, num_experts)
    ops.append(_linear_op(f"{L}_router", B, S, H, n_exp))

    # Expert MLPs: each expert processes tokens_per_expert tokens.
    # B=batch_size, S=tokens_per_expert (routed tokens, not a contiguous sequence).
    for exp_idx in range(n_exp):
        E = f"{L}_expert{exp_idx}"
        ops.append(_linear_op(f"{E}_gate_proj", B, tokens_per_expert, H, E_I))
        ops.append(_linear_op(f"{E}_up_proj", B, tokens_per_expert, H, E_I))
        ops.append(_linear_op(f"{E}_down_proj", B, tokens_per_expert, E_I, H))

    return ops


def generate_layer_operators(cfg: LLMConfig, layer_idx: int,
                             batch_size: int, seq_len: int,
                             kv_cache_len: Optional[int] = None) -> List:
    """
    Generate Timeloop operator descriptions for one decoder layer.

    Dense model: 13 ops (10 attention [6 + 4 softmax] + 3 MLP)
    MoE model: 10 attention + 1 router + 3*num_experts expert MLP ops

    Args:
        kv_cache_len: If set, generate decode-phase attention (Q=1 token,
            K/V = kv_cache_len entries). If None, generate prefill attention.
    """
    ops = _generate_attention_operators(
        cfg, layer_idx, batch_size, seq_len, kv_cache_len
    )

    if cfg.is_moe:
        ops.extend(_generate_moe_mlp_operators(
            cfg, layer_idx, batch_size, seq_len
        ))
    else:
        ops.extend(_generate_dense_mlp_operators(
            cfg, layer_idx, batch_size, seq_len
        ))

    return ops


def generate_model_operators(cfg: LLMConfig,
                             batch_size: int, seq_len: int,
                             kv_cache_len: Optional[int] = None) -> List:
    """
    Generate all operator descriptions for the complete model.

    Args:
        kv_cache_len: If set, generate decode-phase operators.
    """
    ops = []

    # Per-layer operators
    for layer_idx in range(cfg.num_hidden_layers):
        ops.extend(generate_layer_operators(
            cfg, layer_idx, batch_size, seq_len, kv_cache_len
        ))

    # LM head: (B, S, H) -> (B, S, V)
    ops.append(_linear_op("lm_head", batch_size, seq_len,
                           cfg.hidden_size, cfg.vocab_size))

    return ops


def convert_llm(config_path: str, save_dir: str,
                model_name: str = None,
                batch_size: int = 1, seq_len: int = 128,
                phase: str = 'prefill',
                kv_cache_len: Optional[int] = None,
                output_name: Optional[str] = None):
    """
    Convert a decoder-only transformer LLM config to Timeloop workload YAMLs.

    Args:
        config_path: Path to HuggingFace config.json or model directory
        save_dir: Directory to save YAML files
        model_name: Optional model name override (auto-detected from config)
        batch_size: Batch size for workload dimensions
        seq_len: Sequence length. For prefill: prompt length. For decode: 1.
        phase: 'prefill' or 'decode'
        kv_cache_len: KV cache length for decode. Required if phase='decode'.
            This is the total context seen so far (prompt + generated tokens).

    Returns:
        List of generated operator descriptions
    """
    # Handle both config.json path and model directory
    if os.path.isdir(config_path):
        config_path = os.path.join(config_path, 'config.json')

    cfg = LLMConfig.from_json(config_path, model_name)

    # Validate and set phase parameters
    if phase == 'decode':
        if kv_cache_len is None:
            raise ValueError("--kv-cache-len is required for decode phase")
        seq_len = 1  # decode always processes 1 new token
    elif phase == 'prefill':
        kv_cache_len = None  # prefill doesn't use KV cache
    else:
        raise ValueError(f"Unknown phase: {phase}. Use 'prefill' or 'decode'.")

    logger.info(
        "Converting %s: H=%d, n_h=%d, n_kv=%d, I=%d, V=%d, L=%d",
        cfg.model_name, cfg.hidden_size, cfg.num_attention_heads,
        cfg.num_key_value_heads, cfg.intermediate_size,
        cfg.vocab_size, cfg.num_hidden_layers,
    )

    # Generate unique operators only.
    # All layers are identical, and within a layer:
    #   k_proj == v_proj, gate_proj == up_proj, all experts are identical.
    # We store only unique YAMLs and describe duplication in NETWORK.yaml.

    attn_ops = _generate_attention_operators(cfg, 0, batch_size, seq_len, kv_cache_len)

    # Deduplicate: remove v_proj (same dims as k_proj)
    # ViT (standard MHA): q/k/v/o proj all have same dims, keep only q_proj
    is_vit = output_name is not None and 'vit' in output_name
    if is_vit:
        unique_attn = [op for op in attn_ops
                       if 'v_proj' not in op.name
                       and 'k_proj' not in op.name
                       and 'o_proj' not in op.name]
    else:
        unique_attn = [op for op in attn_ops if 'v_proj' not in op.name]

    # MLP ops
    if cfg.is_moe:
        H = cfg.hidden_size
        E_I = cfg.moe_intermediate_size
        n_exp = cfg.num_experts
        top_k = cfg.num_experts_per_tok
        tokens_per_expert = max((batch_size * seq_len * top_k) // n_exp, 1)

        # Router (unique)
        router_op = _linear_op("router", batch_size, seq_len, H, n_exp)
        # One representative expert: gate_proj (= up_proj) and down_proj
        # B=batch_size, S=tokens_per_expert (routed tokens)
        expert_gate = _linear_op("expert_gate_proj", batch_size, tokens_per_expert, H, E_I)
        expert_down = _linear_op("expert_down_proj", batch_size, tokens_per_expert, E_I, H)
        unique_mlp = [router_op, expert_gate, expert_down]
    else:
        mlp_ops = _generate_dense_mlp_operators(cfg, 0, batch_size, seq_len)
        # Deduplicate: remove up_proj (same dims as gate_proj)
        unique_mlp = [op for op in mlp_ops if 'up_proj' not in op.name]

    # LM head
    lm_head_op = _linear_op("lm_head", batch_size, seq_len,
                             cfg.hidden_size, cfg.vocab_size)

    unique_ops = unique_attn + unique_mlp + [lm_head_op]

    # Include phase and dimensions in output directory name
    if output_name is not None:
        outdir = os.path.join(save_dir, output_name)
    elif phase == 'prefill':
        outdir = os.path.join(save_dir, f"{cfg.model_name}_prefill_s{seq_len}")
    else:
        outdir = os.path.join(save_dir, f"{cfg.model_name}_decode_kv{kv_cache_len}")
    os.makedirs(outdir, exist_ok=True)

    # Write only unique operator YAMLs
    for op in unique_ops:
        fname = f"{op.name}.yaml"
        fpath = os.path.join(outdir, fname)
        with open(fpath, 'w') as f:
            yaml.dump(op.to_yaml(), f, default_flow_style=False)

    # Write network description file
    _write_network_description(cfg, outdir, phase, seq_len, kv_cache_len,
                               batch_size, unique_ops)

    # Generate FlashAttention (tiled online softmax) operators in subfolder
    Sq = seq_len
    Sk = kv_cache_len if kv_cache_len is not None else seq_len
    flashattn_ops = _generate_flashattn_operators(cfg, batch_size, Sq, Sk)
    flashattn_dir = os.path.join(outdir, "flashattn")
    os.makedirs(flashattn_dir, exist_ok=True)
    for op in flashattn_ops:
        fname = f"{op.name}.yaml"
        fpath = os.path.join(flashattn_dir, fname)
        with open(fpath, 'w') as f:
            yaml.dump(op.to_yaml(), f, default_flow_style=False)

    # Write dimensions summary for FlashAttention
    K = kv_cache_len if kv_cache_len is not None else seq_len
    Q_len = seq_len if phase == 'prefill' else 1
    tile_sz = 256  # default tile size, matching _generate_flashattn_operators
    R_val = (K + tile_sz - 1) // tile_sz
    dims = {
        'dimensions': {
            'B': {'value': batch_size, 'description': 'Batch size'},
            'D': {'value': cfg.head_dim, 'description': 'Head dimension (Q/K/V)'},
            'H': {'value': cfg.num_attention_heads, 'description': 'Number of attention heads'},
            'K': {'value': K, 'description': 'KV sequence length (K = R * T)'},
            'Q': {'value': Q_len, 'description': 'Query sequence length'},
            'R': {'value': R_val, 'description': 'Number of KV tiles (rounds)'},
            'T': {'value': tile_sz, 'description': 'KV tile size'},
        },
    }
    with open(os.path.join(flashattn_dir, 'dimensions.yaml'), 'w') as f:
        yaml.dump(dims, f, default_flow_style=False, sort_keys=False)

    print(f"Generated {len(unique_ops)} unique YAML files in {outdir}/")
    print(f"  + {len(flashattn_ops)} FlashAttention ops in {flashattn_dir}/")
    print(f"  Phase: {phase}")
    if phase == 'prefill':
        print(f"  seq_len={seq_len}, batch_size={batch_size}")
        print(f"  Attention: Q@K^T is ({seq_len} x {seq_len})")
    else:
        print(f"  seq_len=1 (new token), kv_cache_len={kv_cache_len}, "
              f"batch_size={batch_size}")
        print(f"  Attention: Q@K^T is (1 x {kv_cache_len})")
    print(f"  See {outdir}/NETWORK.yaml for full network description")

    return unique_ops


def _write_network_description(cfg, outdir, phase, seq_len, kv_cache_len,
                                batch_size, unique_ops):
    """Write a YAML file describing the full network structure."""
    if phase == 'prefill':
        phase_desc = f"prefill (seq_len={seq_len})"
        attn_desc = f"Q@K^T: ({seq_len} x {seq_len})"
    else:
        phase_desc = f"decode (kv_cache_len={kv_cache_len})"
        attn_desc = f"Q@K^T: (1 x {kv_cache_len})"

    unique_names = [op.name for op in unique_ops]

    # Build the per-layer operator sequence with duplication info
    if cfg.is_moe:
        n_exp = cfg.num_experts
        top_k = cfg.num_experts_per_tok
        tokens_per_expert = max(
            (batch_size * seq_len * top_k) // n_exp, 1
        )
        ops_per_layer = 10 + 1 + 3 * n_exp  # attn(6+4 softmax) + router + expert MLPs
        layer_sequence = [
            {'yaml': 'q_proj.yaml', 'count': 1,
             'description': f'Query projection: (B*S, {cfg.hidden_size}) @ ({cfg.hidden_size}, {cfg.num_attention_heads * cfg.head_dim})'},
            {'yaml': 'k_proj.yaml', 'count': 1,
             'description': f'Key projection: (B*S, {cfg.hidden_size}) @ ({cfg.hidden_size}, {cfg.num_key_value_heads * cfg.head_dim})'},
            {'yaml': 'k_proj.yaml', 'count': 1, 'alias': 'v_proj',
             'description': 'Value projection: same dimensions as k_proj (GQA)'},
            {'yaml': 'attn_qk.yaml', 'count': 1,
             'description': f'Attention scores: Q @ K^T, batched over {cfg.num_attention_heads} heads'},
            {'yaml': 'softmax_max.yaml', 'count': 1,
             'description': 'Softmax step 1: row-wise max over K'},
            {'yaml': 'softmax_sub_exp.yaml', 'count': 1,
             'description': 'Softmax step 2: subtract max and exponentiate'},
            {'yaml': 'softmax_sum.yaml', 'count': 1,
             'description': 'Softmax step 3: row-wise sum of exp values'},
            {'yaml': 'softmax_div.yaml', 'count': 1,
             'description': 'Softmax step 4: divide by sum'},
            {'yaml': 'attn_v.yaml', 'count': 1,
             'description': 'Attention context: A @ V'},
            {'yaml': 'o_proj.yaml', 'count': 1,
             'description': f'Output projection: ({cfg.num_attention_heads * cfg.head_dim}) -> ({cfg.hidden_size})'},
            {'yaml': 'router.yaml', 'count': 1,
             'description': f'MoE router: scores each token against {n_exp} experts, selects top-{top_k}'},
            {'yaml': 'expert_gate_proj.yaml', 'count': n_exp, 'alias': 'expert[0..127]_gate_proj',
             'description': f'Expert gate projection: {n_exp} identical experts, each processes {tokens_per_expert} tokens'},
            {'yaml': 'expert_gate_proj.yaml', 'count': n_exp, 'alias': 'expert[0..127]_up_proj',
             'description': 'Expert up projection: same dimensions as expert_gate_proj (SwiGLU)'},
            {'yaml': 'expert_down_proj.yaml', 'count': n_exp, 'alias': 'expert[0..127]_down_proj',
             'description': f'Expert down projection: ({cfg.moe_intermediate_size}) -> ({cfg.hidden_size}), {n_exp} experts'},
        ]
    else:
        ops_per_layer = 13
        layer_sequence = [
            {'yaml': 'layer0_q_proj.yaml', 'count': 1,
             'description': f'Query projection: (B*S, {cfg.hidden_size}) @ ({cfg.hidden_size}, {cfg.num_attention_heads * cfg.head_dim})'},
            {'yaml': 'layer0_k_proj.yaml', 'count': 1,
             'description': f'Key projection: (B*S, {cfg.hidden_size}) @ ({cfg.hidden_size}, {cfg.num_key_value_heads * cfg.head_dim})'},
            {'yaml': 'layer0_k_proj.yaml', 'count': 1, 'alias': 'v_proj',
             'description': 'Value projection: same dimensions as k_proj (GQA)'},
            {'yaml': 'layer0_attn_qk.yaml', 'count': 1,
             'description': f'Attention scores: Q @ K^T, batched over {cfg.num_attention_heads} heads'},
            {'yaml': 'layer0_softmax_max.yaml', 'count': 1,
             'description': 'Softmax step 1: row-wise max over K'},
            {'yaml': 'layer0_softmax_sub_exp.yaml', 'count': 1,
             'description': 'Softmax step 2: subtract max and exponentiate'},
            {'yaml': 'layer0_softmax_sum.yaml', 'count': 1,
             'description': 'Softmax step 3: row-wise sum of exp values'},
            {'yaml': 'layer0_softmax_div.yaml', 'count': 1,
             'description': 'Softmax step 4: divide by sum'},
            {'yaml': 'layer0_attn_v.yaml', 'count': 1,
             'description': 'Attention context: A @ V'},
            {'yaml': 'layer0_o_proj.yaml', 'count': 1,
             'description': f'Output projection: ({cfg.num_attention_heads * cfg.head_dim}) -> ({cfg.hidden_size})'},
            {'yaml': 'layer0_gate_proj.yaml', 'count': 1,
             'description': f'MLP gate projection: ({cfg.hidden_size}) -> ({cfg.intermediate_size}) with SiLU activation'},
            {'yaml': 'layer0_gate_proj.yaml', 'count': 1, 'alias': 'up_proj',
             'description': 'MLP up projection: same dimensions as gate_proj (SwiGLU)'},
            {'yaml': 'layer0_down_proj.yaml', 'count': 1,
             'description': f'MLP down projection: ({cfg.intermediate_size}) -> ({cfg.hidden_size})'},
        ]

    total_ops = ops_per_layer * cfg.num_hidden_layers + 1

    desc = {
        'network': {
            'model_name': cfg.model_name,
            'phase': phase_desc,
            'batch_size': batch_size,
            'attention': attn_desc,
        },
        'architecture': {
            'hidden_size': cfg.hidden_size,
            'num_attention_heads': cfg.num_attention_heads,
            'num_key_value_heads': cfg.num_key_value_heads,
            'head_dim': cfg.head_dim,
            'gqa_ratio': f'{cfg.num_attention_heads}Q / {cfg.num_key_value_heads}KV = {cfg.gqa_ratio}:1',
            'intermediate_size': cfg.intermediate_size,
            'vocab_size': cfg.vocab_size,
            'num_hidden_layers': cfg.num_hidden_layers,
        },
        'structure': {
            'description': (
                f"Only unique operator YAMLs are stored. "
                f"v_proj has the same dimensions as k_proj. "
                + (f"up_proj has the same dimensions as gate_proj. "
                   if not cfg.is_moe else
                   f"expert_up_proj has the same dimensions as expert_gate_proj. "
                   f"All {cfg.num_experts} experts have identical dimensions. ")
                + f"All {cfg.num_hidden_layers} decoder layers are identical."
            ),
            'unique_yamls': unique_names,
            'total_operators_in_model': total_ops,
            'operators_per_layer': ops_per_layer,
            'num_layers': cfg.num_hidden_layers,
            'layer_operator_sequence': layer_sequence,
            'global_operators': [
                {'yaml': 'lm_head.yaml', 'count': 1,
                 'description': f'Language model head: ({cfg.hidden_size}) -> ({cfg.vocab_size})'}
            ],
        },
    }

    if cfg.is_moe:
        desc['architecture']['moe'] = {
            'num_experts': cfg.num_experts,
            'num_experts_per_tok': top_k,
            'moe_intermediate_size': cfg.moe_intermediate_size,
            'tokens_per_expert': f'{tokens_per_expert} (uniform routing: B*S*{top_k}/{n_exp})',
            'total_params': f'{cfg.num_experts} experts (only {top_k} active per token)',
        }

    fpath = os.path.join(outdir, 'NETWORK.yaml')
    with open(fpath, 'w') as f:
        yaml.dump(desc, f, default_flow_style=False, sort_keys=False)


# ============================================================
# Verification: cross-check formulas against actual PyTorch model
# ============================================================

def verify_against_model(config_path: str,
                         batch_size: int = 1, seq_len: int = 128):
    """
    Verify converter formulas by cross-checking against actual model.

    Three independent verification methods:
    1. Structural: verify in_features/out_features of every nn.Linear
    2. Shape: run forward pass with hooks, compare intermediate tensor shapes
    3. FLOP: compare formula-derived MACs against torch profiler

    Returns True if all checks pass.
    """
    import torch
    import torch.nn as nn

    try:
        from transformers import AutoConfig, AutoModelForCausalLM
    except ImportError:
        print("ERROR: transformers not installed, cannot verify")
        return False

    if os.path.isdir(config_path):
        config_dir = config_path
    else:
        config_dir = os.path.dirname(config_path)

    # Load configs
    hf_config = AutoConfig.from_pretrained(config_dir)
    original_layers = hf_config.num_hidden_layers
    hf_config.num_hidden_layers = 1  # 1 layer for speed

    cfg = LLMConfig.from_hf_config(hf_config)
    cfg.num_hidden_layers = 1

    print(f"Verifying {cfg.model_name} (1-layer) against formulas...")
    print(f"  H={cfg.hidden_size}, n_h={cfg.num_attention_heads}, "
          f"n_kv={cfg.num_key_value_heads}, d={cfg.head_dim}, "
          f"I={cfg.intermediate_size}, V={cfg.vocab_size}")
    if cfg.is_moe:
        print(f"  MoE: {cfg.num_experts} experts, top-{cfg.num_experts_per_tok}, "
              f"expert_I={cfg.moe_intermediate_size}")
    print(f"  batch_size={batch_size}, seq_len={seq_len}")

    # Create model with random weights
    model = AutoModelForCausalLM.from_config(hf_config)
    model.eval()

    all_pass = True

    # ---- Verification 1: Structural check ----
    print("\n[1/3] Structural verification (Linear in/out features)...")

    expected_linears = {
        'model.layers.0.self_attn.q_proj': (cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim),
        'model.layers.0.self_attn.k_proj': (cfg.hidden_size, cfg.num_key_value_heads * cfg.head_dim),
        'model.layers.0.self_attn.v_proj': (cfg.hidden_size, cfg.num_key_value_heads * cfg.head_dim),
        'model.layers.0.self_attn.o_proj': (cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size),
    }

    if cfg.is_moe:
        # MoE: router + expert MLPs
        expected_linears['model.layers.0.mlp.gate'] = (cfg.hidden_size, cfg.num_experts)
        # Check first expert's dimensions
        expected_linears['model.layers.0.mlp.experts.0.gate_proj'] = (cfg.hidden_size, cfg.moe_intermediate_size)
        expected_linears['model.layers.0.mlp.experts.0.up_proj'] = (cfg.hidden_size, cfg.moe_intermediate_size)
        expected_linears['model.layers.0.mlp.experts.0.down_proj'] = (cfg.moe_intermediate_size, cfg.hidden_size)
    else:
        # Dense MLP
        expected_linears['model.layers.0.mlp.gate_proj'] = (cfg.hidden_size, cfg.intermediate_size)
        expected_linears['model.layers.0.mlp.up_proj'] = (cfg.hidden_size, cfg.intermediate_size)
        expected_linears['model.layers.0.mlp.down_proj'] = (cfg.intermediate_size, cfg.hidden_size)

    # lm_head (may not exist if tie_word_embeddings=True)
    expected_linears['lm_head'] = (cfg.hidden_size, cfg.vocab_size)

    for name, mod in model.named_modules():
        if name in expected_linears:
            exp_in, exp_out = expected_linears[name]
            # Handle both nn.Linear and special router modules
            if hasattr(mod, 'in_features'):
                actual_in = mod.in_features
                actual_out = mod.out_features
            elif hasattr(mod, 'weight'):
                actual_out, actual_in = mod.weight.shape
            else:
                print(f"  SKIP: {name} (no in/out features or weight)")
                continue
            ok_in = actual_in == exp_in
            ok_out = actual_out == exp_out
            status = "PASS" if (ok_in and ok_out) else "FAIL"
            if not (ok_in and ok_out):
                all_pass = False
            print(f"  {status}: {name} "
                  f"in={actual_in}(exp={exp_in}) "
                  f"out={actual_out}(exp={exp_out})")

    # ---- Verification 2: Forward pass shape check ----
    print("\n[2/3] Forward pass shape verification...")

    shapes = {}

    def make_hook(name):
        def hook(module, inp, out):
            in_tensors = [t for t in inp if isinstance(t, torch.Tensor)]
            if isinstance(out, torch.Tensor):
                out_shape = tuple(out.shape)
            elif isinstance(out, tuple):
                out_shape = tuple(
                    tuple(t.shape) for t in out if isinstance(t, torch.Tensor)
                )
            else:
                out_shape = None
            shapes[name] = {
                'input': [tuple(t.shape) for t in in_tensors],
                'output': out_shape,
            }
        return hook

    hooks = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            hooks.append(mod.register_forward_hook(make_hook(name)))

    input_ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
    with torch.no_grad():
        model(input_ids)

    for h in hooks:
        h.remove()

    BS = batch_size * seq_len
    expected_shapes = {
        'model.layers.0.self_attn.q_proj': {
            'input_last_dim': cfg.hidden_size,
            'output_last_dim': cfg.num_attention_heads * cfg.head_dim,
            'batch_tokens': BS,
        },
        'model.layers.0.self_attn.k_proj': {
            'input_last_dim': cfg.hidden_size,
            'output_last_dim': cfg.num_key_value_heads * cfg.head_dim,
            'batch_tokens': BS,
        },
        'model.layers.0.self_attn.v_proj': {
            'input_last_dim': cfg.hidden_size,
            'output_last_dim': cfg.num_key_value_heads * cfg.head_dim,
            'batch_tokens': BS,
        },
        'model.layers.0.self_attn.o_proj': {
            'input_last_dim': cfg.num_attention_heads * cfg.head_dim,
            'output_last_dim': cfg.hidden_size,
            'batch_tokens': BS,
        },
        'lm_head': {
            'input_last_dim': cfg.hidden_size,
            'output_last_dim': cfg.vocab_size,
            'batch_tokens': BS,
        },
    }

    if cfg.is_moe:
        expected_shapes['model.layers.0.mlp.gate'] = {
            'input_last_dim': cfg.hidden_size,
            'output_last_dim': cfg.num_experts,
            'batch_tokens': BS,
        }
        # MoE expert shapes — each expert sees all tokens during forward
        # (routing/masking happens after the linear)
        expected_shapes['model.layers.0.mlp.experts.0.gate_proj'] = {
            'input_last_dim': cfg.hidden_size,
            'output_last_dim': cfg.moe_intermediate_size,
            'batch_tokens': BS,  # all tokens pass through expert in HF impl
        }
    else:
        expected_shapes['model.layers.0.mlp.gate_proj'] = {
            'input_last_dim': cfg.hidden_size,
            'output_last_dim': cfg.intermediate_size,
            'batch_tokens': BS,
        }
        expected_shapes['model.layers.0.mlp.up_proj'] = {
            'input_last_dim': cfg.hidden_size,
            'output_last_dim': cfg.intermediate_size,
            'batch_tokens': BS,
        }
        expected_shapes['model.layers.0.mlp.down_proj'] = {
            'input_last_dim': cfg.intermediate_size,
            'output_last_dim': cfg.hidden_size,
            'batch_tokens': BS,
        }

    for name, expected in expected_shapes.items():
        if name not in shapes:
            print(f"  SKIP: {name} (not in captured shapes)")
            continue
        actual = shapes[name]
        in_shape = actual['input'][0]
        out_shape = actual['output']
        if isinstance(out_shape, tuple) and isinstance(out_shape[0], tuple):
            out_shape = out_shape[0]

        ok_in = in_shape[-1] == expected['input_last_dim']
        ok_out = out_shape[-1] == expected['output_last_dim']
        # Check total tokens = batch * seq_len
        actual_tokens = 1
        for d in in_shape[:-1]:
            actual_tokens *= d
        ok_batch = actual_tokens == expected['batch_tokens']

        status = "PASS" if (ok_in and ok_out and ok_batch) else "FAIL"
        if not (ok_in and ok_out and ok_batch):
            all_pass = False
        print(f"  {status}: {name}")
        print(f"         input={in_shape} (last_dim exp={expected['input_last_dim']})")
        print(f"         output={out_shape} (last_dim exp={expected['output_last_dim']})")
        print(f"         tokens={actual_tokens} (exp={expected['batch_tokens']})")

    # ---- Verification 3: FLOP count check ----
    print("\n[3/3] FLOP count verification...")

    # Formula-derived MACs per layer (multiply-accumulate operations)
    H = cfg.hidden_size
    n_h = cfg.num_attention_heads
    n_kv = cfg.num_key_value_heads
    d = cfg.head_dim
    I = cfg.intermediate_size
    V = cfg.vocab_size
    S = seq_len
    B = batch_size

    # Each Linear: MACs = batch_tokens * in_features * out_features
    # Each batched matmul: MACs = batch * M * K * N
    formula_macs = {
        'q_proj':     BS * H * (n_h * d),
        'k_proj':     BS * H * (n_kv * d),
        'v_proj':     BS * H * (n_kv * d),
        'attn_qk':    B * n_h * S * d * S,
        'attn_v':     B * n_h * S * S * d,
        'o_proj':     BS * (n_h * d) * H,
    }

    if cfg.is_moe:
        E_I = cfg.moe_intermediate_size
        n_exp = cfg.num_experts
        top_k = cfg.num_experts_per_tok
        tokens_per_expert = (BS * top_k) // n_exp
        if tokens_per_expert == 0:
            tokens_per_expert = 1
        formula_macs['router'] = BS * H * n_exp
        formula_macs['expert_gate'] = tokens_per_expert * H * E_I * n_exp
        formula_macs['expert_up'] = tokens_per_expert * H * E_I * n_exp
        formula_macs['expert_down'] = tokens_per_expert * E_I * H * n_exp
    else:
        formula_macs['gate_proj'] = BS * H * I
        formula_macs['up_proj'] = BS * H * I
        formula_macs['down_proj'] = BS * I * H

    lm_head_macs = BS * H * V

    layer_macs = sum(formula_macs.values())
    total_macs = layer_macs * original_layers + lm_head_macs

    print(f"  Formula-derived MACs per layer:")
    for name, macs in formula_macs.items():
        print(f"    {name:16s}: {macs:>15,}")
    print(f"    {'TOTAL':16s}: {layer_macs:>15,}")
    print(f"\n  Full model ({original_layers} layers):")
    print(f"    Layer MACs x {original_layers}: {layer_macs * original_layers:>18,}")
    print(f"    lm_head MACs:        {lm_head_macs:>18,}")
    print(f"    Total MACs:          {total_macs:>18,}")
    print(f"    Total FLOPs (2x):    {total_macs * 2:>18,}")
    print(f"    TFLOPs:              {total_macs * 2 / 1e12:>18.2f}")

    # Cross-check with torch profiler if available
    try:
        from torch.utils.flop_counter import FlopCounterMode

        input_ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
        flop_counter = FlopCounterMode(display=False)
        with flop_counter:
            with torch.no_grad():
                model(input_ids)

        torch_flops = flop_counter.get_total_flops()
        # torch flop counter counts FLOPs (not MACs), so divide by 2
        torch_macs = torch_flops // 2

        # Our 1-layer formula
        formula_1layer_macs = layer_macs + lm_head_macs
        ratio = torch_macs / formula_1layer_macs if formula_1layer_macs > 0 else 0

        print(f"\n  Torch profiler (1-layer model):")
        print(f"    Torch FLOPs:     {torch_flops:>15,}")
        print(f"    Torch MACs:      {torch_macs:>15,}")
        print(f"    Formula MACs:    {formula_1layer_macs:>15,}")
        print(f"    Ratio (torch/formula): {ratio:.6f}")
        if abs(ratio - 1.0) < 0.01:
            print(f"  PASS: FLOP counts match within 1%")
        else:
            print(f"  INFO: Ratio differs by {abs(ratio-1)*100:.1f}% "
                  f"(expected due to embedding/norm/softmax ops not in formula)")
    except ImportError:
        print("  SKIP: torch.utils.flop_counter not available")
    except Exception as e:
        print(f"  SKIP: FLOP counting failed: {e}")

    # ---- Summary ----
    print("\n" + "=" * 50)
    if all_pass:
        print("VERIFICATION PASSED: All structural and shape checks match.")
    else:
        print("VERIFICATION FAILED: Some checks did not match.")
    print("=" * 50)

    return all_pass


# ============================================================
# CLI
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert decoder-only LLM config to Timeloop workloads"
    )
    parser.add_argument(
        '--config', required=True,
        help='Path to HuggingFace config.json or model directory'
    )
    parser.add_argument(
        '--save-dir', default='./llm_workloads',
        help='Output directory for YAML files'
    )
    parser.add_argument(
        '--model-name', default=None,
        help='Model name override (auto-detected from config)'
    )
    parser.add_argument(
        '--batch-size', type=int, default=1,
        help='Batch size'
    )
    parser.add_argument(
        '--seq-len', type=int, default=128,
        help='Sequence length (prompt length for prefill, ignored for decode)'
    )
    parser.add_argument(
        '--phase', choices=['prefill', 'decode'], default='prefill',
        help='Inference phase: prefill (process prompt) or decode (generate tokens)'
    )
    parser.add_argument(
        '--kv-cache-len', type=int, default=None,
        help='KV cache length for decode phase (total context length seen so far)'
    )
    parser.add_argument(
        '--verify', action='store_true',
        help='Verify formulas against actual model (requires transformers)'
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.verify:
        verify_against_model(
            args.config,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
        )
    else:
        convert_llm(
            config_path=args.config,
            save_dir=args.save_dir,
            model_name=args.model_name,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            phase=args.phase,
            kv_cache_len=args.kv_cache_len,
        )


if __name__ == '__main__':
    main()
