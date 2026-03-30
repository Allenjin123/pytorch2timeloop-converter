"""
StableHLO MLIR frontend for Timeloop workload generation.

Parses StableHLO IR (.mlir text format) and extracts compute operators
(dot_general, convolution, reduce) into Timeloop problem YAML files.
Also auto-builds a DAG from SSA value flow.

This provides a framework-agnostic frontend: any model exported from
PyTorch (via torch_xla), JAX, or TensorFlow as StableHLO can be
converted to Timeloop workloads.

Usage:
    python -m pytorch2timeloop.convert_stablehlo \\
        --mlir model.mlir \\
        --save-dir ./workloads/my_model \\
        --model-name my_model

Programmatic:
    from pytorch2timeloop.convert_stablehlo import convert_stablehlo
    ops, dag = convert_stablehlo("model.mlir", "./workloads", "my_model")
"""

import re
import os
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path

import yaml

from .dag import NetworkDAG, DAGNode, DAGEdge

logger = logging.getLogger(__name__)


# ============================================================
# Tensor type parsing
# ============================================================

@dataclass
class TensorType:
    """Parsed tensor type from MLIR, e.g. tensor<4x512x768xf32>."""
    shape: List[int]
    dtype: str

    @classmethod
    def parse(cls, type_str: str) -> Optional['TensorType']:
        """Parse 'tensor<4x512x768xf32>' or 'tensor<f32>' (scalar)."""
        m = re.match(r'tensor<([^>]+)>', type_str.strip())
        if not m:
            return None
        inner = m.group(1)
        # Split on 'x', last part is dtype
        parts = inner.split('x')
        if len(parts) == 1:
            # Scalar tensor like tensor<f32>
            return cls(shape=[], dtype=parts[0])
        dtype = parts[-1]
        shape = [int(p) for p in parts[:-1] if p.strip()]
        return cls(shape=shape, dtype=dtype)

    @property
    def rank(self) -> int:
        return len(self.shape)

    def num_elements(self) -> int:
        r = 1
        for s in self.shape:
            r *= s
        return r


# ============================================================
# StableHLO operation representation
# ============================================================

@dataclass
class StableHLOOp:
    """A single StableHLO operation extracted from MLIR text."""
    op_name: str             # e.g. 'stablehlo.dot_general', 'stablehlo.convolution'
    result_names: List[str]  # SSA value names for results, e.g. ['%42']
    result_types: List[TensorType]
    operand_names: List[str] # SSA value names for operands, e.g. ['%40', '%41']
    operand_types: List[TensorType]
    attributes: Dict[str, str]  # raw attribute strings
    raw_text: str            # original MLIR text line(s)
    line_number: int         # for debugging


@dataclass
class DotGeneralAttrs:
    """Parsed attributes for stablehlo.dot_general."""
    lhs_batching_dims: List[int]
    rhs_batching_dims: List[int]
    lhs_contracting_dims: List[int]
    rhs_contracting_dims: List[int]


@dataclass
class ConvolutionAttrs:
    """Parsed attributes for stablehlo.convolution."""
    input_batch_dim: int
    input_feature_dim: int
    input_spatial_dims: List[int]
    kernel_input_feature_dim: int
    kernel_output_feature_dim: int
    kernel_spatial_dims: List[int]
    output_batch_dim: int
    output_feature_dim: int
    output_spatial_dims: List[int]
    strides: List[int]
    padding: List[List[int]]
    lhs_dilation: List[int]
    rhs_dilation: List[int]
    feature_group_count: int
    batch_group_count: int


# ============================================================
# Timeloop operator wrappers (for StableHLO-derived ops)
# ============================================================

@dataclass
class StableHLOLinearOp:
    """A dot_general mapped to Timeloop linear problem [B..., N, C, M].

    For matmuls with arbitrary batch dims, we flatten all batch dims into B.
    Contracting dim → C (input features), remaining non-batch dim → M (output features).
    """
    name: str
    batch_dims: Dict[str, int]   # e.g. {'B': 4} or {'B': 1, 'H': 32}
    n_dim: int                   # "row" dimension of LHS (not contracted)
    c_dim: int                   # contracting dimension
    m_dim: int                   # "col" dimension of RHS (not contracted)
    source_op: Optional[StableHLOOp] = field(default=None, repr=False)

    def to_yaml(self):
        # Use simple [B, N, C, M] for 2D matmul or reshape batch dims
        flat_batch = 1
        for v in self.batch_dims.values():
            flat_batch *= v

        if len(self.batch_dims) <= 1:
            dims = list(self.batch_dims.keys()) + ['N', 'C', 'M']
            instance = dict(self.batch_dims)
        elif len(self.batch_dims) == 2:
            # Attention-like: keep two batch dims (e.g. B, H)
            dims = list(self.batch_dims.keys()) + ['N', 'C', 'M']
            instance = dict(self.batch_dims)
        else:
            # Flatten all batch dims into B
            dims = ['B', 'N', 'C', 'M']
            instance = {'B': flat_batch}

        instance['N'] = self.n_dim
        instance['C'] = self.c_dim
        instance['M'] = self.m_dim

        # Data spaces follow the standard convention
        batch_proj = [[[d]] for d in self.batch_dims.keys()] if len(self.batch_dims) <= 2 else [[['B']]]

        return {
            'problem': {
                'shape': {
                    'name': self.name,
                    'dimensions': dims,
                    'data_spaces': [
                        {
                            'name': 'Inputs1',
                            'projection': batch_proj + [[['C']], [['M']]],
                        },
                        {
                            'name': 'Inputs2',
                            'projection': batch_proj + [[['N']], [['C']]],
                        },
                        {
                            'name': 'Outputs',
                            'projection': batch_proj + [[['N']], [['M']]],
                            'read_write': True,
                        },
                    ],
                },
                'instance': instance,
            },
        }


@dataclass
class StableHLOConvOp:
    """A stablehlo.convolution mapped to Timeloop conv problem."""
    name: str
    batch_size: int
    groups: int              # feature_group_count
    in_channels: int         # C per group
    out_channels: int        # M per group
    kernel_size: List[int]   # [R, S] for 2D
    output_size: List[int]   # [P, Q] for 2D
    strides: List[int]
    source_op: Optional[StableHLOOp] = field(default=None, repr=False)

    def to_yaml(self):
        G = self.groups
        C = self.in_channels
        M = self.out_channels

        if len(self.kernel_size) == 2:
            R, S = self.kernel_size
            P, Q = self.output_size
            Hs, Ws = self.strides if len(self.strides) == 2 else (1, 1)

            instance = {'G': G, 'C': C, 'M': M, 'R': R, 'S': S,
                        'N': self.batch_size, 'P': P, 'Q': Q}

            return {
                'problem': {
                    'shape': {
                        'name': self.name,
                        'dimensions': ['G', 'C', 'M', 'R', 'S', 'N', 'P', 'Q'],
                        'coefficients': [
                            {'name': 'Cgroup', 'default': C},
                            {'name': 'Mgroup', 'default': M},
                            {'name': 'Hstride', 'default': Hs},
                            {'name': 'Wstride', 'default': Ws},
                        ],
                        'data_spaces': [
                            {
                                'name': 'Weights',
                                'projection': [[['G']], [['C']], [['M']], [['R']], [['S']]],
                            },
                            {
                                'name': 'Inputs',
                                'projection': [
                                    [['N']],
                                    [['G', 'Cgroup'], ['C']],
                                    [['R'], ['P', 'Hstride']],
                                    [['S'], ['Q', 'Wstride']],
                                ],
                            },
                            {
                                'name': 'Outputs',
                                'projection': [
                                    [['N']],
                                    [['G', 'Mgroup'], ['M']],
                                    [['P']],
                                    [['Q']],
                                ],
                                'read_write': True,
                            },
                        ],
                    },
                    'instance': instance,
                },
            }
        else:
            # 1D conv: use [G, C, M, R, N, P]
            R = self.kernel_size[0]
            P = self.output_size[0]
            Hs = self.strides[0] if self.strides else 1

            instance = {'G': G, 'C': C, 'M': M, 'R': R,
                        'N': self.batch_size, 'P': P}
            return {
                'problem': {
                    'shape': {
                        'name': self.name,
                        'dimensions': ['G', 'C', 'M', 'R', 'N', 'P'],
                        'coefficients': [
                            {'name': 'Cgroup', 'default': C},
                            {'name': 'Mgroup', 'default': M},
                            {'name': 'Hstride', 'default': Hs},
                        ],
                        'data_spaces': [
                            {
                                'name': 'Weights',
                                'projection': [[['G']], [['C']], [['M']], [['R']]],
                            },
                            {
                                'name': 'Inputs',
                                'projection': [
                                    [['N']],
                                    [['G', 'Cgroup'], ['C']],
                                    [['R'], ['P', 'Hstride']],
                                ],
                            },
                            {
                                'name': 'Outputs',
                                'projection': [
                                    [['N']],
                                    [['G', 'Mgroup'], ['M']],
                                    [['P']],
                                ],
                                'read_write': True,
                            },
                        ],
                    },
                    'instance': instance,
                },
            }


@dataclass
class StableHLOReduceOp:
    """A stablehlo.reduce mapped to Timeloop reduction problem."""
    name: str
    reduce_type: str         # 'max', 'add', 'min'
    input_shape: List[int]
    reduce_dims: List[int]
    source_op: Optional[StableHLOOp] = field(default=None, repr=False)

    def to_yaml(self):
        # Name dimensions generically: D0, D1, D2, ...
        all_dims = [f'D{i}' for i in range(len(self.input_shape))]
        kept_dims = [d for i, d in enumerate(all_dims) if i not in self.reduce_dims]
        reduced_dims = [d for i, d in enumerate(all_dims) if i in self.reduce_dims]

        instance = {d: s for d, s in zip(all_dims, self.input_shape)}

        return {
            'problem': {
                'shape': {
                    'name': self.name,
                    'dimensions': all_dims,
                    'data_spaces': [
                        {
                            'name': 'Inputs',
                            'projection': [[[d]] for d in all_dims],
                        },
                        {
                            'name': 'Outputs',
                            'projection': [[[d]] for d in kept_dims],
                            'read_write': True,
                        },
                    ],
                },
                'instance': instance,
            },
        }


@dataclass
class StableHLOElementwiseOp:
    """Elementwise op (add, multiply, exp, etc.) - tracked for DAG but not always
    written to YAML (these are typically memory-bound, not compute-bound)."""
    name: str
    ewise_type: str          # 'add', 'multiply', 'subtract', 'divide', 'exponential'
    shape: List[int]
    source_op: Optional[StableHLOOp] = field(default=None, repr=False)

    def to_yaml(self):
        dims = [f'D{i}' for i in range(len(self.shape))]
        instance = {d: s for d, s in zip(dims, self.shape)}
        return {
            'problem': {
                'shape': {
                    'name': self.name,
                    'dimensions': dims,
                    'data_spaces': [
                        {'name': 'Inputs1', 'projection': [[[d]] for d in dims]},
                        {'name': 'Inputs2', 'projection': [[[d]] for d in dims]},
                        {'name': 'Outputs', 'projection': [[[d]] for d in dims], 'read_write': True},
                    ],
                },
                'instance': instance,
            },
        }


# ============================================================
# StableHLO MLIR text parser
# ============================================================

class StableHLOParser:
    """
    Parse StableHLO MLIR text format and extract operations.

    This is a regex-based parser for the MLIR text format. It doesn't require
    the stablehlo or mlir Python packages — just reads the .mlir text directly.
    This makes it portable and avoids version-matching headaches.
    """

    # Ops we care about for Timeloop conversion
    COMPUTE_OPS = {
        'stablehlo.dot_general',
        'stablehlo.convolution',
    }
    REDUCE_OPS = {
        'stablehlo.reduce',
        'stablehlo.reduce_window',
    }
    ELEMENTWISE_OPS = {
        'stablehlo.add', 'stablehlo.subtract', 'stablehlo.multiply',
        'stablehlo.divide', 'stablehlo.exponential', 'stablehlo.maximum',
        'stablehlo.minimum', 'stablehlo.negate',
    }
    # Ops we track for DAG edges but don't generate YAML for
    DATA_OPS = {
        'stablehlo.reshape', 'stablehlo.transpose', 'stablehlo.broadcast_in_dim',
        'stablehlo.concatenate', 'stablehlo.slice', 'stablehlo.dynamic_slice',
        'stablehlo.pad', 'stablehlo.convert', 'stablehlo.gather',
        'stablehlo.select', 'stablehlo.iota', 'stablehlo.constant',
        'stablehlo.compare', 'stablehlo.get_tuple_element',
    }

    def __init__(self):
        self.ops: List[StableHLOOp] = []
        # SSA value → defining op index
        self.value_to_op: Dict[str, int] = {}
        # SSA value → tensor type
        self.value_types: Dict[str, TensorType] = {}
        # Function arguments
        self.func_args: List[Tuple[str, TensorType]] = []

    def parse(self, mlir_text: str) -> List[StableHLOOp]:
        """Parse the full MLIR module text and return all operations."""
        self.ops = []
        self.value_to_op = {}
        self.value_types = {}
        self.func_args = []

        # Extract function arguments
        self._parse_func_args(mlir_text)

        # Parse operations line by line (handling multi-line ops)
        lines = mlir_text.split('\n')
        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Skip empty lines, comments, module/func/return/block declarations
            if (not line or line.startswith('//') or line.startswith('module')
                    or line.startswith('func.func') or line.startswith('return')
                    or line.startswith('}') or line.startswith('^')):
                i += 1
                continue

            # Accumulate multi-line ops (ops that span { ... } blocks)
            op_text = line
            if '{' in line and 'reducer' not in line.split('{')[0]:
                # Could be a region-bearing op; accumulate until balanced
                brace_count = op_text.count('{') - op_text.count('}')
                while brace_count > 0 and i + 1 < len(lines):
                    i += 1
                    op_text += '\n' + lines[i]
                    brace_count += lines[i].count('{') - lines[i].count('}')
            elif line.endswith(',') or (': (' in line and ')' not in line):
                # Multi-line type signature
                while i + 1 < len(lines) and not lines[i].rstrip().endswith(')'):
                    i += 1
                    op_text += ' ' + lines[i].strip()

            # Try to parse as a StableHLO op
            op = self._parse_op(op_text, i)
            if op is not None:
                op_idx = len(self.ops)
                self.ops.append(op)
                for rname in op.result_names:
                    self.value_to_op[rname] = op_idx
                for rname, rtype in zip(op.result_names, op.result_types):
                    if rtype:
                        self.value_types[rname] = rtype

            i += 1

        return self.ops

    def _parse_func_args(self, mlir_text: str):
        """Extract function argument names and types."""
        # Match: func.func @main(%arg0: tensor<4x3x224x224xf32>, %arg1: ...)
        m = re.search(r'func\.func\s+@\w+\(([^)]+)\)', mlir_text)
        if not m:
            return
        args_str = m.group(1)
        for arg_match in re.finditer(r'(%\w+)\s*:\s*(tensor<[^>]+>)', args_str):
            name = arg_match.group(1)
            ttype = TensorType.parse(arg_match.group(2))
            if ttype:
                self.func_args.append((name, ttype))
                self.value_types[name] = ttype

    def _parse_op(self, text: str, line_num: int) -> Optional[StableHLOOp]:
        """Try to parse a single MLIR operation from text."""
        # Pattern: %result = op_name %operand1, %operand2 ... : (types) -> result_type
        # Or: %result:2 = op_name ...

        # Extract result names
        result_match = re.match(r'(%[\w.:]+(?:,\s*%[\w.:]+)*)\s*=\s*', text)
        if not result_match:
            # Could be a void op; skip these
            return None

        result_str = result_match.group(1)
        # Handle %0:2 (multiple results) or %0, %1
        result_names = []
        for r in result_str.split(','):
            r = r.strip()
            colon_m = re.match(r'(%\w+):(\d+)', r)
            if colon_m:
                base = colon_m.group(1)
                count = int(colon_m.group(2))
                for j in range(count):
                    result_names.append(f'{base}#{j}')
            else:
                result_names.append(r)

        rest = text[result_match.end():].strip()

        # Extract op name (stablehlo.xxx or other dialect)
        op_match = re.match(r'("?[\w.]+(?:\.\w+)*"?)', rest)
        if not op_match:
            return None
        op_name = op_match.group(1).strip('"')

        # Check if it's an op we care about
        all_tracked = (self.COMPUTE_OPS | self.REDUCE_OPS |
                       self.ELEMENTWISE_OPS | self.DATA_OPS)
        if op_name not in all_tracked:
            # Still register result types if we can find them
            self._register_result_types(text, result_names)
            return None

        # Extract operand names (% references before the attribute section)
        rest_after_op = rest[op_match.end():].strip()
        operand_names = re.findall(r'(%[\w.#]+)', rest_after_op.split(':')[0]
                                   if ':' in rest_after_op else rest_after_op)

        # Extract result and operand types from the MLIR type signature.
        # The type sig is at the very end:  : (operand_types) -> result_types
        # For convolution, dim_numbers also contains '->' so we must find
        # the LAST occurrence of the pattern ': (...) -> ...' or ') -> ...'.
        result_types = []
        operand_types = []

        # Find the final type signature: look for last ') -> tensor<'
        # which separates operand types from result types
        type_sig_match = re.search(
            r':\s*\(([^)]*tensor<[^)]*)\)\s*->\s*(.*?)\s*$', text, re.DOTALL)
        if type_sig_match:
            # Operand types from the (...) part
            for tm in re.finditer(r'tensor<[^>]+>', type_sig_match.group(1)):
                operand_types.append(TensorType.parse(tm.group()))
            # Result types from after ->
            for tm in re.finditer(r'tensor<[^>]+>', type_sig_match.group(2)):
                result_types.append(TensorType.parse(tm.group()))
        else:
            # Fallback: find any '-> tensor<...>' at the end
            type_match = re.search(r'->\s*(tensor<[^>]+>)\s*$', text)
            if type_match:
                result_types.append(TensorType.parse(type_match.group(1)))
        # Fill from value_types if we can
        for i, oname in enumerate(operand_names):
            if i >= len(operand_types) or operand_types[i] is None:
                if oname in self.value_types:
                    if i < len(operand_types):
                        operand_types[i] = self.value_types[oname]
                    else:
                        operand_types.append(self.value_types[oname])

        # Extract attributes (everything between the op name and the type sig)
        attrs = {}
        # dot_general dimension numbers
        dn_match = re.search(
            r'dot_dimension_numbers\s*=\s*#stablehlo\.dot<'
            r'\s*lhs_batching_dimensions\s*=\s*\[([^\]]*)\]'
            r'\s*,\s*rhs_batching_dimensions\s*=\s*\[([^\]]*)\]'
            r'\s*,\s*lhs_contracting_dimensions\s*=\s*\[([^\]]*)\]'
            r'\s*,\s*rhs_contracting_dimensions\s*=\s*\[([^\]]*)\]',
            text, re.DOTALL)
        if dn_match:
            attrs['dot_dimension_numbers'] = dn_match.group(0)
            attrs['lhs_batching_dims'] = dn_match.group(1).strip()
            attrs['rhs_batching_dims'] = dn_match.group(2).strip()
            attrs['lhs_contracting_dims'] = dn_match.group(3).strip()
            attrs['rhs_contracting_dims'] = dn_match.group(4).strip()

        # Also try the inline format: batching_dims = [0] x [0], contracting_dims = [2] x [1]
        inline_dn = re.search(
            r'batching_dims\s*=\s*\[([^\]]*)\]\s*x\s*\[([^\]]*)\]'
            r'\s*,\s*contracting_dims\s*=\s*\[([^\]]*)\]\s*x\s*\[([^\]]*)\]',
            text)
        if inline_dn:
            attrs['lhs_batching_dims'] = inline_dn.group(1).strip()
            attrs['rhs_batching_dims'] = inline_dn.group(2).strip()
            attrs['lhs_contracting_dims'] = inline_dn.group(3).strip()
            attrs['rhs_contracting_dims'] = inline_dn.group(4).strip()

        # Convolution dim_numbers
        conv_dn_match = re.search(
            r'dim_numbers\s*=\s*\[([^\]]+)\]\s*x\s*\[([^\]]+)\]\s*->\s*\[([^\]]+)\]',
            text)
        if conv_dn_match:
            attrs['input_dims'] = conv_dn_match.group(1).strip()
            attrs['kernel_dims'] = conv_dn_match.group(2).strip()
            attrs['output_dims'] = conv_dn_match.group(3).strip()

        # Strides
        stride_match = re.search(r'stride\s*=\s*\[([^\]]*)\]', text)
        if stride_match:
            attrs['stride'] = stride_match.group(1).strip()

        # Padding
        pad_match = re.search(r'pad\s*=\s*\[([^\]]*(?:\[[^\]]*\])*[^\]]*)\]', text)
        if pad_match:
            attrs['pad'] = pad_match.group(1).strip()

        # feature_group_count, batch_group_count
        fgc_match = re.search(r'feature_group_count\s*=\s*(\d+)', text)
        if fgc_match:
            attrs['feature_group_count'] = fgc_match.group(1)
        bgc_match = re.search(r'batch_group_count\s*=\s*(\d+)', text)
        if bgc_match:
            attrs['batch_group_count'] = bgc_match.group(1)

        # Reduce dimensions
        reduce_dim_match = re.search(r'across\s+dimensions\s*=\s*\[([^\]]*)\]', text)
        if reduce_dim_match:
            attrs['reduce_dims'] = reduce_dim_match.group(1).strip()

        # Window for reduce_window
        window_match = re.search(r'window\s*=\s*\{([^}]+)\}', text)
        if window_match:
            attrs['window'] = window_match.group(1).strip()

        op = StableHLOOp(
            op_name=op_name,
            result_names=result_names,
            result_types=result_types,
            operand_names=operand_names,
            operand_types=operand_types,
            attributes=attrs,
            raw_text=text[:200],  # truncate for readability
            line_number=line_num,
        )

        # Register result types
        for rname, rtype in zip(result_names, result_types):
            if rtype:
                self.value_types[rname] = rtype

        return op

    def _register_result_types(self, text: str, result_names: List[str]):
        """Try to extract and register result types even for skipped ops."""
        # Find last type signature
        type_sig_match = re.search(
            r':\s*\(([^)]*tensor<[^)]*)\)\s*->\s*(.*?)\s*$', text, re.DOTALL)
        if type_sig_match:
            types = [TensorType.parse(tm.group())
                     for tm in re.finditer(r'tensor<[^>]+>', type_sig_match.group(2))]
        else:
            type_match = re.search(r'->\s*(tensor<[^>]+>)\s*$', text)
            types = [TensorType.parse(type_match.group(1))] if type_match else []
        for rname, rtype in zip(result_names, types):
            if rtype:
                self.value_types[rname] = rtype


# ============================================================
# Op-to-Timeloop mapping
# ============================================================

def _parse_int_list(s: str) -> List[int]:
    """Parse '0, 1, 2' or '' into a list of ints."""
    s = s.strip()
    if not s:
        return []
    return [int(x.strip()) for x in s.split(',') if x.strip()]


def _map_dot_general(op: StableHLOOp, idx: int, name_prefix: str) -> Optional[StableHLOLinearOp]:
    """Map a stablehlo.dot_general to a Timeloop linear operator."""
    attrs = op.attributes
    if 'lhs_contracting_dims' not in attrs:
        logger.warning(f"dot_general at line {op.line_number} missing dimension numbers, skipping")
        return None

    lhs_batch = _parse_int_list(attrs['lhs_batching_dims'])
    rhs_batch = _parse_int_list(attrs['rhs_batching_dims'])
    lhs_contract = _parse_int_list(attrs['lhs_contracting_dims'])
    rhs_contract = _parse_int_list(attrs['rhs_contracting_dims'])

    lhs_type = op.operand_types[0] if len(op.operand_types) > 0 else None
    rhs_type = op.operand_types[1] if len(op.operand_types) > 1 else None
    result_type = op.result_types[0] if op.result_types else None

    if not lhs_type or not rhs_type:
        logger.warning(f"dot_general at line {op.line_number}: missing operand types")
        return None

    lhs_shape = lhs_type.shape
    rhs_shape = rhs_type.shape

    # Identify batch, contracting, and free dimensions
    batch_dims = {}
    batch_dim_names = []
    for i, (lb, rb) in enumerate(zip(lhs_batch, rhs_batch)):
        # Use meaningful names for common patterns
        if i == 0 and len(lhs_batch) == 1:
            dim_name = 'B'
        elif i == 0 and len(lhs_batch) == 2:
            dim_name = 'B'
        elif i == 1 and len(lhs_batch) == 2:
            dim_name = 'H'
        else:
            dim_name = f'B{i}'
        batch_dims[dim_name] = lhs_shape[lb]
        batch_dim_names.append(dim_name)

    # Contracting dimension(s) — typically just one
    c_size = 1
    for ld in lhs_contract:
        c_size *= lhs_shape[ld]

    # Free dimensions: LHS dims not in batch or contracting → N
    lhs_free = [d for d in range(len(lhs_shape))
                if d not in lhs_batch and d not in lhs_contract]
    n_size = 1
    for d in lhs_free:
        n_size *= lhs_shape[d]

    # RHS dims not in batch or contracting → M
    rhs_free = [d for d in range(len(rhs_shape))
                if d not in rhs_batch and d not in rhs_contract]
    m_size = 1
    for d in rhs_free:
        m_size *= rhs_shape[d]

    op_name = f"{name_prefix}dot_general_{idx}"
    return StableHLOLinearOp(
        name=op_name,
        batch_dims=batch_dims,
        n_dim=n_size,
        c_dim=c_size,
        m_dim=m_size,
        source_op=op,
    )


def _map_convolution(op: StableHLOOp, idx: int, name_prefix: str) -> Optional[StableHLOConvOp]:
    """Map a stablehlo.convolution to a Timeloop convolution operator."""
    attrs = op.attributes
    lhs_type = op.operand_types[0] if len(op.operand_types) > 0 else None
    rhs_type = op.operand_types[1] if len(op.operand_types) > 1 else None
    result_type = op.result_types[0] if op.result_types else None

    if not lhs_type or not rhs_type or not result_type:
        logger.warning(f"convolution at line {op.line_number}: missing types")
        return None

    input_shape = lhs_type.shape
    kernel_shape = rhs_type.shape
    output_shape = result_type.shape

    fgc = int(attrs.get('feature_group_count', '1'))
    strides = _parse_int_list(attrs.get('stride', ''))

    # Parse dim_numbers to identify batch/feature/spatial dimensions
    # Default layout: NCHW for input, OIHW for kernel
    if 'input_dims' in attrs:
        input_dims_str = attrs['input_dims']
        kernel_dims_str = attrs['kernel_dims']
        output_dims_str = attrs['output_dims']

        # Parse layout strings like "b, 0, 1, f" or "b, f, 0, 1"
        input_dims = [d.strip() for d in input_dims_str.split(',')]
        kernel_dims = [d.strip() for d in kernel_dims_str.split(',')]
        output_dims = [d.strip() for d in output_dims_str.split(',')]

        # Find batch and feature positions
        batch_pos = input_dims.index('b') if 'b' in input_dims else 0
        feat_pos = input_dims.index('f') if 'f' in input_dims else 1
        spatial_pos = [i for i, d in enumerate(input_dims) if d not in ('b', 'f')]

        N = input_shape[batch_pos]
        # For grouped conv: total input channels / groups
        total_in_ch = input_shape[feat_pos]
        C = total_in_ch // fgc

        # Kernel: find i (input) and o (output) positions
        k_in_pos = kernel_dims.index('i') if 'i' in kernel_dims else None
        k_out_pos = kernel_dims.index('o') if 'o' in kernel_dims else None
        k_spatial = [i for i, d in enumerate(kernel_dims) if d not in ('i', 'o')]

        total_out_ch = kernel_shape[k_out_pos] if k_out_pos is not None else kernel_shape[0]
        M = total_out_ch // fgc

        kernel_size = [kernel_shape[d] for d in k_spatial]

        # Output spatial dims
        out_feat_pos = output_dims.index('f') if 'f' in output_dims else 1
        out_spatial = [i for i, d in enumerate(output_dims) if d not in ('b', 'f')]
        output_size = [output_shape[d] for d in out_spatial]
    else:
        # Fallback: assume NCHW layout
        N = input_shape[0]
        C = input_shape[1] // fgc
        rank = len(input_shape)
        kernel_size = list(kernel_shape[2:])
        M = kernel_shape[0] // fgc
        output_size = list(output_shape[2:])

    if not strides:
        strides = [1] * len(kernel_size)

    op_name = f"{name_prefix}conv_{idx}"
    return StableHLOConvOp(
        name=op_name,
        batch_size=N,
        groups=fgc,
        in_channels=C,
        out_channels=M,
        kernel_size=kernel_size,
        output_size=output_size,
        strides=strides,
        source_op=op,
    )


def _map_reduce(op: StableHLOOp, idx: int, name_prefix: str) -> Optional[StableHLOReduceOp]:
    """Map a stablehlo.reduce to a Timeloop reduction operator."""
    attrs = op.attributes
    reduce_dims = _parse_int_list(attrs.get('reduce_dims', ''))

    input_type = op.operand_types[0] if op.operand_types else None
    if not input_type:
        return None

    # Detect reduction type from the body
    raw = op.raw_text
    if 'maximum' in raw or 'max' in raw.lower():
        rtype = 'max'
    elif 'add' in raw:
        rtype = 'add'
    elif 'minimum' in raw or 'min' in raw.lower():
        rtype = 'min'
    else:
        rtype = 'unknown'

    op_name = f"{name_prefix}reduce_{rtype}_{idx}"
    return StableHLOReduceOp(
        name=op_name,
        reduce_type=rtype,
        input_shape=input_type.shape,
        reduce_dims=reduce_dims,
        source_op=op,
    )


def _map_elementwise(op: StableHLOOp, idx: int, name_prefix: str) -> Optional[StableHLOElementwiseOp]:
    """Map an elementwise StableHLO op."""
    ewise_name = op.op_name.split('.')[-1]  # e.g. 'add', 'multiply'
    result_type = op.result_types[0] if op.result_types else None
    if not result_type:
        return None

    op_name = f"{name_prefix}{ewise_name}_{idx}"
    return StableHLOElementwiseOp(
        name=op_name,
        ewise_type=ewise_name,
        shape=result_type.shape,
        source_op=op,
    )


# ============================================================
# Softmax pattern detection
# ============================================================

def _detect_softmax_patterns(ops: List[StableHLOOp], mapped: List) -> List[dict]:
    """
    Detect softmax patterns: reduce_max → subtract → exp → reduce_add → divide.
    Returns annotations for the mapped ops.
    """
    patterns = []
    # Look for reduce_max followed (within a window) by exp then reduce_add then divide
    reduce_max_indices = [i for i, op in enumerate(ops)
                          if op.op_name == 'stablehlo.reduce' and 'maximum' in op.raw_text]

    for rmi in reduce_max_indices:
        window = ops[rmi:rmi+20]  # look ahead
        has_exp = any(o.op_name == 'stablehlo.exponential' for o in window)
        has_reduce_add = any(o.op_name == 'stablehlo.reduce' and 'add' in o.raw_text
                             for o in window[1:])
        has_div = any(o.op_name == 'stablehlo.divide' for o in window)

        if has_exp and has_reduce_add and has_div:
            patterns.append({
                'type': 'softmax',
                'start_op_idx': rmi,
                'description': 'softmax = max → sub → exp → sum → div',
            })

    return patterns


# ============================================================
# DAG builder from SSA
# ============================================================

def build_dag_from_stablehlo(
    model_name: str,
    parser: StableHLOParser,
    mapped_ops: List,
    include_elementwise: bool = False,
) -> NetworkDAG:
    """
    Auto-build a NetworkDAG from StableHLO SSA value flow.

    Only compute ops (dot_general, convolution) and optionally reduces
    are included as nodes. Edges represent data flow through intermediate
    ops (reshape, transpose, etc.) — we trace through non-compute ops
    to connect compute-to-compute.
    """
    dag = NetworkDAG(model_name=model_name)

    # Build map: SSA value name → mapped op (the Timeloop operator)
    ssa_to_mapped = {}
    mapped_by_source_line = {}
    for mop in mapped_ops:
        if mop.source_op:
            for rname in mop.source_op.result_names:
                ssa_to_mapped[rname] = mop
            mapped_by_source_line[mop.source_op.line_number] = mop

    # Add function arguments as input placeholder nodes
    for arg_name, arg_type in parser.func_args:
        node_id = arg_name.replace('%', 'input_')
        dag.add_node(DAGNode(
            id=node_id,
            op_type='placeholder',
            yaml_file='',
            tensor_dims={f'D{i}': s for i, s in enumerate(arg_type.shape)},
        ))
        ssa_to_mapped[arg_name] = type('Placeholder', (), {
            'name': node_id, 'source_op': None})()

    # Add compute/reduce nodes
    for mop in mapped_ops:
        if isinstance(mop, StableHLOElementwiseOp) and not include_elementwise:
            continue
        yaml_file = f"{mop.name}.yaml"
        if isinstance(mop, StableHLOLinearOp):
            op_type = 'linear'
            tensor_dims = dict(mop.batch_dims)
            tensor_dims.update({'N': mop.n_dim, 'C': mop.c_dim, 'M': mop.m_dim})
        elif isinstance(mop, StableHLOConvOp):
            op_type = 'conv2d' if len(mop.kernel_size) == 2 else 'conv1d'
            tensor_dims = {'N': mop.batch_size, 'G': mop.groups,
                           'C': mop.in_channels, 'M': mop.out_channels}
        elif isinstance(mop, StableHLOReduceOp):
            op_type = f'reduce_{mop.reduce_type}'
            tensor_dims = {f'D{i}': s for i, s in enumerate(mop.input_shape)}
        elif isinstance(mop, StableHLOElementwiseOp):
            op_type = mop.ewise_type
            tensor_dims = {f'D{i}': s for i, s in enumerate(mop.shape)}
        else:
            continue

        dag.add_node(DAGNode(
            id=mop.name,
            op_type=op_type,
            yaml_file=yaml_file,
            tensor_dims=tensor_dims,
        ))

    # Build edges by tracing SSA def-use chains
    # For each mapped op, trace its operands back through non-compute ops
    # to find the nearest compute/input producer
    all_ops_by_result = {}
    for op in parser.ops:
        for rname in op.result_names:
            all_ops_by_result[rname] = op

    def _trace_producer(ssa_name: str, visited: set = None) -> Optional[str]:
        """Trace an SSA value back to the nearest mapped op or input."""
        if visited is None:
            visited = set()
        if ssa_name in visited:
            return None
        visited.add(ssa_name)

        if ssa_name in ssa_to_mapped:
            mop = ssa_to_mapped[ssa_name]
            return mop.name

        # Check if it's defined by a non-compute op; trace through its operands
        if ssa_name in all_ops_by_result:
            defn_op = all_ops_by_result[ssa_name]
            # For reshape/transpose/broadcast, trace through first operand
            for oname in defn_op.operand_names:
                result = _trace_producer(oname, visited)
                if result:
                    return result

        return None

    added_edges = set()
    for mop in mapped_ops:
        if isinstance(mop, StableHLOElementwiseOp) and not include_elementwise:
            continue
        if not mop.source_op:
            continue
        dst_id = mop.name
        if dst_id not in dag.nodes:
            continue

        for oname in mop.source_op.operand_names:
            src_id = _trace_producer(oname)
            if src_id and src_id != dst_id and src_id in dag.nodes:
                edge_key = (src_id, dst_id)
                if edge_key not in added_edges:
                    # Get tensor shape from SSA value type
                    shape = []
                    if oname in parser.value_types:
                        shape = parser.value_types[oname].shape
                    dag.add_edge(src_id, dst_id, oname.replace('%', 'v'), shape)
                    added_edges.add(edge_key)

    return dag


# ============================================================
# Deduplication
# ============================================================

def _dedup_ops(ops: List) -> Tuple[List, Dict[str, List[str]]]:
    """
    Deduplicate operators with identical Timeloop problem shapes.
    Returns (unique_ops, alias_map) where alias_map[unique_name] = [dup_names].
    """
    seen = {}  # yaml_content_hash → first op
    unique = []
    alias_map = {}

    for op in ops:
        if isinstance(op, StableHLOElementwiseOp):
            unique.append(op)
            continue

        y = op.to_yaml()
        # Hash by instance dimensions (the actual tensor sizes)
        inst = y['problem']['instance']
        dims = y['problem']['shape']['dimensions']
        key = (tuple(dims), tuple(sorted(inst.items())))

        if key in seen:
            first_name = seen[key]
            if first_name not in alias_map:
                alias_map[first_name] = []
            alias_map[first_name].append(op.name)
        else:
            seen[key] = op.name
            unique.append(op)

    return unique, alias_map


# ============================================================
# Main conversion function
# ============================================================

def convert_stablehlo(
    mlir_path: str,
    save_dir: str,
    model_name: str = None,
    include_reduces: bool = True,
    include_elementwise: bool = False,
    build_dag: bool = True,
    dedup: bool = True,
) -> Tuple[List, Optional[NetworkDAG]]:
    """
    Convert a StableHLO MLIR file to Timeloop workload YAMLs.

    Args:
        mlir_path: Path to .mlir file (StableHLO text format).
        save_dir: Directory to write YAML files.
        model_name: Model name for NETWORK.yaml and DAG.
        include_reduces: Include reduce ops (softmax stages) in output.
        include_elementwise: Include elementwise ops in output.
        build_dag: Build and save DAG.yaml.
        dedup: Deduplicate identical operators.

    Returns:
        (operators, dag) — list of mapped operators and optional NetworkDAG.
    """
    if model_name is None:
        model_name = Path(mlir_path).stem

    name_prefix = ""

    # Parse MLIR
    logger.info(f"Parsing StableHLO from {mlir_path}")
    with open(mlir_path, 'r') as f:
        mlir_text = f.read()

    parser = StableHLOParser()
    all_ops = parser.parse(mlir_text)
    logger.info(f"Parsed {len(all_ops)} StableHLO operations")

    # Map to Timeloop operators
    mapped = []
    for i, op in enumerate(all_ops):
        if op.op_name == 'stablehlo.dot_general':
            mop = _map_dot_general(op, i, name_prefix)
            if mop:
                mapped.append(mop)
        elif op.op_name == 'stablehlo.convolution':
            mop = _map_convolution(op, i, name_prefix)
            if mop:
                mapped.append(mop)
        elif op.op_name in StableHLOParser.REDUCE_OPS and include_reduces:
            mop = _map_reduce(op, i, name_prefix)
            if mop:
                mapped.append(mop)
        elif op.op_name in StableHLOParser.ELEMENTWISE_OPS and include_elementwise:
            mop = _map_elementwise(op, i, name_prefix)
            if mop:
                mapped.append(mop)

    logger.info(f"Mapped {len(mapped)} operators to Timeloop problems")

    # Detect softmax patterns
    softmax_patterns = _detect_softmax_patterns(all_ops, mapped)
    if softmax_patterns:
        logger.info(f"Detected {len(softmax_patterns)} softmax pattern(s)")

    # Deduplicate
    alias_map = {}
    if dedup:
        unique_ops, alias_map = _dedup_ops(mapped)
        logger.info(f"After dedup: {len(unique_ops)} unique ops "
                     f"({len(mapped) - len(unique_ops)} duplicates)")
    else:
        unique_ops = mapped

    # Write YAML files
    os.makedirs(save_dir, exist_ok=True)

    for mop in unique_ops:
        yaml_data = mop.to_yaml()
        fname = os.path.join(save_dir, f"{mop.name}.yaml")
        with open(fname, 'w') as f:
            yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
        logger.debug(f"Wrote {fname}")

    # Write NETWORK.yaml
    _write_network_yaml(save_dir, model_name, unique_ops, mapped, alias_map,
                        softmax_patterns, parser)

    # Build DAG
    dag = None
    if build_dag:
        dag = build_dag_from_stablehlo(model_name, parser, mapped,
                                       include_elementwise=include_elementwise)
        dag_path = os.path.join(save_dir, 'DAG.yaml')
        dag.save_yaml(dag_path)
        logger.info(f"Wrote DAG with {len(dag.nodes)} nodes, {len(dag.edges)} edges")

    return mapped, dag


def _write_network_yaml(save_dir, model_name, unique_ops, all_ops, alias_map,
                         softmax_patterns, parser):
    """Write NETWORK.yaml describing the full model structure."""
    network = {
        'network': {
            'model_name': model_name,
            'frontend': 'StableHLO',
            'total_operators': len(all_ops),
            'unique_operators': len(unique_ops),
        },
        'structure': {
            'unique_yamls': [op.name for op in unique_ops
                             if not isinstance(op, StableHLOElementwiseOp)],
            'operator_sequence': [],
        },
    }

    # Operator sequence
    for op in all_ops:
        entry = {'name': op.name}
        if isinstance(op, StableHLOLinearOp):
            entry['type'] = 'dot_general'
            entry['description'] = (
                f"MatMul: batch={op.batch_dims}, "
                f"({op.n_dim} x {op.c_dim}) @ ({op.c_dim} x {op.m_dim})"
            )
        elif isinstance(op, StableHLOConvOp):
            entry['type'] = 'convolution'
            entry['description'] = (
                f"Conv: N={op.batch_size}, G={op.groups}, "
                f"C={op.in_channels}, M={op.out_channels}, "
                f"kernel={op.kernel_size}, out={op.output_size}"
            )
        elif isinstance(op, StableHLOReduceOp):
            entry['type'] = f'reduce_{op.reduce_type}'
            entry['description'] = (
                f"Reduce({op.reduce_type}) over dims {op.reduce_dims} "
                f"of shape {op.input_shape}"
            )
        elif isinstance(op, StableHLOElementwiseOp):
            entry['type'] = op.ewise_type
            entry['description'] = f"Elementwise {op.ewise_type}, shape={op.shape}"
        network['structure']['operator_sequence'].append(entry)

    # Aliases (dedup info)
    if alias_map:
        network['structure']['aliases'] = {
            k: v for k, v in alias_map.items()
        }

    # Softmax annotations
    if softmax_patterns:
        network['structure']['detected_patterns'] = softmax_patterns

    # Input/output shapes from function signature
    if parser.func_args:
        network['network']['inputs'] = [
            {'name': name.replace('%', ''), 'shape': t.shape, 'dtype': t.dtype}
            for name, t in parser.func_args
        ]

    path = os.path.join(save_dir, 'NETWORK.yaml')
    with open(path, 'w') as f:
        yaml.dump(network, f, default_flow_style=False, sort_keys=False)


# ============================================================
# CLI
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Convert StableHLO MLIR to Timeloop workloads')
    parser.add_argument('--mlir', required=True, help='Path to .mlir file')
    parser.add_argument('--save-dir', required=True, help='Output directory')
    parser.add_argument('--model-name', default=None, help='Model name')
    parser.add_argument('--include-reduces', action='store_true', default=True,
                        help='Include reduce ops (default: True)')
    parser.add_argument('--include-elementwise', action='store_true', default=False,
                        help='Include elementwise ops (default: False)')
    parser.add_argument('--no-dag', action='store_true', default=False,
                        help='Skip DAG generation')
    parser.add_argument('--no-dedup', action='store_true', default=False,
                        help='Skip deduplication')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format='%(levelname)s: %(message)s')

    ops, dag = convert_stablehlo(
        mlir_path=args.mlir,
        save_dir=args.save_dir,
        model_name=args.model_name,
        include_reduces=args.include_reduces,
        include_elementwise=args.include_elementwise,
        build_dag=not args.no_dag,
        dedup=not args.no_dedup,
    )

    print(f"\nConverted {len(ops)} operators to Timeloop workloads in {args.save_dir}/")
    if dag:
        print(f"DAG: {len(dag.nodes)} nodes, {len(dag.edges)} edges")


if __name__ == '__main__':
    main()
