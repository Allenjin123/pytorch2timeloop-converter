"""
DAG (Directed Acyclic Graph) representation for neural network operator graphs.

Each node is a compute operator (conv, linear, matmul, etc.) with a reference
to its Timeloop problem YAML.  Edges carry tensor metadata (shape, dtype)
and represent data dependencies.

The DAG captures:
  - Which operators can execute in parallel (no dependency between them)
  - Data flow between operators (tensor shapes for communication cost)
  - Skip/residual connections
  - Repeatable subgraphs (e.g., identical decoder layers)

Serializes to a DAG.yaml that Mozart can consume for pipeline stage mapping.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import yaml


@dataclass
class DAGNode:
    """A single compute operator in the network DAG."""
    id: str
    op_type: str          # 'linear', 'conv2d', 'depthwise_conv2d', 'attn_qk',
                          # 'attn_v', 'elementwise_add', 'elementwise_mul', etc.
    yaml_file: str        # Timeloop problem YAML filename (relative)
    tensor_dims: dict     # key dimension sizes, e.g. {'B':1, 'N':2048, 'C':4096, 'M':4096}

    def to_dict(self):
        return {
            'id': self.id,
            'op_type': self.op_type,
            'yaml_file': self.yaml_file,
            'tensor_dims': self.tensor_dims,
        }


@dataclass
class DAGEdge:
    """A data dependency between two operators."""
    src: str              # source node id
    dst: str              # destination node id
    tensor_name: str      # semantic name, e.g. 'hidden_states', 'query', 'scores'
    tensor_shape: list    # shape as list of ints, e.g. [1, 2048, 4096]

    def to_dict(self):
        return {
            'src': self.src,
            'dst': self.dst,
            'tensor_name': self.tensor_name,
            'tensor_shape': self.tensor_shape,
        }


@dataclass
class NetworkDAG:
    """Complete DAG for a neural network."""
    model_name: str
    nodes: Dict[str, DAGNode] = field(default_factory=dict)
    edges: List[DAGEdge] = field(default_factory=list)
    # For repeated subgraphs (e.g., decoder layers): maps a prefix to repeat count
    repeat_blocks: Dict[str, dict] = field(default_factory=dict)

    def add_node(self, node: DAGNode):
        self.nodes[node.id] = node

    def add_edge(self, src: str, dst: str, tensor_name: str,
                 tensor_shape: list):
        self.edges.append(DAGEdge(src, dst, tensor_name, tensor_shape))

    def predecessors(self, node_id: str) -> List[str]:
        return [e.src for e in self.edges if e.dst == node_id]

    def successors(self, node_id: str) -> List[str]:
        return [e.dst for e in self.edges if e.src == node_id]

    def topological_order(self) -> List[str]:
        """Return node ids in topological order."""
        in_degree = {nid: 0 for nid in self.nodes}
        for e in self.edges:
            if e.dst in in_degree:
                in_degree[e.dst] += 1

        queue = [nid for nid, d in in_degree.items() if d == 0]
        order = []
        while queue:
            # Sort for determinism
            queue.sort()
            nid = queue.pop(0)
            order.append(nid)
            for e in self.edges:
                if e.src == nid and e.dst in in_degree:
                    in_degree[e.dst] -= 1
                    if in_degree[e.dst] == 0:
                        queue.append(e.dst)
        return order

    def parallel_groups(self) -> List[List[str]]:
        """
        Identify groups of nodes that can execute concurrently.

        Two nodes can be parallel if neither is an ancestor of the other.
        Returns a list of "levels" (like a BFS wavefront) where all nodes
        in the same level are independent.
        """
        in_degree = {nid: 0 for nid in self.nodes}
        for e in self.edges:
            if e.dst in in_degree:
                in_degree[e.dst] += 1

        levels = []
        remaining = dict(in_degree)

        while remaining:
            # Find all nodes with in_degree 0 in remaining
            level = sorted([nid for nid, d in remaining.items() if d == 0])
            if not level:
                break  # cycle detected (shouldn't happen in a DAG)
            levels.append(level)
            for nid in level:
                del remaining[nid]
                for e in self.edges:
                    if e.src == nid and e.dst in remaining:
                        remaining[e.dst] -= 1

        return levels

    def to_dict(self) -> dict:
        topo = self.topological_order()
        levels = self.parallel_groups()

        result = {
            'dag': {
                'model_name': self.model_name,
                'num_nodes': len(self.nodes),
                'num_edges': len(self.edges),
                'topological_order': topo,
                'parallel_levels': [
                    {'level': i, 'nodes': lvl}
                    for i, lvl in enumerate(levels)
                ],
                'nodes': [self.nodes[nid].to_dict() for nid in topo],
                'edges': [e.to_dict() for e in self.edges],
            }
        }

        if self.repeat_blocks:
            result['dag']['repeat_blocks'] = self.repeat_blocks

        return result

    def save_yaml(self, path: str):
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False,
                      sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str) -> 'NetworkDAG':
        with open(path) as f:
            data = yaml.safe_load(f)
        d = data['dag']
        dag = cls(model_name=d['model_name'])
        for nd in d['nodes']:
            dag.add_node(DAGNode(
                id=nd['id'],
                op_type=nd['op_type'],
                yaml_file=nd['yaml_file'],
                tensor_dims=nd.get('tensor_dims', {}),
            ))
        for ed in d['edges']:
            dag.add_edge(ed['src'], ed['dst'],
                         ed['tensor_name'], ed['tensor_shape'])
        dag.repeat_blocks = d.get('repeat_blocks', {})
        return dag


# ============================================================
# LLM DAG builders
# ============================================================

def build_llm_decoder_layer_dag(
    cfg,
    layer_idx: int,
    batch_size: int,
    seq_len: int,
    kv_cache_len: Optional[int] = None,
) -> Tuple[NetworkDAG, str, str]:
    """
    Build the DAG for one decoder layer of a dense LLM.

    Returns (dag, input_node_id, output_node_id) so layers can be chained.

    Structure:
        input
        ├──→ q_proj ──────────────┐
        ├──→ k_proj ──────────────┤
        └──→ v_proj ───────────┐  │
                               │  ↓
                          attn_qk
                               │
                          attn_v ←┘
                               │
                          o_proj
                               │
                     attn_residual_add ← input (skip)
                               │
                ├──→ gate_proj ──────┐
                └──→ up_proj ────────┤  (SwiGLU)
                                     │
                               down_proj
                                     │
                          mlp_residual_add ← attn_residual_add (skip)
                                     │
                                  output
    """
    H = cfg.hidden_size
    n_h = cfg.num_attention_heads
    n_kv = cfg.num_key_value_heads
    d = cfg.head_dim
    I = cfg.intermediate_size
    B = batch_size
    S = seq_len
    Sq = S
    Sk = kv_cache_len if kv_cache_len is not None else S
    L = f"layer{layer_idx}"

    dag = NetworkDAG(model_name=f"{cfg.model_name}_{L}")
    hidden_shape = [B, S, H]

    # Input placeholder
    inp = f"{L}_input"
    dag.add_node(DAGNode(inp, 'placeholder', '', {'B': B, 'N': S, 'C': H}))

    # --- Attention projections (q, k, v are parallel) ---
    q_id = f"{L}_q_proj"
    k_id = f"{L}_k_proj"
    v_id = f"{L}_v_proj"
    dag.add_node(DAGNode(q_id, 'linear', f"{L}_q_proj.yaml",
                         {'B': B, 'N': S, 'C': H, 'M': n_h * d}))
    dag.add_node(DAGNode(k_id, 'linear', f"{L}_k_proj.yaml",
                         {'B': B, 'N': S, 'C': H, 'M': n_kv * d}))
    dag.add_node(DAGNode(v_id, 'linear', f"{L}_k_proj.yaml",   # same dims as k
                         {'B': B, 'N': S, 'C': H, 'M': n_kv * d}))

    dag.add_edge(inp, q_id, 'hidden_states', hidden_shape)
    dag.add_edge(inp, k_id, 'hidden_states', hidden_shape)
    dag.add_edge(inp, v_id, 'hidden_states', hidden_shape)

    # --- Attention QK ---
    qk_id = f"{L}_attn_qk"
    dag.add_node(DAGNode(qk_id, 'attn_qk', f"{L}_attn_qk.yaml",
                         {'B': B, 'H': n_h, 'Q': Sq, 'K': Sk, 'D': d}))
    dag.add_edge(q_id, qk_id, 'query', [B, n_h, Sq, d])
    dag.add_edge(k_id, qk_id, 'key', [B, n_h, Sk, d])

    # --- Attention V ---
    av_id = f"{L}_attn_v"
    dag.add_node(DAGNode(av_id, 'attn_v', f"{L}_attn_v.yaml",
                         {'B': B, 'H': n_h, 'Q': Sq, 'K': Sk, 'D': d}))
    dag.add_edge(qk_id, av_id, 'scores', [B, n_h, Sq, Sk])
    dag.add_edge(v_id, av_id, 'value', [B, n_h, Sk, d])

    # --- O projection ---
    o_id = f"{L}_o_proj"
    dag.add_node(DAGNode(o_id, 'linear', f"{L}_o_proj.yaml",
                         {'B': B, 'N': S, 'C': n_h * d, 'M': H}))
    dag.add_edge(av_id, o_id, 'attn_output', [B, S, n_h * d])

    # --- Attention residual add ---
    attn_res_id = f"{L}_attn_residual_add"
    dag.add_node(DAGNode(attn_res_id, 'residual_add', '',
                         {'B': B, 'N': S, 'C': H}))
    dag.add_edge(o_id, attn_res_id, 'o_proj_output', hidden_shape)
    dag.add_edge(inp, attn_res_id, 'residual', hidden_shape)  # skip connection

    # --- MLP: gate_proj and up_proj are parallel ---
    gate_id = f"{L}_gate_proj"
    up_id = f"{L}_up_proj"
    dag.add_node(DAGNode(gate_id, 'linear', f"{L}_gate_proj.yaml",
                         {'B': B, 'N': S, 'C': H, 'M': I}))
    dag.add_node(DAGNode(up_id, 'linear', f"{L}_gate_proj.yaml",  # same dims
                         {'B': B, 'N': S, 'C': H, 'M': I}))
    dag.add_edge(attn_res_id, gate_id, 'post_attn', hidden_shape)
    dag.add_edge(attn_res_id, up_id, 'post_attn', hidden_shape)

    # --- Down projection ---
    down_id = f"{L}_down_proj"
    dag.add_node(DAGNode(down_id, 'linear', f"{L}_down_proj.yaml",
                         {'B': B, 'N': S, 'C': I, 'M': H}))
    dag.add_edge(gate_id, down_id, 'gate_output', [B, S, I])
    dag.add_edge(up_id, down_id, 'up_output', [B, S, I])

    # --- MLP residual add ---
    mlp_res_id = f"{L}_mlp_residual_add"
    dag.add_node(DAGNode(mlp_res_id, 'residual_add', '',
                         {'B': B, 'N': S, 'C': H}))
    dag.add_edge(down_id, mlp_res_id, 'down_proj_output', hidden_shape)
    dag.add_edge(attn_res_id, mlp_res_id, 'residual', hidden_shape)  # skip

    return dag, inp, mlp_res_id


def build_llm_moe_decoder_layer_dag(
    cfg,
    layer_idx: int,
    batch_size: int,
    seq_len: int,
    kv_cache_len: Optional[int] = None,
) -> Tuple[NetworkDAG, str, str]:
    """
    Build the DAG for one MoE decoder layer.

    Same attention block as dense, but MLP is replaced by:
        router → N parallel experts (each: gate_proj || up_proj → down_proj)
        → expert combine → residual add
    """
    H = cfg.hidden_size
    n_h = cfg.num_attention_heads
    n_kv = cfg.num_key_value_heads
    d = cfg.head_dim
    E_I = cfg.moe_intermediate_size
    n_exp = cfg.num_experts
    top_k = cfg.num_experts_per_tok
    B = batch_size
    S = seq_len
    Sq = S
    Sk = kv_cache_len if kv_cache_len is not None else S
    L = f"layer{layer_idx}"

    tokens_per_expert = max((B * S * top_k) // n_exp, 1)

    dag = NetworkDAG(model_name=f"{cfg.model_name}_{L}")
    hidden_shape = [B, S, H]

    # Input placeholder
    inp = f"{L}_input"
    dag.add_node(DAGNode(inp, 'placeholder', '', {'B': B, 'N': S, 'C': H}))

    # --- Attention (same as dense) ---
    q_id = f"{L}_q_proj"
    k_id = f"{L}_k_proj"
    v_id = f"{L}_v_proj"
    dag.add_node(DAGNode(q_id, 'linear', f"q_proj.yaml",
                         {'B': B, 'N': S, 'C': H, 'M': n_h * d}))
    dag.add_node(DAGNode(k_id, 'linear', f"k_proj.yaml",
                         {'B': B, 'N': S, 'C': H, 'M': n_kv * d}))
    dag.add_node(DAGNode(v_id, 'linear', f"k_proj.yaml",
                         {'B': B, 'N': S, 'C': H, 'M': n_kv * d}))
    dag.add_edge(inp, q_id, 'hidden_states', hidden_shape)
    dag.add_edge(inp, k_id, 'hidden_states', hidden_shape)
    dag.add_edge(inp, v_id, 'hidden_states', hidden_shape)

    qk_id = f"{L}_attn_qk"
    dag.add_node(DAGNode(qk_id, 'attn_qk', f"attn_qk.yaml",
                         {'B': B, 'H': n_h, 'Q': Sq, 'K': Sk, 'D': d}))
    dag.add_edge(q_id, qk_id, 'query', [B, n_h, Sq, d])
    dag.add_edge(k_id, qk_id, 'key', [B, n_h, Sk, d])

    av_id = f"{L}_attn_v"
    dag.add_node(DAGNode(av_id, 'attn_v', f"attn_v.yaml",
                         {'B': B, 'H': n_h, 'Q': Sq, 'K': Sk, 'D': d}))
    dag.add_edge(qk_id, av_id, 'scores', [B, n_h, Sq, Sk])
    dag.add_edge(v_id, av_id, 'value', [B, n_h, Sk, d])

    o_id = f"{L}_o_proj"
    dag.add_node(DAGNode(o_id, 'linear', f"o_proj.yaml",
                         {'B': B, 'N': S, 'C': n_h * d, 'M': H}))
    dag.add_edge(av_id, o_id, 'attn_output', [B, S, n_h * d])

    attn_res_id = f"{L}_attn_residual_add"
    dag.add_node(DAGNode(attn_res_id, 'residual_add', '',
                         {'B': B, 'N': S, 'C': H}))
    dag.add_edge(o_id, attn_res_id, 'o_proj_output', hidden_shape)
    dag.add_edge(inp, attn_res_id, 'residual', hidden_shape)

    # --- Router ---
    router_id = f"{L}_router"
    dag.add_node(DAGNode(router_id, 'linear', f"router.yaml",
                         {'B': B, 'N': S, 'C': H, 'M': n_exp}))
    dag.add_edge(attn_res_id, router_id, 'post_attn', hidden_shape)

    # --- Experts (all parallel, each with gate || up → down) ---
    expert_down_ids = []
    expert_shape = [B, tokens_per_expert, E_I]
    for exp_idx in range(n_exp):
        E = f"{L}_expert{exp_idx}"
        g_id = f"{E}_gate_proj"
        u_id = f"{E}_up_proj"
        d_id = f"{E}_down_proj"

        dag.add_node(DAGNode(g_id, 'linear', f"expert_gate_proj.yaml",
                             {'B': B, 'N': tokens_per_expert, 'C': H, 'M': E_I}))
        dag.add_node(DAGNode(u_id, 'linear', f"expert_gate_proj.yaml",
                             {'B': B, 'N': tokens_per_expert, 'C': H, 'M': E_I}))
        dag.add_node(DAGNode(d_id, 'linear', f"expert_down_proj.yaml",
                             {'B': B, 'N': tokens_per_expert, 'C': E_I, 'M': H}))

        # Router dispatches tokens to experts
        dag.add_edge(router_id, g_id, 'routed_tokens', [B, tokens_per_expert, H])
        dag.add_edge(router_id, u_id, 'routed_tokens', [B, tokens_per_expert, H])
        dag.add_edge(g_id, d_id, 'gate_output', expert_shape)
        dag.add_edge(u_id, d_id, 'up_output', expert_shape)

        expert_down_ids.append(d_id)

    # --- Expert combine + residual ---
    combine_id = f"{L}_moe_combine"
    dag.add_node(DAGNode(combine_id, 'moe_combine', '',
                         {'B': B, 'N': S, 'C': H}))
    for d_id in expert_down_ids:
        dag.add_edge(d_id, combine_id, 'expert_output',
                     [B, tokens_per_expert, H])

    mlp_res_id = f"{L}_mlp_residual_add"
    dag.add_node(DAGNode(mlp_res_id, 'residual_add', '',
                         {'B': B, 'N': S, 'C': H}))
    dag.add_edge(combine_id, mlp_res_id, 'moe_output', hidden_shape)
    dag.add_edge(attn_res_id, mlp_res_id, 'residual', hidden_shape)

    return dag, inp, mlp_res_id


def build_full_llm_dag(
    cfg,
    batch_size: int,
    seq_len: int,
    kv_cache_len: Optional[int] = None,
) -> NetworkDAG:
    """
    Build complete DAG for an LLM, chaining all decoder layers + lm_head.

    For efficiency, only layer 0 is fully expanded in the DAG.
    A 'repeat_blocks' annotation tells Mozart that layers 1..N-1 are identical.
    """
    full_dag = NetworkDAG(model_name=cfg.model_name)
    H = cfg.hidden_size
    V = cfg.vocab_size
    B = batch_size
    S = seq_len
    hidden_shape = [B, S, H]

    prev_output = None

    # Build layer 0 fully
    if cfg.is_moe:
        layer_dag, layer_inp, layer_out = build_llm_moe_decoder_layer_dag(
            cfg, 0, batch_size, seq_len, kv_cache_len)
    else:
        layer_dag, layer_inp, layer_out = build_llm_decoder_layer_dag(
            cfg, 0, batch_size, seq_len, kv_cache_len)

    # Merge layer 0 into full DAG
    for nid, node in layer_dag.nodes.items():
        full_dag.add_node(node)
    for edge in layer_dag.edges:
        full_dag.edges.append(edge)

    # Annotate repetition
    if cfg.num_hidden_layers > 1:
        full_dag.repeat_blocks['decoder_layer'] = {
            'template_prefix': 'layer0_',
            'repeat_count': cfg.num_hidden_layers,
            'chain_input': layer_inp,
            'chain_output': layer_out,
            'description': (
                f"Layers 0..{cfg.num_hidden_layers - 1} are identical. "
                f"Only layer 0 is expanded. Each layer's output feeds the "
                f"next layer's input via the residual stream."
            ),
        }

    # LM head
    lm_id = "lm_head"
    full_dag.add_node(DAGNode(lm_id, 'linear', "lm_head.yaml",
                              {'B': B, 'N': S, 'C': H, 'M': V}))
    full_dag.add_edge(layer_out, lm_id, 'final_hidden', hidden_shape)

    return full_dag


# ============================================================
# CNN DAG builders (from torch.fx trace)
# ============================================================

def build_cnn_dag_from_fx(model_name: str,
                          layer_data: list,
                          fx_graph) -> NetworkDAG:
    """
    Build a DAG from a torch.fx traced graph and extracted layer data.

    The fx graph contains the true data flow including skip connections,
    parallel branches, and concatenations.

    Args:
        model_name: name for the DAG
        layer_data: list of LayerDescription objects from the Converter
        fx_graph: the torch.fx GraphModule
    """
    dag = NetworkDAG(model_name=model_name)

    # Map fx node names to layer indices
    # The Converter appends to summary in execution order, but the fx graph
    # has the full dependency information.
    graph = fx_graph.graph

    # Build a map from fx node name → output tensor name used in converter
    # and a map from output tensor name → DAG node id
    node_output_map = {}   # fx_node_name → dag_node_id
    layer_idx = 0

    for fx_node in graph.nodes:
        if fx_node.op == 'placeholder':
            dag_id = f"input_{fx_node.name}"
            dag.add_node(DAGNode(dag_id, 'placeholder', '', {}))
            node_output_map[fx_node.name] = dag_id

        elif fx_node.op == 'call_module':
            # Check if this module produced a layer in our summary
            if layer_idx < len(layer_data):
                ld = layer_data[layer_idx]
                # Match by checking if the fx target appears in the layer name
                # The converter names layers by their module target path
                target_parts = fx_node.target.split('.')
                name_match = any(
                    part in ld.name for part in target_parts
                    if len(part) > 2
                )

                if name_match or fx_node.target in ld.name or ld.name in fx_node.target:
                    dag_id = f"layer{layer_idx + 1}_{ld.name}"
                    yaml_file = f"layer{layer_idx + 1}_{ld.name}.yaml"

                    # Determine op_type and dims
                    op_type, dims = _classify_layer(ld)
                    dag.add_node(DAGNode(dag_id, op_type, yaml_file, dims))
                    node_output_map[fx_node.name] = dag_id
                    layer_idx += 1

                    # Add edges from predecessors
                    for arg in fx_node.args:
                        if hasattr(arg, 'name') and arg.name in node_output_map:
                            src_id = node_output_map[arg.name]
                            dag.add_edge(src_id, dag_id,
                                         f"{arg.name}_to_{fx_node.name}", [])
                    continue

            # Module not in summary (bypassed/ignored) — propagate mapping
            if fx_node.args and hasattr(fx_node.args[0], 'name'):
                src_name = fx_node.args[0].name
                if src_name in node_output_map:
                    node_output_map[fx_node.name] = node_output_map[src_name]

        elif fx_node.op == 'call_function':
            # For add/mul/cat — these create merge points in the DAG
            import operator as op_module
            import torch
            import torch.nn.functional as F

            merge_ops = {
                op_module.add: 'elementwise_add',
                torch.add: 'elementwise_add',
                op_module.mul: 'elementwise_mul',
                torch.mul: 'elementwise_mul',
                torch.cat: 'concat',
            }

            if fx_node.target in merge_ops:
                dag_id = f"merge_{fx_node.name}"
                dag.add_node(DAGNode(dag_id, merge_ops[fx_node.target],
                                     '', {}))
                node_output_map[fx_node.name] = dag_id

                for arg in fx_node.args:
                    if hasattr(arg, 'name') and arg.name in node_output_map:
                        src_id = node_output_map[arg.name]
                        dag.add_edge(src_id, dag_id,
                                     f"{arg.name}_to_{fx_node.name}", [])
                    elif isinstance(arg, (list, tuple)):
                        for sub_arg in arg:
                            if hasattr(sub_arg, 'name') and sub_arg.name in node_output_map:
                                src_id = node_output_map[sub_arg.name]
                                dag.add_edge(src_id, dag_id,
                                             f"{sub_arg.name}_to_{fx_node.name}", [])
            else:
                # Propagate from first tensor arg
                for arg in fx_node.args:
                    if hasattr(arg, 'name') and arg.name in node_output_map:
                        node_output_map[fx_node.name] = node_output_map[arg.name]
                        break

        elif fx_node.op == 'output':
            pass
        else:
            # get_attr, etc. — propagate
            if fx_node.args and hasattr(fx_node.args[0], 'name'):
                src_name = fx_node.args[0].name
                if src_name in node_output_map:
                    node_output_map[fx_node.name] = node_output_map[src_name]

    return dag


def _classify_layer(ld) -> Tuple[str, dict]:
    """Classify a LayerDescription into op_type and dimension dict."""
    from pytorch2timeloop.utils.layer_descriptions import (
        ConvLayerDescription,
        MaxPoolLayerDescription,
        MatmulFuncDescription,
    )

    if isinstance(ld, ConvLayerDescription):
        if ld.g > 1 and ld.g == ld.c * ld.g:
            # depthwise: groups == in_channels
            op_type = 'depthwise_conv2d'
        elif ld.s == 1 and ld.r == 1:
            op_type = 'pointwise_conv2d'
        else:
            op_type = 'conv2d'

        dims = {
            'N': ld.n, 'G': ld.g, 'C': ld.c // ld.g,
            'M': ld.m // ld.g, 'R': ld.r, 'S': ld.s,
            'P': ld.p, 'Q': ld.q
        }
        return op_type, dims

    elif isinstance(ld, MaxPoolLayerDescription):
        dims = {
            'N': ld.n, 'C': ld.c, 'R': ld.r, 'S': ld.s,
            'P': ld.p, 'Q': ld.q
        }
        return 'pool', dims

    elif isinstance(ld, MatmulFuncDescription):
        dims = {'M': ld.m, 'N': ld.n, 'K': ld.k}
        return 'matmul', dims

    else:
        return 'unknown', {}
