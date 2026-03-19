#!/usr/bin/env python3
"""
Visualize DAG.yaml files as publication-quality SVG/PDF diagrams.

Uses graphviz `dot` directly (no Python graphviz package needed).
Generates one diagram per DAG with:
  - Color-coded operator types
  - Parallel branches visually grouped
  - Skip/residual connections as dashed red edges
  - Tensor shapes annotated on edges
  - Repeat blocks shown with a border annotation

Usage:
    python visualize_dag.py                          # all DAGs in dag_workloads/
    python visualize_dag.py --dag dag_workloads/llama3.1_8b_prefill_s2048/DAG.yaml
    python visualize_dag.py --format pdf             # PDF instead of SVG
    python visualize_dag.py --compact                # compact mode (no edge labels)
"""

import argparse
import os
import subprocess
import sys
import yaml
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
DAG_DIR = SCRIPT_DIR / "dag_workloads"

# Color palette for operator types
OP_COLORS = {
    'placeholder':      '#E8E8E8',  # light gray
    'linear':           '#4A90D9',  # blue
    'conv2d':           '#50B86C',  # green
    'pointwise_conv2d': '#7BC67B',  # light green
    'depthwise_conv2d': '#2D8A4E',  # dark green
    'attn_qk':          '#E8963E',  # orange
    'attn_v':           '#D4722C',  # dark orange
    'residual_add':     '#E85454',  # red
    'elementwise_add':  '#E87070',  # light red
    'elementwise_mul':  '#C75050',  # dark red
    'moe_combine':      '#9B59B6',  # purple
    'pool':             '#95A5A6',  # gray
    'concat':           '#F39C12',  # yellow
    'unknown':          '#BDC3C7',  # silver
}

OP_SHAPES = {
    'placeholder':      'ellipse',
    'linear':           'box',
    'conv2d':           'box',
    'pointwise_conv2d': 'box',
    'depthwise_conv2d': 'box',
    'attn_qk':          'diamond',
    'attn_v':           'diamond',
    'residual_add':     'circle',
    'elementwise_add':  'circle',
    'elementwise_mul':  'circle',
    'moe_combine':      'hexagon',
    'pool':             'box',
    'concat':           'trapezium',
}


def _short_label(node_id: str, op_type: str, tensor_dims: dict,
                 compact: bool) -> str:
    """Generate a concise node label."""
    # Strip layer prefix for readability
    label = node_id
    for prefix in ['layer0_', 'stages_', 'block']:
        if label.startswith(prefix):
            label = label[len(prefix):]

    if compact:
        return label

    # Add key dims
    if op_type == 'linear' and 'C' in tensor_dims and 'M' in tensor_dims:
        label += f"\\n{tensor_dims['C']}x{tensor_dims['M']}"
    elif op_type in ('conv2d', 'pointwise_conv2d', 'depthwise_conv2d'):
        if 'R' in tensor_dims and 'S' in tensor_dims:
            r, s = tensor_dims['R'], tensor_dims['S']
            c = tensor_dims.get('C', '?')
            m = tensor_dims.get('M', '?')
            label += f"\\n{c}->{m} {r}x{s}"
    elif op_type in ('attn_qk', 'attn_v'):
        if 'Q' in tensor_dims and 'K' in tensor_dims:
            label += f"\\n{tensor_dims['Q']}x{tensor_dims['K']}"
    elif op_type == 'residual_add':
        label = "+"
    elif op_type == 'elementwise_add':
        label = "+"
    elif op_type == 'elementwise_mul':
        label = "x"

    return label


def _edge_label(tensor_name: str, tensor_shape: list,
                compact: bool) -> str:
    """Generate edge label."""
    if compact:
        return ""

    if not tensor_shape:
        return ""

    # Short shape string
    shape_str = "x".join(str(s) for s in tensor_shape)

    # Simplify tensor name
    name = tensor_name
    for strip in ['hidden_states', 'tensor_name']:
        if name == strip:
            name = ""

    if name and len(name) > 15:
        name = name[:12] + "..."

    if name:
        return f"{name}\\n[{shape_str}]"
    return f"[{shape_str}]"


def dag_to_dot(dag_data: dict, compact: bool = False,
               title: str = None) -> str:
    """Convert DAG dict to graphviz DOT format."""
    d = dag_data['dag']
    model_name = d['model_name']

    lines = []
    lines.append('digraph G {')
    lines.append(f'  label="{title or model_name}";')
    lines.append('  labelloc="t";')
    lines.append('  fontsize=16;')
    lines.append('  fontname="Helvetica";')
    lines.append('  rankdir=TB;')
    lines.append('  nodesep=0.4;')
    lines.append('  ranksep=0.6;')
    lines.append('  node [fontname="Helvetica", fontsize=10, '
                 'style="filled,rounded", penwidth=1.5];')
    lines.append('  edge [fontname="Helvetica", fontsize=8, '
                 'color="#555555"];')
    lines.append('')

    # Build node lookup
    node_map = {n['id']: n for n in d['nodes']}

    # Use parallel_levels for rank constraints
    if 'parallel_levels' in d:
        for level_info in d['parallel_levels']:
            level_nodes = level_info['nodes']
            if len(level_nodes) > 1:
                # Force same rank for parallel nodes
                node_ids = ' '.join(
                    f'"{nid}"' for nid in level_nodes
                    if nid in node_map
                )
                if node_ids:
                    lines.append(f'  {{ rank=same; {node_ids} }}')

    lines.append('')

    # Nodes
    for node in d['nodes']:
        nid = node['id']
        op_type = node.get('op_type', 'unknown')
        dims = node.get('tensor_dims', {})

        color = OP_COLORS.get(op_type, OP_COLORS['unknown'])
        shape = OP_SHAPES.get(op_type, 'box')
        label = _short_label(nid, op_type, dims, compact)

        # Make residual_add nodes small
        if op_type in ('residual_add', 'elementwise_add', 'elementwise_mul'):
            lines.append(
                f'  "{nid}" [label="{label}", shape={shape}, '
                f'fillcolor="{color}", width=0.3, height=0.3, '
                f'fixedsize=true, fontsize=14];'
            )
        elif op_type == 'placeholder':
            lines.append(
                f'  "{nid}" [label="{label}", shape={shape}, '
                f'fillcolor="{color}", style="filled"];'
            )
        else:
            lines.append(
                f'  "{nid}" [label="{label}", shape={shape}, '
                f'fillcolor="{color}", fontcolor="white"];'
            )

    lines.append('')

    # Edges
    for edge in d['edges']:
        src = edge['src']
        dst = edge['dst']
        tname = edge.get('tensor_name', '')
        tshape = edge.get('tensor_shape', [])

        label = _edge_label(tname, tshape, compact)

        # Style residual/skip edges differently
        is_skip = 'residual' in tname
        if is_skip:
            style = 'dashed'
            color = '#E85454'
            penwidth = '2.0'
        else:
            style = 'solid'
            color = '#555555'
            penwidth = '1.0'

        if label and not compact:
            lines.append(
                f'  "{src}" -> "{dst}" [label="{label}", '
                f'style={style}, color="{color}", penwidth={penwidth}];'
            )
        else:
            lines.append(
                f'  "{src}" -> "{dst}" ['
                f'style={style}, color="{color}", penwidth={penwidth}];'
            )

    # Repeat block annotation
    if 'repeat_blocks' in d:
        for name, info in d['repeat_blocks'].items():
            count = info.get('repeat_count', 1)
            lines.append('')
            lines.append(f'  // Repeat block: {name} x{count}')
            # Add a note node
            lines.append(
                f'  "repeat_note" [label="Repeated x{count} layers", '
                f'shape=note, fillcolor="#FFFFCC", style="filled", '
                f'fontsize=11, fontcolor="#666666"];'
            )
            # Connect to the output of the repeated block
            chain_out = info.get('chain_output', '')
            if chain_out:
                lines.append(
                    f'  "{chain_out}" -> "repeat_note" '
                    f'[style=dotted, color="#999999", arrowhead=none];'
                )

    # Legend
    lines.append('')
    lines.append('  subgraph cluster_legend {')
    lines.append('    label="Legend";')
    lines.append('    fontsize=11;')
    lines.append('    style="rounded";')
    lines.append('    color="#CCCCCC";')
    lines.append('    bgcolor="#F8F8F8";')

    legend_ops = []
    seen_types = {n.get('op_type') for n in d['nodes']}
    for op in ['linear', 'conv2d', 'pointwise_conv2d', 'depthwise_conv2d',
               'attn_qk', 'attn_v', 'residual_add', 'elementwise_mul',
               'moe_combine']:
        if op in seen_types:
            legend_ops.append(op)

    for i, op in enumerate(legend_ops):
        color = OP_COLORS.get(op, '#BDC3C7')
        shape = OP_SHAPES.get(op, 'box')
        display = op.replace('_', ' ')
        lines.append(
            f'    "leg_{op}" [label="{display}", shape={shape}, '
            f'fillcolor="{color}", fontcolor="white", '
            f'fontsize=9, width=1.2, height=0.25];'
        )

    # Chain legend nodes invisibly for vertical layout
    if len(legend_ops) > 1:
        chain = ' -> '.join(f'"leg_{op}"' for op in legend_ops)
        lines.append(f'    {chain} [style=invis];')

    # Add skip edge to legend
    if any('residual' in e.get('tensor_name', '') for e in d['edges']):
        lines.append(
            '    "leg_skip_src" [label="", shape=point, width=0.1];'
        )
        lines.append(
            '    "leg_skip_dst" [label="skip conn.", shape=plaintext, '
            'fontsize=9];'
        )
        lines.append(
            '    "leg_skip_src" -> "leg_skip_dst" '
            '[style=dashed, color="#E85454", penwidth=2.0];'
        )
        if legend_ops:
            lines.append(
                f'    "leg_{legend_ops[-1]}" -> "leg_skip_src" [style=invis];'
            )

    lines.append('  }')
    lines.append('}')

    return '\n'.join(lines)


def render_dot(dot_str: str, output_path: str, fmt: str = 'svg'):
    """Render DOT string to file using graphviz dot."""
    cmd = ['dot', f'-T{fmt}', '-o', output_path]
    result = subprocess.run(cmd, input=dot_str, capture_output=True,
                            text=True, timeout=30)
    if result.returncode != 0:
        print(f"  ERROR: dot failed: {result.stderr}")
        return False
    return True


def visualize_dag(dag_path: str, output_dir: str = None,
                  fmt: str = 'svg', compact: bool = False):
    """Load a DAG.yaml and render it."""
    with open(dag_path) as f:
        data = yaml.safe_load(f)

    dag_dir = os.path.dirname(dag_path)
    dir_name = os.path.basename(dag_dir)

    if output_dir is None:
        output_dir = dag_dir

    os.makedirs(output_dir, exist_ok=True)

    # Check if DAG is too large for full visualization
    num_nodes = data['dag']['num_nodes']
    if num_nodes > 50:
        # For large DAGs (MoE), render compact version
        compact = True
        print(f"  Note: {num_nodes} nodes, using compact mode")

    # For very large DAGs (MoE with 396 nodes), render a simplified version
    if num_nodes > 100:
        data = _simplify_moe_dag(data)
        print(f"  Simplified MoE DAG to {data['dag']['num_nodes']} nodes")

    dot_str = dag_to_dot(data, compact=compact, title=dir_name)

    # Save DOT file
    dot_path = os.path.join(output_dir, f"DAG.dot")
    with open(dot_path, 'w') as f:
        f.write(dot_str)

    # Render
    out_path = os.path.join(output_dir, f"DAG.{fmt}")
    ok = render_dot(dot_str, out_path, fmt)
    if ok:
        print(f"  Rendered: {out_path}")
    return ok


def _simplify_moe_dag(data: dict) -> dict:
    """Simplify MoE DAGs by collapsing experts into a single representative."""
    import copy
    d = copy.deepcopy(data)
    dag = d['dag']

    # Find expert nodes (pattern: expert1..N)
    keep_nodes = []
    remove_ids = set()
    expert_count = 0

    for node in dag['nodes']:
        nid = node['id']
        # Keep expert0, remove expert1..N
        if 'expert' in nid and not nid.startswith('layer0_expert0'):
            is_numbered = False
            for i in range(1, 300):
                if f'expert{i}' in nid:
                    is_numbered = True
                    expert_count += 1
                    break
            if is_numbered:
                remove_ids.add(nid)
                continue
        keep_nodes.append(node)

    # Rename expert0 nodes to indicate they represent all experts
    for node in keep_nodes:
        if 'expert0' in node['id']:
            node['id'] = node['id'].replace('expert0', 'expert_repr')
            n_experts = expert_count // 3 + 1  # 3 ops per expert
            # Keep original label info, just note it represents N experts

    # Filter edges
    keep_edges = []
    for edge in dag['edges']:
        src = edge['src']
        dst = edge['dst']
        if src in remove_ids or dst in remove_ids:
            continue
        if 'expert0' in src:
            edge['src'] = src.replace('expert0', 'expert_repr')
        if 'expert0' in dst:
            edge['dst'] = dst.replace('expert0', 'expert_repr')
        keep_edges.append(edge)

    dag['nodes'] = keep_nodes
    dag['edges'] = keep_edges
    dag['num_nodes'] = len(keep_nodes)
    dag['num_edges'] = len(keep_edges)

    # Update topological order
    keep_set = {n['id'] for n in keep_nodes}
    dag['topological_order'] = [
        (nid.replace('expert0', 'expert_repr') if 'expert0' in nid else nid)
        for nid in dag.get('topological_order', [])
        if nid not in remove_ids
    ]
    # Deduplicate (expert0 renamed nodes might duplicate)
    seen = set()
    deduped = []
    for nid in dag['topological_order']:
        if nid not in seen:
            deduped.append(nid)
            seen.add(nid)
    dag['topological_order'] = deduped

    # Update parallel levels
    if 'parallel_levels' in dag:
        new_levels = []
        for lvl in dag['parallel_levels']:
            nodes = []
            seen_lvl = set()
            for nid in lvl['nodes']:
                if nid in remove_ids:
                    continue
                renamed = nid.replace('expert0', 'expert_repr') if 'expert0' in nid else nid
                if renamed not in seen_lvl:
                    nodes.append(renamed)
                    seen_lvl.add(renamed)
            if nodes:
                new_levels.append({'level': len(new_levels), 'nodes': nodes})
        dag['parallel_levels'] = new_levels

    return d


def main():
    parser = argparse.ArgumentParser(
        description="Visualize DAG.yaml files as diagrams"
    )
    parser.add_argument(
        '--dag', type=str, default=None,
        help='Path to a specific DAG.yaml (default: all in dag_workloads/)'
    )
    parser.add_argument(
        '--format', type=str, default='svg', choices=['svg', 'pdf', 'png'],
        help='Output format (default: svg)'
    )
    parser.add_argument(
        '--compact', action='store_true',
        help='Compact mode: no edge labels, shorter node labels'
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Output directory (default: same as DAG.yaml directory)'
    )
    args = parser.parse_args()

    if args.dag:
        visualize_dag(args.dag, args.output_dir, args.format, args.compact)
    else:
        # Find all DAG.yaml files
        dag_files = sorted(DAG_DIR.glob("*/DAG.yaml"))
        if not dag_files:
            print(f"No DAG.yaml files found in {DAG_DIR}/")
            return

        print(f"Found {len(dag_files)} DAG files to visualize")
        for dag_file in dag_files:
            print(f"\nProcessing: {dag_file.parent.name}")
            visualize_dag(str(dag_file), args.output_dir, args.format,
                          args.compact)

    print("\nDone!")


if __name__ == '__main__':
    main()
