"""
Visualize the computational graph built by automatic differentiation.

Produces a Graphviz diagram showing:
  - Tensor nodes (rectangles): value and gradient
  - Operation nodes (ellipses): +, *, @, relu, etc.
  - Edges: which tensors flow into which operations

Usage:
    from autograd.viz import draw_graph
    loss = (a * b + c).relu().sum()
    loss.backward()
    draw_graph(loss, filename="my_graph")  # produces my_graph.png
"""

import numpy as np
from graphviz import Digraph


def draw_graph(
    root,
    filename="comp_graph",
    format="png",
    rankdir="TB",
    show_data=True,
    show_grad=True,
    max_elements=6,
    param_names=None,
    dpi="150",
):
    """
    Draw the computational graph ending at `root`.

    Args:
        root:         The output Tensor (e.g. the loss).
        filename:     Output file name (without extension).
        format:       'png', 'svg', or 'pdf'.
        rankdir:      'TB' (top-to-bottom) or 'LR' (left-to-right).
        show_data:    Show tensor values in the nodes.
        show_grad:    Show gradient values in the nodes.
        max_elements: Max number of array elements to display.
        param_names:  Optional dict {id(tensor): "name"} for labeling.
        dpi:          Resolution for PNG output.

    Returns:
        The graphviz.Digraph object (also saves to file).
    """
    if param_names is None:
        param_names = {}

    # Auto-detect Parameter names
    _collect_param_names(root, param_names, set())

    dot = Digraph(
        format=format,
        graph_attr={
            "rankdir": rankdir,
            "dpi": dpi,
            "bgcolor": "#ffffff",
            "fontname": "Helvetica",
            "pad": "0.5",
        },
        node_attr={"fontname": "Helvetica", "fontsize": "11"},
        edge_attr={"fontname": "Helvetica", "fontsize": "9"},
    )

    visited = set()

    def _fmt_array(arr, max_el=max_elements):
        """Format a numpy array compactly for display."""
        if arr.size == 1:
            return f"{arr.flat[0]:.4g}"
        if arr.size <= max_el:
            flat = ", ".join(f"{v:.3g}" for v in arr.flat)
            return f"[{flat}]"
        flat_start = ", ".join(f"{v:.3g}" for v in arr.flat[: max_el // 2])
        flat_end = ", ".join(f"{v:.3g}" for v in arr.flat[-max_el // 2 :])
        return f"[{flat_start}, …, {flat_end}]"

    def _tensor_label(t):
        """Build the HTML label for a tensor node."""
        tid = id(t)
        name = param_names.get(tid, "")

        rows = []

        # Header row with name or shape
        if name:
            rows.append(
                f'<TR><TD COLSPAN="2"><B>{name}</B></TD></TR>'
            )
        rows.append(
            f'<TR><TD COLSPAN="2"><FONT COLOR="#666666">shape: {t.data.shape}</FONT></TD></TR>'
        )

        if show_data:
            rows.append(
                f'<TR><TD ALIGN="LEFT"><B>data</B></TD>'
                f'<TD ALIGN="LEFT">{_fmt_array(t.data)}</TD></TR>'
            )
        if show_grad and np.any(t.grad != 0):
            rows.append(
                f'<TR><TD ALIGN="LEFT"><FONT COLOR="#cc4444"><B>grad</B></FONT></TD>'
                f'<TD ALIGN="LEFT"><FONT COLOR="#cc4444">{_fmt_array(t.grad)}</FONT></TD></TR>'
            )

        html = '<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="2">'
        html += "".join(rows)
        html += "</TABLE>>"
        return html

    def _tensor_style(t):
        """Pick color based on tensor role."""
        tid = id(t)
        if tid in param_names:
            # Named parameter — blue
            return {"fillcolor": "#dae8fc", "color": "#6c8ebf", "style": "filled,bold"}
        if len(t._prev) == 0:
            # Leaf / input — green
            return {"fillcolor": "#d5e8d4", "color": "#82b366", "style": "filled"}
        if t is root:
            # Output / loss — orange
            return {"fillcolor": "#fff2cc", "color": "#d6b656", "style": "filled,bold"}
        # Intermediate
        return {"fillcolor": "#f5f5f5", "color": "#999999", "style": "filled"}

    def _add_nodes(t):
        if id(t) in visited:
            return
        visited.add(id(t))

        tid = str(id(t))

        # Tensor node
        dot.node(
            tid,
            label=_tensor_label(t),
            shape="Mrecord",
            **_tensor_style(t),
        )

        # If this tensor was produced by an operation, add the op node
        if t._op:
            op_id = tid + "_op"
            dot.node(
                op_id,
                label=t._op,
                shape="ellipse",
                fillcolor="#e1d5e7",
                color="#9673a6",
                style="filled",
                fontsize="12",
                width="0.6",
                height="0.4",
            )
            # Edge: op → output tensor
            dot.edge(op_id, tid)

            # Edges: input tensors → op
            for child in t._prev:
                _add_nodes(child)
                dot.edge(str(id(child)), op_id)
        else:
            # Leaf node — no parent op
            pass

    _add_nodes(root)

    dot.render(filename, cleanup=True)
    return dot


def _collect_param_names(tensor, names, visited):
    """Walk the graph and auto-detect Parameter objects by their .name attribute."""
    if id(tensor) in visited:
        return
    visited.add(id(tensor))

    # Check if it's a Parameter (duck-typed)
    if hasattr(tensor, "name") and tensor.name and id(tensor) not in names:
        names[id(tensor)] = tensor.name

    for child in tensor._prev:
        _collect_param_names(child, names, visited)