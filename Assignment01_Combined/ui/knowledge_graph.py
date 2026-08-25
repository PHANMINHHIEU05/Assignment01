from __future__ import annotations

from html import escape

import streamlit.components.v1 as components
from pyvis.network import Network


GROUP_COLORS = {
    "model": "#38BDF8",
    "feature": "#818CF8",
    "metric": "#A78BFA",
    "prediction": "#22D3EE",
    "observation": "#60A5FA",
    "outcome": "#FBBF24",
    "target": "#34D399",
    "disease": "#34D399",
    "risk": "#FB7185",
    "guidance": "#34D399",
    "source": "#94A3B8",
    "representation": "#C084FC",
    "location": "#F59E0B",
    "factor": "#818CF8",
}


def render_network_graph(nodes, edges, key, height=520):
    try:
        network = Network(
            height=f"{height}px",
            width="100%",
            bgcolor="#111827",
            font_color="#F8FAFC",
            directed=True,
            cdn_resources="in_line",
        )
        network.set_options(
            """
            {
              "interaction": {"hover": true, "dragNodes": true, "zoomView": true, "dragView": true},
              "physics": {
                "enabled": true,
                "stabilization": {"iterations": 120},
                "barnesHut": {"gravitationalConstant": -2600, "springLength": 145, "springConstant": 0.045}
              },
              "edges": {
                "arrows": {"to": {"enabled": true, "scaleFactor": 0.55}},
                "color": {"color": "rgba(148,163,184,0.55)"},
                "font": {"color": "#94A3B8", "size": 10, "strokeWidth": 0}
              },
              "nodes": {
                "borderWidth": 1,
                "shape": "dot",
                "font": {"color": "#F8FAFC", "size": 14, "face": "Inter"}
              }
            }
            """
        )
        for node in nodes:
            group = node.get("group", "feature")
            network.add_node(
                node["id"],
                label=node.get("label", node["id"]),
                title=node.get("title", node.get("label", node["id"])),
                color=GROUP_COLORS.get(group, "#818CF8"),
                size=node.get("size", 18),
            )
        for edge in edges:
            network.add_edge(
                edge["source"],
                edge["target"],
                label=edge.get("label", ""),
                title=edge.get("title", edge.get("label", "")),
            )
        html = network.generate_html(notebook=False)
        components.html(html, height=height + 36, scrolling=False)
    except Exception:
        _render_fallback(nodes, edges, height)


def _render_fallback(nodes, edges, height):
    edge_text = ", ".join(
        f"{escape(str(edge['source']))} -> {escape(str(edge['target']))}"
        for edge in edges[:10]
    )
    node_html = "".join(
        f"<span style='display:inline-block;margin:6px;padding:8px 10px;border:1px solid rgba(255,255,255,.14);border-radius:999px;color:#F8FAFC;background:#111827'>{escape(str(node.get('label', node['id'])))}</span>"
        for node in nodes
    )
    components.html(
        f"""
        <div style="height:{height}px;background:#111827;border:1px solid rgba(255,255,255,.08);border-radius:18px;padding:18px;overflow:auto;font-family:Inter,system-ui">
            <div>{node_html}</div>
            <p style="color:#94A3B8;margin-top:18px">Fallback edge preview: {edge_text}</p>
        </div>
        """,
        height=height + 36,
    )
