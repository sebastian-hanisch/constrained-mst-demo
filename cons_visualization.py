"""Plotly-Abbildungen: Karte mit Baum und Verstößen (Grad, Tiefe), Schwellwert-Ansicht für den Bottleneck, Lagrange-Verlauf, Kosten-Balken, Sweeps, Zahl der bottleneck-optimalen Bäume.
Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht zoomen."""

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import cons_algorithm as A

TREE_COLOR = "#2F6B65"
BAD_COLOR = "#d62728"
DIFF_COLOR = "#ff7f0e"
MISS_COLOR = "rgba(120,120,120,0.6)"
GREY = "rgba(150,150,150,0.28)"
NODE_COLOR = "#4c78a8"
METHOD_COLORS = {"mst": "#7b3fbf", "greedy": "#4c78a8", "lagrange": "#e8a13a", "exact": "#2F6B65", "min": "#2F6B65", "max": "#d62728", "random": "#e8a13a"}
METHOD_LABELS = {"mst": "MST (ohne Grenze)", "greedy": "Greedy + Kantentausch", "lagrange": "Lagrange + Kantentausch", "exact": "Exakt (Branch-and-Bound)", "min": "MST (billigster)",
                 "max": "Teuerster bottleneck-optimaler", "random": "Zufälliger bottleneck-optimaler"}


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height, legend_y=-0.1):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=legend_y), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _map_axes(fig, height=440):
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, scaleanchor="y", scaleratio=1)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)
    return _base(fig, height)


def _height(inst):
    return 340 if inst.kind == "textbook" else 460


def _edges(fig, inst, ids, color, width=2.5, dash="solid", name="", showlegend=False):
    ids = list(ids)
    if not ids:
        return
    xs, ys = [], []
    for i in ids:
        u, v, _w = inst.edges[i]
        xs += [inst.xy[u][0], inst.xy[v][0], None]
        ys += [inst.xy[u][1], inst.xy[v][1], None]
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=color, width=width, dash=dash), name=name, hoverinfo="skip", showlegend=showlegend))


def _points(fig, inst, colors, size=9, texts=None, sizes=None):
    labels = texts if texts is not None else (list(inst.labels) if inst.labels is not None else None)
    fig.add_trace(go.Scatter(x=inst.xy[:, 0], y=inst.xy[:, 1], mode="markers+text" if labels is not None else "markers", text=labels, textposition="top center",
                             marker=dict(size=sizes if sizes is not None else size, color=colors, line=dict(width=1, color="white")), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=[inst.xy[0][0]], y=[inst.xy[0][1]], mode="markers", marker=dict(size=17, symbol="star", color="#2ca02c", line=dict(width=1, color="white")), name="Depot", hoverinfo="skip", showlegend=False))


def build_instance(inst, mst, deg_limit=None, hop_limit=None):
    """Kandidatenkanten grau, MST dick; Knoten, die die Grad- bzw. Hop-Grenze verletzen, rot. Punktgröße wächst mit dem Grad im MST."""
    n, E = inst.n, inst.edges
    deg = A.degrees(n, E, mst)
    depth = A.depths_from(n, E, mst)
    fig = go.Figure()
    _edges(fig, inst, range(inst.m), GREY, 1.0)
    _edges(fig, inst, mst, TREE_COLOR, 3.0, name="MST", showlegend=True)
    bad = [(deg_limit is not None and deg[v] > deg_limit) or (hop_limit is not None and depth[v] > hop_limit) for v in range(n)]
    _points(fig, inst, [BAD_COLOR if b else NODE_COLOR for b in bad], sizes=[7 + 3 * d for d in deg])
    return _map_axes(fig, _height(inst))


def build_degree_view(inst, tree, delta):
    """Der Baum mit Knotengrad als Beschriftung; Knoten über der Grenze rot, Knoten genau an der Grenze orange."""
    n = inst.n
    deg = A.degrees(n, inst.edges, tree)
    fig = go.Figure()
    _edges(fig, inst, range(inst.m), "rgba(150,150,150,0.16)", 1.0)
    _edges(fig, inst, tree, TREE_COLOR, 3.0)
    col = [BAD_COLOR if d > delta else (DIFF_COLOR if d == delta else NODE_COLOR) for d in deg]
    _points(fig, inst, col, sizes=[8 + 3 * d for d in deg], texts=[str(d) for d in deg])
    return _map_axes(fig, _height(inst))


def build_hop_view(inst, tree, hops):
    """Der Baum mit Tiefe (Hops vom Depot) als Beschriftung; Knoten tiefer als die Grenze rot."""
    depth = A.depths_from(inst.n, inst.edges, tree)
    fig = go.Figure()
    _edges(fig, inst, range(inst.m), "rgba(150,150,150,0.16)", 1.0)
    _edges(fig, inst, tree, TREE_COLOR, 3.0)
    top = max(max(depth), 1)
    col = [BAD_COLOR if d > hops else f"hsl({210 - 150 * d / top:.0f},55%,46%)" for d in depth]
    _points(fig, inst, col, size=13, texts=[str(d) for d in depth])
    return _map_axes(fig, _height(inst))


def build_threshold(inst, order, j, labels):
    """Die `j` billigsten Kanten (Kruskal-Reihenfolge `order`); Punktfarbe = Komponente (`labels`), die letzte Kante orange."""
    fig = go.Figure()
    _edges(fig, inst, range(inst.m), "rgba(150,150,150,0.12)", 1.0)
    taken = order[:j]
    _edges(fig, inst, taken[:-1], TREE_COLOR, 2.4)
    if taken:
        _edges(fig, inst, taken[-1:], DIFF_COLOR, 4.0)
    palette = {}
    col = []
    for c in labels:
        palette.setdefault(c, f"hsl({(len(palette) * 137) % 360},58%,46%)")
        col.append(palette[c])
    _points(fig, inst, col, 9)
    return _map_axes(fig, _height(inst))


def build_tree(inst, tree, mst, feasible=True):
    """Der beschränkte Baum: grün, wo er mit dem MST übereinstimmt, orange, wo er abweicht; MST-Kanten, die er nicht hat, grau gestrichelt."""
    fig = go.Figure()
    _edges(fig, inst, range(inst.m), "rgba(150,150,150,0.14)", 1.0)
    if feasible:
        same = [i for i in tree if i in set(mst)]
        diff = [i for i in tree if i not in set(mst)]
        miss = [i for i in mst if i not in set(tree)]
        _edges(fig, inst, miss, MISS_COLOR, 2.0, dash="dash", name="MST-Kante fehlt", showlegend=bool(miss))
        _edges(fig, inst, same, TREE_COLOR, 3.2, name="wie der MST", showlegend=True)
        _edges(fig, inst, diff, DIFF_COLOR, 3.8, name="weicht ab", showlegend=bool(diff))
    _points(fig, inst, NODE_COLOR, 8)
    return _map_axes(fig, _height(inst))


def build_lagrange(hist, mst_cost):
    """Untergrenze L(pi) je Iteration (dünn), beste Untergrenze und beste zulässige Lösung (Obergrenze); MST-Kosten als Linie."""
    its = [h[0] for h in hist]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=its, y=[h[1] for h in hist], mode="lines", line=dict(color="rgba(150,150,150,0.7)", width=1), name="L(π) je Iteration"))
    fig.add_trace(go.Scatter(x=its, y=[h[2] for h in hist], mode="lines", line=dict(color="#e8a13a", width=2.6), name="beste Untergrenze"))
    ub = [h[3] if h[3] < float("inf") else None for h in hist]
    if any(v is not None for v in ub):
        fig.add_trace(go.Scatter(x=its, y=ub, mode="lines", line=dict(color=TREE_COLOR, width=2.6), name="beste zulässige Lösung", connectgaps=False))
    fig.add_hline(y=mst_cost, line=dict(color="#7b3fbf", dash="dash", width=1.5), annotation_text="MST", annotation_position="bottom right")
    fig.update_xaxes(title_text="Iteration")
    fig.update_yaxes(title_text="Kosten")
    return _base(fig, 320, legend_y=-0.3)


def build_cost_bars(costs, mst_cost, order):
    """Kosten je Verfahren als Aufschlag gegen den MST (%); Verfahren ohne gültigen Baum als Text."""
    fig = go.Figure()
    ys = [None if costs.get(k) is None else 100.0 * (costs[k] / mst_cost - 1.0) for k in order]
    fig.add_trace(go.Bar(x=[METHOD_LABELS[k] for k in order], y=[0 if y is None else y for y in ys], marker_color=[METHOD_COLORS[k] for k in order],
                         text=["kein Baum" if y is None else f"{y:+.2f} %" for y in ys], textposition="outside"))
    fig.update_yaxes(title_text="Mehrkosten gegen den MST (%)", rangemode="tozero")
    return _base(fig, 320)


def build_sweep(rows, param_label, series, y_label, log_y=False, ref_line=None, ref_label=None):
    """`series` = [(key, Name, Farbe)]: Median als Linie, 10. bis 90. Perzentil als Band (`<key>_lo`/`<key>_hi`)."""
    xs = [str(r["value"]) for r in rows]
    fig = go.Figure()
    for key, name, color in series:
        ys = [None if r[key] != r[key] else r[key] for r in rows]
        lo = [None if r.get(f"{key}_lo", r[key]) != r.get(f"{key}_lo", r[key]) else r.get(f"{key}_lo", r[key]) for r in rows]
        hi = [None if r.get(f"{key}_hi", r[key]) != r.get(f"{key}_hi", r[key]) else r.get(f"{key}_hi", r[key]) for r in rows]
        rgb = tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))
        if all(v is not None for v in lo + hi):
            fig.add_trace(go.Scatter(x=xs + xs[::-1], y=hi + lo[::-1], mode="lines", fill="toself", fillcolor=f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0.13)", line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", line=dict(color=color, width=2.5), name=name, connectgaps=False))
    if ref_line is not None:
        fig.add_hline(y=ref_line, line=dict(color="#888", dash="dash", width=1.5), annotation_text=ref_label, annotation_position="top left")
    fig.update_xaxes(title_text=param_label, type="category")
    fig.update_yaxes(title_text=y_label, type="log" if log_y else "linear")
    return _base(fig, 360, legend_y=-0.3)


def build_bottleneck_counts(rows):
    """Links: log10 der Zahl bottleneck-optimaler Bäume über n; rechts: Mehrkosten des teuersten und eines zufälligen gegen den MST."""
    ns = [str(r["n"]) for r in rows]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Zahl bottleneck-optimaler Bäume (log10)", "Mehrkosten gegen den MST (%)"), horizontal_spacing=0.12)
    fig.add_trace(go.Bar(x=ns, y=[r["log10_count"] for r in rows], marker_color=TREE_COLOR, name="log10 Zahl", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=ns, y=[r["gap_max"] for r in rows], mode="lines+markers", line=dict(color=BAD_COLOR, width=2.5), name="teuerster"), row=1, col=2)
    fig.add_trace(go.Scatter(x=ns, y=[r["gap_random"] for r in rows], mode="lines+markers", line=dict(color="#e8a13a", width=2.5), name="zufälliger"), row=1, col=2)
    fig.update_xaxes(title_text="Filialen n", type="category")
    return _base(fig, 330, legend_y=-0.3)
