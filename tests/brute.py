"""Referenzen nur für die Tests: alle Spannbäume aufzählen, Zufallsgraphen erzeugen."""

from itertools import combinations

import numpy as np

import cons_algorithm as A


def all_spanning_trees(n, edges):
    """Alle Spannbäume als Listen von Kantenindizes (nur für sehr kleine Graphen)."""
    m = len(edges)
    out = []
    for combo in combinations(range(m), n - 1):
        if A.is_spanning_tree(n, edges, list(combo)):
            out.append(list(combo))
    return out


def random_graph(n, extra, seed, integer=False, connected=True):
    """Zusammenhängender Zufallsgraph: zufälliger Pfad über alle Knoten plus `extra` weitere Kanten; Kosten zufällig (ganzzahlig 1..5 bei integer=True, dann viele Gleichstände)."""
    rng = np.random.default_rng([seed, 4711])
    pairs = set()
    perm = [int(x) for x in rng.permutation(n)]
    for a, b in zip(perm, perm[1:]):
        pairs.add((min(a, b), max(a, b)))
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    for j in rng.permutation(len(all_pairs))[: extra]:
        pairs.add(all_pairs[int(j)])
    edges = []
    for u, v in sorted(pairs):
        w = float(rng.integers(1, 6)) if integer else float(np.round(rng.uniform(1, 10), 3))
        edges.append((u, v, w))
    return tuple(edges)


def brute_best(n, edges, ok):
    """Kleinste Kosten unter allen Spannbäumen, die `ok(tree)` erfüllen (None, wenn keiner)."""
    best = None
    for t in all_spanning_trees(n, edges):
        if ok(t):
            c = A.tree_cost(edges, t)
            if best is None or c < best - 1e-9:
                best = c
    return best
