"""Unabhängiges Orakel: Gradgrenze und Hop-Grenze gegen ein CP-SAT-Modell (Arboreszenz mit Tiefenmarken, OR-Tools) auf Zufallsgraphen mit 6 bis 12 Knoten und gegen die Aufzählung aller Spannbäume
(networkx) auf winzigen Graphen; Lagrange-Untergrenze <= Optimum, Greedy nie unter dem Optimum, "Greedy scheitert" nur dort, wo wirklich ein Baum existiert; Bottleneck-Aussagen
(b*, billigster/teuerster bottleneck-optimaler Baum, Zahl der Bäume) gegen die Aufzählung."""

import math
import random

import pytest

import cons_algorithm as A

SC = 1_000_000


def _edges(rng, n, extra, wmax, integer):
    perm = list(range(n))
    rng.shuffle(perm)
    es = {(min(a, b), max(a, b)) for a, b in zip(perm, perm[1:])}
    pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    es |= set(rng.sample(pairs, min(extra, len(pairs))))
    return tuple((u, v, float(rng.randint(1, wmax)) if integer else round(rng.uniform(1, 10), 3)) for u, v in sorted(es))


def _cpsat(n, edges, kind, limit, root=0):
    cp = pytest.importorskip("ortools.sat.python.cp_model")
    m = cp.CpModel()
    arcs = [(u, v, w) for u, v, w in edges] + [(v, u, w) for u, v, w in edges]
    x = [m.NewBoolVar(f"x{a}") for a in range(len(arcs))]
    d = [m.NewIntVar(0, n - 1 if kind == "degree" else min(limit, n - 1), f"d{v}") for v in range(n)]
    m.Add(d[root] == 0)
    for v in range(n):
        m.Add(sum(x[a] for a, (_p, c, _w) in enumerate(arcs) if c == v) == (0 if v == root else 1))
    for a, (p, c, _w) in enumerate(arcs):
        m.Add(d[c] >= d[p] + 1).OnlyEnforceIf(x[a])
    if kind == "degree":
        for v in range(n):
            m.Add(sum(x[a] for a, (p, c, _w) in enumerate(arcs) if c == v or p == v) <= limit)
    m.Minimize(sum(int(round(arcs[a][2] * SC)) * x[a] for a in range(len(arcs))))
    s = cp.CpSolver()
    s.parameters.max_time_in_seconds = 20
    s.parameters.num_workers = 1
    st = s.Solve(m)
    if st == cp.INFEASIBLE:
        return False, math.inf
    assert st == cp.OPTIMAL
    return True, s.ObjectiveValue() / SC


def _enum_trees(n, edges):
    nx = pytest.importorskip("networkx")
    g = nx.Graph()
    g.add_nodes_from(range(n))
    for i, (u, v, w) in enumerate(edges):
        g.add_edge(u, v, weight=w, idx=i)
    for t in nx.SpanningTreeIterator(g, minimum=True, weight="weight"):
        yield sorted(d["idx"] for _u, _v, d in t.edges(data=True))


def _check(n, edges, kind, limit, ref):
    has, opt = ref
    if kind == "degree":
        g = A.degree_greedy(n, edges, limit)
        lg = A.lagrange_degree(n, edges, limit, g.tree if g.feasible else None)
        if has:
            assert lg.lb <= opt + 1e-6
        trees = [g] + ([lg] if lg.tree else [])
        ex = A.exact_tree(n, edges, "degree", limit, pi=lg.pi, ub_tree=lg.tree or (g.tree if g.feasible else None))
        if lg.tree:
            assert A.is_spanning_tree(n, edges, lg.tree) and max(A.degrees(n, edges, lg.tree)) <= limit and has and lg.ub >= opt - 1e-6
        if g.feasible:
            assert has and A.is_spanning_tree(n, edges, g.tree) and max(A.degrees(n, edges, g.tree)) <= limit and g.cost >= opt - 1e-6
        del trees
    else:
        g = A.hop_greedy(n, edges, limit)
        if g.feasible:
            assert has and A.is_spanning_tree(n, edges, g.tree) and max(A.depths_from(n, edges, g.tree)) <= limit and g.cost >= opt - 1e-6
        else:
            assert not has                                          # der Schichtenbaum findet einen Baum, sobald überhaupt einer existiert
        ex = A.exact_tree(n, edges, "hops", limit, ub_tree=g.tree if g.feasible else None)
    if ex.proved:
        assert ex.feasible == has
        if has:
            assert ex.cost == pytest.approx(opt, abs=1e-5)


def test_degree_and_hop_limits_against_cp_sat_on_random_graphs():
    rng = random.Random(21)
    for _ in range(25):
        n = rng.randint(6, 11)
        edges = _edges(rng, n, rng.randint(n // 2, 2 * n), rng.choice([3, 6, 10]), rng.random() < 0.5)
        for kind, limit in (("degree", rng.choice([2, 3])), ("hops", rng.choice([2, 3, 4]))):
            _check(n, edges, kind, limit, _cpsat(n, edges, kind, limit))


def test_limits_and_bottleneck_facts_against_spanning_tree_enumeration():
    rng = random.Random(22)
    for _ in range(40):
        n = rng.randint(2, 6)
        edges = _edges(rng, n, rng.randint(0, 6), rng.choice([2, 5, 9]), rng.random() < 0.6)
        trees = list(_enum_trees(n, edges))
        for kind, limit in (("degree", rng.choice([1, 2, 3])), ("hops", rng.choice([1, 2, 5]))):
            ok = [t for t in trees if (max(A.degrees(n, edges, t)) <= limit if kind == "degree" else max(A.depths_from(n, edges, t)) <= limit)]
            ref = (bool(ok), min((sum(edges[i][2] for i in t) for t in ok), default=math.inf))
            _check(n, edges, kind, limit, ref)
        bott = [max((edges[i][2] for i in t), default=0.0) for t in trees]
        bstar = min(bott)
        assert A.bottleneck_threshold(n, edges) == pytest.approx(bstar)
        best = [t for t, b in zip(trees, bott) if abs(b - bstar) < 1e-9]
        costs = [sum(edges[i][2] for i in t) for t in best]
        out = A.bottleneck_optimal_trees(n, edges, seed=3)
        assert sum(edges[i][2] for i in out["min"]) == pytest.approx(min(costs)) and sum(edges[i][2] for i in out["max"]) == pytest.approx(max(costs))
        assert A.count_spanning_trees(n, edges, A.threshold_edges(edges, bstar))[0] == pytest.approx(len(best))
