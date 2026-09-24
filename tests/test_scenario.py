"""Instanzen: Erzeugung, Determinismus, Zusammenhang, Kandidatengraph, Geländefaktor, Runden, Ortschaften, Kreuz-Fixture."""

import numpy as np
import pytest

import cons_algorithm as A
import cons_constants as C
import cons_scenario as S
from cons_unionfind import UnionFind


def _connected(inst):
    uf = UnionFind(inst.n, "full")
    for u, v, _w in inst.edges:
        uf.union(u, v)
    return uf.components == 1


@pytest.mark.parametrize("kind", ["depot", "hubs"])
def test_generate_is_deterministic_and_well_formed(kind):
    a = S.generate(20, 6, 0.3, False, 7, kind, 5)
    b = S.generate(20, 6, 0.3, False, 7, kind, 5)
    assert a.edges == b.edges and np.array_equal(a.xy, b.xy)
    assert a.n == 21 and a.kind == kind and a.depot == 0
    assert tuple(a.xy[0]) == C.DEPOT_XY
    assert all(u < v for u, v, _w in a.edges) and list(a.edges) == sorted(a.edges, key=lambda e: (e[0], e[1]))
    assert len({(u, v) for u, v, _w in a.edges}) == a.m
    assert S.generate(20, 6, 0.3, False, 8, kind, 5).edges != a.edges


def test_candidate_graph_is_connected_even_with_tiny_k_and_complete_when_k_is_large():
    for k in (1, 2, 3):
        for seed in range(15):
            assert _connected(S.generate(25, k, 0.0, False, seed))
    inst = S.generate(12, 1000, 0.0, False, 1)
    assert inst.m == 13 * 12 // 2
    thin = S.generate(12, 3, 0.0, False, 1)
    assert thin.m < inst.m and {(u, v) for u, v, _w in thin.edges} <= {(u, v) for u, v, _w in inst.edges}


def test_terrain_factor_and_rounding():
    plain = S.generate(15, 1000, 0.0, False, 3)
    for u, v, w in plain.edges:
        assert w == pytest.approx(float(np.hypot(*(plain.xy[u] - plain.xy[v]))))
    hilly = S.generate(15, 1000, 0.5, False, 3)
    for (u, v, w), (_u, _v, w0) in zip(hilly.edges, plain.edges):
        assert w0 - 1e-9 <= w <= 1.5 * w0 + 1e-9
    rounded = S.generate(15, 1000, 0.0, True, 3)
    assert all(w == round(w) and w >= 1.0 for _u, _v, w in rounded.edges)
    assert len({w for _u, _v, w in rounded.edges}) < len({w for _u, _v, w in plain.edges})


def test_terrain_factor_does_not_depend_on_the_candidate_graph():
    dense = {(u, v): w for u, v, w in S.generate(14, 1000, 0.6, False, 9).edges}
    thin = {(u, v): w for u, v, w in S.generate(14, 4, 0.6, False, 9).edges}
    assert all(dense[e] == pytest.approx(w) for e, w in thin.items())


def test_hub_layout_has_rings_of_satellites():
    for sats in (3, 5, 8):
        inst = S.generate(sats * 2 + 2, 1000, 0.0, False, 4, "hubs", sats)
        pts = inst.xy[1:]
        assert inst.sats == sats and len(pts) == sats * 2 + 2
        hub = pts[0]
        ring = pts[1:sats + 1]
        d = np.hypot(ring[:, 0] - hub[0], ring[:, 1] - hub[1])
        assert d.min() >= 5.39 and d.max() <= 6.61
    assert S.generate(12, 6, 0.0, False, 4, "depot").sats == 0
    with pytest.raises(ValueError):
        S.generate(12, 6, 0.0, False, 4, "grid")


def test_hub_layout_truncates_to_the_requested_number_of_customers():
    for n in (5, 7, 11, 16):
        assert S.generate(n, 1000, 0.0, False, 2, "hubs", 5).n == n + 1


def test_cross_fixture():
    t = S.textbook_instance()
    assert t.kind == "textbook" and t.n == 5 and t.m == 10 and t.labels == ("W", "C", "E", "N", "S")
    assert np.allclose(t.xy[1], (40.0, 50.0)) and np.allclose(t.xy[0], (20.0, 50.0))
    costs = sorted(round(w, 4) for _u, _v, w in t.edges)
    assert costs == sorted([20.0] * 4 + [round(20 * 2 ** 0.5, 4)] * 4 + [40.0] * 2)
    assert A.kruskal(t.n, t.edges).cost == pytest.approx(80.0)
