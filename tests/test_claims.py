"""Jede Zahl, die App-Text, Preset-Hilfen und README nennen, wird hier über die echten Auswertungsfunktionen (ev.analyse / ev.run_config / ev.sweep / ev.feasibility / ev.bottleneck_counts /
ev.degree_profile) belegt - nie über ein Ad-hoc-Skript. Kosten sind Gleitkommazahlen, Kennzahlen daher auf Rundungsstellen verglichen; Zähler, Grade und Tiefen sind ganzzahlig."""

from dataclasses import replace
from functools import lru_cache

import pytest

import cons_algorithm as A
import cons_constants as C
import cons_evaluation as ev

BASE = ev.Settings(seed=0)


@lru_cache(maxsize=None)
def _ana(**kw):
    return ev.analyse(ev.Settings(**kw))


@lru_cache(maxsize=None)
def _cfg(**kw):
    return ev.run_config(replace(BASE, **kw))


@lru_cache(maxsize=None)
def _sweep(param, **kw):
    return ev.sweep(param, replace(BASE, **kw))


@lru_cache(maxsize=None)
def _feas(**kw):
    return ev.feasibility(replace(BASE, **kw))


def _col(rows, key, digits=2):
    return [None if r[key] != r[key] else round(r[key], digits) for r in rows]


def r2(x):
    return round(x, 2)


# --- Preset-Hilfen (jeweils die Einzelinstanz des Presets) ------------------------------------------------------------------------------------------


def test_preset_standard_case_degree_two():
    a = _ana()
    assert (r2(a.mst_cost), a.mst_max_degree, a.mst_depth) == (215.65, 3, 9)
    assert [r2(a.trees[k].cost) for k in ("greedy", "lagrange", "exact")] == [224.37, 224.37, 224.37] and r2(a.gap("exact")) == 4.04
    assert a.lagrange.iterations == 3 and a.lagrange.optimal and r2(a.lagrange.lb) == 224.37
    assert a.exact.nodes == 1 and a.proved


def test_preset_greedy_fails():
    a = _ana(seed=100001)
    assert not a.trees["greedy"].feasible and not a.trees["greedy"].tree
    assert a.trees["lagrange"].feasible and r2(a.trees["lagrange"].cost) == 267.66 and r2(a.trees["exact"].cost) == 267.66 and r2(a.mst_cost) == 245.43 and r2(a.gap("exact")) == 9.06
    g = A.degree_greedy(a.inst.n, a.inst.edges, 2)
    assert not g.detail["kruskal_ok"] and not g.detail["prim_ok"]


def test_preset_villages_degree_three():
    a = _ana(kind="hubs", n=20, delta=3)
    assert r2(a.mst_cost) == 189.43 and a.mst_max_degree == 5
    assert r2(a.gap("exact")) == 1.07 and r2(a.gap("lagrange")) == 1.07 and r2(a.gap("greedy")) == 1.89 and r2(a.lb_gap) == 1.07 and a.proved


def test_preset_free_limit():
    a = _ana(n=20, delta=3)
    assert a.mst_max_degree == 3 and not a.mst_violates and r2(a.gap("exact")) == 0.0 and r2(a.gap("greedy")) == 0.0


def test_preset_hop_limit_three():
    a = _ana(constraint="hops", hops=3)
    assert a.mst_depth == 9 and r2(a.gap("exact")) == 18.42 and r2(a.gap("greedy")) == 33.91 and a.proved and a.exact.nodes == 7809
    assert round(ev.detour(a.inst, a.trees["mst"].tree)[1], 2) == 2.74 and round(ev.detour(a.inst, a.trees["exact"].tree)[1], 2) == 1.90
    assert r2(a.b_star) == 36.93 and r2(A.bottleneck(a.inst.edges, a.trees["exact"].tree)) == 42.43


def test_preset_star():
    a = _ana(constraint="hops", hops=1, k=1000)
    assert r2(a.trees["exact"].cost) == 555.17 and r2(a.mst_cost) == 215.65 and r2(a.gap("exact")) == 157.44 and a.proved
    assert round(ev.detour(a.inst, a.trees["exact"].tree)[1], 2) == 1.0 and max(A.degrees(a.inst.n, a.inst.edges, a.trees["exact"].tree)) == 12


def test_preset_bottleneck_billions():
    a = _ana(constraint="bottleneck", n=20)
    assert round(10 ** a.count_log10, -8) == 4.6e9 and r2(a.b_star) == 33.30 and r2(a.mst_cost) == 316.00
    assert r2(a.gap("random")) == 49.08 and r2(a.gap("max")) == 74.57 and a.threshold_edges == 50


def test_preset_cross_fixture():
    for con, kw, cost in (("degree", dict(delta=2), 96.57), ("degree", dict(delta=3), 88.28), ("hops", dict(hops=1), 116.57)):
        a = _ana(kind="textbook", constraint=con, **kw)
        assert r2(a.trees["exact"].cost) == cost and a.proved
    a = _ana(kind="textbook", delta=2)
    assert r2(a.mst_cost) == 80.0 and a.mst_max_degree == 4 and r2(a.gap("exact")) == 20.71
    assert r2(_ana(kind="textbook", delta=3).gap("exact")) == 10.36 and r2(_ana(kind="textbook", constraint="hops", hops=1).gap("exact")) == 45.71


# --- Grad -------------------------------------------------------------------------------------------------------------------------------------------


def test_price_of_the_degree_limit_on_uniform_instances():
    rows = _sweep("delta", n=20)
    assert [r["value"] for r in rows] == [2, 3, 4, 5, 6]
    assert _col(rows, "gap_best") == [7.29, 0.0, 0.0, 0.0, 0.0] and _col(rows, "gap_greedy") == [13.7, 0.0, 0.0, 0.0, 0.0] and _col(rows, "gap_lb") == [6.82, 0.0, 0.0, 0.0, 0.0]
    assert [r["viol_mst"] for r in rows] == [100.0, 0.0, 0.0, 0.0, 0.0] and [r["mst_maxdeg"] for r in rows] == [3.0] * 5


def test_price_of_the_degree_limit_in_villages():
    rows = _sweep("delta", kind="hubs", n=20)
    assert _col(rows, "gap_best") == [4.58, 1.06, 0.23, 0.0, 0.0] and _col(rows, "gap_greedy") == [9.42, 1.51, 0.49, 0.0, 0.0] and _col(rows, "gap_lb") == [4.58, 1.06, 0.23, 0.0, 0.0]
    assert [r["viol_mst"] for r in rows] == [100.0, 100.0, 100.0, 0.0, 0.0] and rows[0]["mst_maxdeg"] == 5.0


def test_greedy_fails_often_at_degree_two_but_lagrange_never():
    for kw, expect in ((dict(n=12), 20.0), (dict(n=20), 20.0), (dict(kind="hubs", n=20), 60.0)):
        r = _cfg(delta=2, **kw)
        assert r["fail_greedy"] == expect and r["fail_lagrange"] == 0.0 and r["feasible_share"] == 100.0
    assert _cfg(kind="hubs", n=20, delta=3)["fail_greedy"] == 0.0


def test_feasibility_experiment_degree():
    f = _feas(n=20, delta=2)
    assert (f["greedy_fail"], f["lagrange_rescue"], f["lagrange_fail"], f["found_share"], f["mst_violates"]) == (34.0, 34.0, 0.0, 100.0, 100.0)
    assert _feas(n=12, delta=2)["greedy_fail"] == 6.0 and _feas(kind="hubs", n=20, delta=2)["greedy_fail"] == 50.0 and _feas(kind="hubs", n=20, delta=3)["greedy_fail"] == 0.0


def test_greedy_versus_exact_gap_for_degree_two():
    r = _cfg(n=20, delta=2)
    assert r2(r["gap_greedy"]) == 13.7 and r2(r["gap_exact"]) == 7.29 and r2(r["excess_greedy"]) == 4.67 and r2(r["gap_lb"]) == 6.82 and r["proved_share"] == 100.0
    assert r2(r["bott_gap"]) == 7.55


def test_degree_price_over_n_is_not_monotone():
    rows = _sweep("n", delta=2)
    assert [r["value"] for r in rows] == [8, 12, 16, 20, 30]
    assert _col(rows, "gap_best") == [1.65, 0.66, 6.36, 7.29, 9.09] and _col(rows, "gap_greedy") == [2.84, 0.33, 16.26, 13.7, 10.84]
    assert [r["offered_share"] for r in rows] == [100.0] * 4 + [0.0] and rows[4]["fail_greedy"] == 60.0 and rows[4]["fail_lagrange"] == 0.0
    assert _col(rows, "gap_lb") == [1.65, 0.66, 6.36, 6.82, 7.85] and [r["proved_share"] for r in rows] == [100.0] * 4 + [0.0]


def test_lagrange_can_come_back_empty_and_that_is_reported_as_unknown_not_infeasible():
    a = _ana(kind="hubs", n=30, sats=6, seed=100002, delta=2)
    assert not a.trees["greedy"].feasible and not a.trees["lagrange"].feasible and a.best_name is None and not a.infeasible and not a.exact_offered
    assert r2(a.lagrange.lb) == 286.36 and a.lagrange.iterations == 150 and not a.lagrange.tree


def test_thin_candidate_graphs_make_the_greedy_fail_more_often():
    rows = _sweep("k", delta=2)
    assert [r["fail_greedy"] for r in rows] == [40.0, 20.0, 20.0, 0.0, 0.0] and _col(rows, "gap_best") == [0.66] * 5


def test_village_size_changes_the_price():
    rows = _sweep("sats", kind="hubs", n=20, delta=3)
    assert [r["value"] for r in rows] == [3, 4, 5, 6, 8]
    assert [abs(x) for x in _col(rows, "gap_best")] == [0.0, 3.44, 1.06, 0.0, 0.0] and _col(rows, "mst_maxdeg", 0) == [3.0, 4.0, 5.0, 3.0, 3.0]


def test_euclidean_degree_profile():
    assert ev.degree_profile("depot", 30, 0.0) == {3: 250, 4: 50}
    assert ev.degree_profile("depot", 30, 1.0) == {3: 111, 4: 169, 5: 19, 6: 1}
    assert ev.degree_profile("hubs", 30, 0.0, 5) == {4: 3, 5: 297}


# --- Hops -------------------------------------------------------------------------------------------------------------------------------------------


def test_price_of_the_hop_limit_with_k_six():
    rows = _sweep("hops", n=12, constraint="hops")
    assert [r["value"] for r in rows] == [1, 2, 3, 4, 5, 6, 8, 10]
    assert [r["infeasible_share"] for r in rows] == [100.0, 0, 0, 0, 0, 0, 0, 0]
    assert _col(rows, "gap_best")[1:] == [40.62, 15.06, 6.49, 4.45, 1.4, 0.0, 0.0] and _col(rows, "gap_greedy")[1:] == [42.31, 18.1, 9.4, 5.09, 1.4, 0.0, 0.0]
    assert [r["fail_prim"] for r in rows] == [0.0, 40.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert [r["proved_share"] for r in rows] == [0.0, 0.0, 80.0, 100.0, 100.0, 100.0, 100.0, 100.0]


def test_price_and_detour_of_the_hop_limit_with_a_complete_candidate_graph():
    rows = _sweep("hops", n=12, constraint="hops", k=1000)
    assert _col(rows, "gap_best") == [133.6, 32.8, 15.06, 6.49, 4.45, 1.4, 0.0, 0.0]
    assert _col(rows, "detour_best") == [1.0, 2.2, 1.59, 1.9, 1.9, 1.9, 2.25, 2.25] and _col(rows, "detour_mst") == [2.25] * 8
    assert _col(rows, "depth_best", 0) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0] and _col(rows, "mst_depth", 0) == [8.0] * 8
    assert _col(rows, "bott_gap") == [119.69, 77.86, 21.91, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert [r["viol_mst"] for r in rows] == [100.0, 100.0, 100.0, 100.0, 80.0, 80.0, 40.0, 0.0]


def test_greedy_versus_exact_gap_for_hops():
    r = _cfg(n=12, constraint="hops", hops=4)
    assert r2(r["gap_greedy"]) == 9.4 and r2(r["gap_exact"]) == 6.49 and r["proved_share"] == 100.0 and r2(r["excess_greedy"]) == 2.73


def test_prim_with_a_depth_limit_gets_stuck_on_larger_instances():
    r = _cfg(n=20, constraint="hops", hops=4)
    assert r["fail_prim"] == 80.0 and r["fail_greedy"] == 0.0 and r["feasible_share"] == 100.0 and r["proved_share"] == 0.0 and r["offered_share"] == 0.0
    f = _feas(n=20, constraint="hops", hops=4)
    assert (f["prim_fail"], f["layer_fail"], f["found_share"]) == (60.0, 0.0, 100.0)
    f3 = _feas(n=20, constraint="hops", hops=3)
    assert (f3["prim_fail"], f3["layer_fail"], f3["infeasible_share"], f3["found_share"]) == (82.0, 0.0, 8.0, 92.0)
    assert _feas(n=12, constraint="hops", hops=4)["prim_fail"] == 4.0


def test_hop_price_grows_with_n_at_fixed_limit():
    rows = _sweep("n", constraint="hops", hops=4)
    assert [r["value"] for r in rows] == [8, 12, 16, 20, 30] and _col(rows, "gap_best") == [3.65, 6.49, 11.02, 13.54, 22.9]
    assert [r["infeasible_share"] for r in rows] == [0.0, 0.0, 0.0, 0.0, 40.0] and _col(rows, "mst_depth", 0) == [5.0, 8.0, 8.0, 10.0, 14.0]


def test_hop_limit_one_needs_a_complete_candidate_graph():
    assert _cfg(n=12, constraint="hops", hops=1)["infeasible_share"] == 100.0 and _cfg(n=12, constraint="hops", hops=1, k=1000)["feasible_share"] == 100.0


# --- Bottleneck -------------------------------------------------------------------------------------------------------------------------------------


def test_bottleneck_optimal_trees_are_astronomically_many_and_cost_more():
    rows = ev.bottleneck_counts(BASE)
    assert [r["n"] for r in rows] == [10, 20, 30, 60]
    assert _col(rows, "log10_count", 1) == [3.2, 10.6, 14.2, 37.3] and _col(rows, "gap_max", 1) == [45.6, 74.2, 64.4, 100.8] and _col(rows, "gap_random", 1) == [21.1, 41.3, 30.2, 52.2]
    assert _col(rows, "threshold_share", 0) == [46.0, 67.0, 65.0, 83.0]


def test_bottleneck_sweep_over_n():
    rows = _sweep("n", constraint="bottleneck")
    assert [r["value"] for r in rows] == [8, 12, 16, 20, 30]
    assert _col(rows, "log10_count", 1) == [2.2, 4.5, 6.0, 10.6, 14.2] and _col(rows, "gap_max", 1) == [61.7, 53.4, 55.2, 74.2, 64.4] and _col(rows, "gap_random", 1) == [17.0, 26.9, 17.3, 41.3, 30.2]


def test_rounding_and_terrain_leave_the_bottleneck_story_intact():
    r = _cfg(constraint="bottleneck", n=20, round_costs=True)
    assert round(r["log10_count"], 1) == 10.6 and round(r["gap_max"], 1) == 74.5 and round(r["gap_random"], 1) == 41.6
    t = _cfg(constraint="bottleneck", n=20, terrain=0.6)
    assert round(t["log10_count"], 1) == 8.0 and round(t["gap_max"], 1) == 62.0 and round(t["gap_random"], 1) == 31.9


def test_mst_is_bottleneck_optimal_on_every_instance_of_the_sweeps():
    for seed in C.SWEEP_SEEDS:
        for kind in ("depot", "hubs"):
            a = ev.analyse(ev.Settings(kind=kind, n=20, seed=seed, constraint="bottleneck"))
            assert A.bottleneck(a.inst.edges, a.trees["mst"].tree) == pytest.approx(a.b_star)
