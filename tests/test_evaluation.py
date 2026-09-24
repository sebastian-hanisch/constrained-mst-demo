"""Auswertung: Analysis-Felder, Kennzahlen über feste Instanzen, Sweeps, Machbarkeits-Experiment, Bottleneck-Zählung, Determinismus."""

import math
from dataclasses import replace

import pytest

import cons_algorithm as A
import cons_constants as C
import cons_evaluation as ev

BASE = ev.Settings(seed=0)


def test_analyse_degree_fields_are_consistent():
    a = ev.analyse(ev.Settings())
    assert a.constraint == "degree" and a.limit == 2 and a.exact_offered and a.proved and not a.infeasible
    assert set(a.trees) == {"mst", "greedy", "lagrange", "exact"}
    for name in ("greedy", "lagrange", "exact"):
        f = a.trees[name]
        assert f.feasible and A.is_spanning_tree(a.inst.n, a.inst.edges, f.tree) and max(A.degrees(a.inst.n, a.inst.edges, f.tree)) <= 2
        assert f.cost == pytest.approx(A.tree_cost(a.inst.edges, f.tree))
    assert a.mst_violates and a.best_name in ("greedy", "lagrange", "exact")
    assert a.lower_bound <= a.trees["exact"].cost + 1e-9 and a.lower_bound >= a.mst_cost - 1e-9 and a.lb_gap >= 0
    assert a.gap("mst") == 0.0 and a.gap("nope") is None


def test_analyse_hops_fields_are_consistent():
    a = ev.analyse(ev.Settings(constraint="hops", hops=3))
    assert a.limit == 3 and a.min_hops == 2 and a.hop_detail["layer_ok"] and a.mst_violates
    assert set(a.trees) == {"mst", "greedy", "exact"} and a.lagrange is None and a.lower_bound == a.mst_cost
    assert max(A.depths_from(a.inst.n, a.inst.edges, a.trees["exact"].tree)) <= 3
    assert a.trees["exact"].cost <= a.trees["greedy"].cost + 1e-9


def test_analyse_bottleneck_fields_are_consistent():
    a = ev.analyse(ev.Settings(constraint="bottleneck", n=20))
    assert set(a.trees) == {"mst", "min", "max", "random"} and not a.mst_violates and not a.infeasible
    for name in ("min", "max", "random"):
        assert A.bottleneck(a.inst.edges, a.trees[name].tree) == pytest.approx(a.b_star)
    assert a.gap("min") == pytest.approx(0.0, abs=1e-9) and a.gap("max") > a.gap("random") > 0
    assert a.threshold_edges > 0 and a.count_log10 > 1
    assert a.best_name is None


def test_infeasible_hop_limits_are_detected_and_never_claim_a_tree():
    a = ev.analyse(ev.Settings(constraint="hops", hops=1, k=6))
    assert a.min_hops > 1 and a.infeasible and a.best_name is None and not a.trees["greedy"].feasible
    assert a.trees["greedy"].cost == math.inf and a.gap("greedy") is None
    assert ev.run_config(replace(BASE, constraint="hops", hops=1))["infeasible_share"] == 100.0


def test_exact_is_only_offered_for_small_instances():
    small = ev.analyse(ev.Settings(n=C.N_EXACT["degree"]))
    big = ev.analyse(ev.Settings(n=C.N_EXACT["degree"] + 1))
    assert small.exact_offered and not big.exact_offered and big.exact is None and "exact" not in big.trees
    assert ev.analyse(ev.Settings(constraint="hops", n=12)).exact_offered and not ev.analyse(ev.Settings(constraint="hops", n=13)).exact_offered
    assert not ev.analyse(ev.Settings(constraint="bottleneck")).exact_offered


def test_analyse_is_deterministic():
    for con in ev.CONSTRAINTS:
        s = ev.Settings(constraint=con, n=15)
        a, b = ev.analyse(s), ev.analyse(s)
        assert {k: (f.cost, f.tree) for k, f in a.trees.items()} == {k: (f.cost, f.tree) for k, f in b.trees.items()}


def test_run_config_is_deterministic_and_uses_five_fixed_instances():
    for con in ev.CONSTRAINTS:
        s = replace(BASE, constraint=con)
        a, b = ev.run_config(s), ev.run_config(s)
        assert a == b or all((x == y) or (x != x and y != y) for x, y in zip(a.values(), b.values()))
        assert a["n_runs"] == 5
    assert C.SWEEP_SEEDS == tuple(range(100000, 100005))


def test_run_config_ignores_the_seed_of_the_settings():
    a = ev.run_config(replace(BASE, seed=1))
    b = ev.run_config(replace(BASE, seed=999))
    assert a["gap_best"] == b["gap_best"] and a["cost"] == b["cost"]


def test_run_config_percentile_bands_enclose_the_median():
    r = ev.run_config(replace(BASE, n=20))
    for key in ("gap_best", "gap_greedy", "gap_lb", "mst_depth"):
        assert r[f"{key}_lo"] - 1e-9 <= r[key] <= r[f"{key}_hi"] + 1e-9


def test_lower_bound_never_exceeds_the_best_gap():
    for delta in (2, 3):
        r = ev.run_config(replace(BASE, n=20, delta=delta))
        assert r["gap_lb"] <= r["gap_best"] + 1e-9
    r = ev.run_config(replace(BASE, n=20, delta=2))
    assert r["gap_best"] <= r["gap_greedy"] + 1e-9 and r["excess_lagrange"] == pytest.approx(0.0, abs=1e-9)


def test_sweeps_have_one_row_per_value_and_only_valid_parameters():
    for con in ev.CONSTRAINTS:
        for kind in ("depot", "hubs"):
            params = ev.sweep_params(con, kind)
            assert params[0] == {"degree": "delta", "hops": "hops", "bottleneck": "n"}[con] and ("sats" in params) == (kind == "hubs")
            assert len(set(params)) == len(params) and all(p in ev.SWEEP_LABELS for p in params)
    rows = ev.sweep("delta", replace(BASE, n=10))
    assert [r["value"] for r in rows] == list(C.DELTA_SWEEP)
    rows = ev.sweep("n", replace(BASE, n=10, constraint="bottleneck"), values=(6, 9))
    assert [r["value"] for r in rows] == [6, 9]


def test_feasibility_experiment_reports_consistent_shares():
    base = replace(BASE, n=12, delta=2)
    f = ev.feasibility(base, seeds=range(200000, 200010))
    assert f["n_runs"] == 10 and 0 <= f["greedy_fail"] <= 100 and f["lagrange_rescue"] <= f["greedy_fail"] + 1e-9
    assert f["found_share"] + f["infeasible_share"] + f["unknown_share"] == pytest.approx(100.0)
    h = ev.feasibility(replace(base, constraint="hops", hops=3, n=15), seeds=range(200000, 200008))
    assert h["layer_fail"] == 0.0 and h["found_share"] + h["infeasible_share"] == pytest.approx(100.0)


def test_bottleneck_counts_grow_with_n():
    rows = ev.bottleneck_counts(BASE, ns=(8, 14, 20))
    assert [r["n"] for r in rows] == [8, 14, 20]
    logs = [r["log10_count"] for r in rows]
    assert logs == sorted(logs) and logs[0] > 0


def test_detour_is_one_for_a_star_and_larger_for_a_deep_tree():
    inst = ev.instance_of(ev.Settings(k=1000))
    star = A.star_tree(inst.n, inst.edges)
    med, mx = ev.detour(inst, star)
    assert med == pytest.approx(1.0) and mx == pytest.approx(1.0)
    _m, mx_mst = ev.detour(inst, A.kruskal(inst.n, inst.edges).tree)
    assert mx_mst > 1.5


def test_degree_profile_counts_add_up():
    p = ev.degree_profile("depot", 12, 0.0, seeds=range(20))
    assert sum(p.values()) == 20 and list(p) == sorted(p)
