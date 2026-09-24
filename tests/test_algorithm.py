"""Die Korrektheitskette der Verfahren (zuerst, vor jeder Messung): Bottleneck-Satz, Gültigkeit, Exaktheit gegen Brute-Force, Sonderfälle, Euklid-Satz, Heuristiken und Schranken, schwierige Fixtures."""

import math
from itertools import permutations

import numpy as np
import pytest

import cons_algorithm as A
import cons_evaluation as ev
import cons_scenario as S
from brute import all_spanning_trees, brute_best, random_graph

# per Skriptsuche gefundene kleine Fixtures (fest verdrahtet): (n, Kanten)
DEG_FAIL = (7, [(0, 1, 4.594), (0, 2, 5.708), (1, 2, 9.858), (1, 5, 4.833), (2, 4, 7.531), (2, 6, 6.522), (3, 5, 7.454), (3, 6, 5.169), (5, 6, 8.135)])
DEG_WORSE = (7, [(0, 1, 5.452), (0, 3, 9.897), (0, 4, 1.889), (0, 6, 4.762), (1, 2, 6.41), (1, 3, 4.931), (1, 4, 5.966), (1, 5, 2.206), (1, 6, 8.663), (2, 4, 1.428), (2, 5, 6.602), (3, 5, 8.983),
                 (3, 6, 9.268), (4, 5, 1.612), (5, 6, 9.332)])
HOP_FAIL = (6, [(0, 1, 8.913), (0, 2, 1.446), (1, 2, 6.735), (1, 3, 7.072), (1, 4, 9.297), (1, 5, 2.569), (2, 3, 6.158), (2, 4, 1.033), (3, 4, 4.414), (3, 5, 6.498), (4, 5, 2.004)])
HOP_WORSE = (7, [(0, 1, 9.168), (0, 2, 6.062), (0, 3, 6.689), (0, 4, 8.783), (0, 5, 6.827), (1, 2, 4.302), (1, 3, 6.849), (1, 5, 1.534), (1, 6, 2.538), (2, 3, 1.651), (2, 5, 4.077), (2, 6, 2.966),
                (3, 5, 7.652), (4, 6, 9.764)])


def _graphs(count=120, integer_every=3):
    for seed in range(count):
        n = 4 + seed % 4
        yield n, random_graph(n, 2 + seed % 7, seed, integer=(seed % integer_every == 0))


def _deg_ok(n, edges, delta):
    return lambda t: max(A.degrees(n, edges, t)) <= delta


def _hop_ok(n, edges, hops):
    return lambda t: max(A.depths_from(n, edges, t)) <= hops


# --- 1. Bottleneck-Satz -----------------------------------------------------------------------------------------------------------------------------


def test_mst_minimizes_the_longest_edge_over_all_spanning_trees():
    for n, edges in _graphs(200):
        trees = all_spanning_trees(n, edges)
        b_min = min(A.bottleneck(edges, t) for t in trees)
        mst = A.kruskal(n, edges)
        assert A.bottleneck(edges, mst.tree) == pytest.approx(b_min)
        assert A.bottleneck_threshold(n, edges) == pytest.approx(b_min)


def test_bottleneck_optimal_trees_are_exactly_the_spanning_trees_of_the_threshold_graph():
    for n, edges in _graphs(150):
        trees = all_spanning_trees(n, edges)
        b = A.bottleneck_threshold(n, edges)
        opt = {frozenset(t) for t in trees if A.bottleneck(edges, t) <= b + 1e-12}
        sub = A.threshold_edges(edges, b)
        thr = {frozenset(t) for t in all_spanning_trees(n, [edges[i] for i in sub])}
        assert {frozenset(sub[i] for i in t) for t in map(list, thr)} == opt
        assert round(A.count_spanning_trees(n, edges, sub)[0]) == len(opt)


def test_cheapest_and_dearest_bottleneck_optimal_trees_match_brute_force():
    for n, edges in _graphs(120):
        b = A.bottleneck_threshold(n, edges)
        costs = [A.tree_cost(edges, t) for t in all_spanning_trees(n, edges) if A.bottleneck(edges, t) <= b + 1e-12]
        trees = A.bottleneck_optimal_trees(n, edges, seed=3)
        assert set(trees) == {"min", "max", "random"}
        assert A.tree_cost(edges, trees["min"]) == pytest.approx(min(costs)) and A.tree_cost(edges, trees["max"]) == pytest.approx(max(costs))
        for t in trees.values():
            assert A.is_spanning_tree(n, edges, t) and A.bottleneck(edges, t) == pytest.approx(b)
        assert min(costs) - 1e-9 <= A.tree_cost(edges, trees["random"]) <= max(costs) + 1e-9
        assert sorted(trees["min"]) == sorted(A.kruskal(n, edges).tree)


def test_mst_path_is_a_minimax_path():
    for n, edges in list(_graphs(60))[:40]:
        if n > 6:
            continue
        mst = A.kruskal(n, edges).tree
        adj = [[] for _ in range(n)]
        for u, v, w in edges:
            adj[u].append((v, w))
            adj[v].append((u, w))
        for a in range(n):
            for b in range(a + 1, n):
                best = math.inf
                others = [x for x in range(n) if x not in (a, b)]
                for r in range(len(others) + 1):
                    for mid in permutations(others, r):
                        path = [a, *mid, b]
                        ws = []
                        for x, y in zip(path, path[1:]):
                            w = [w for v, w in adj[x] if v == y]
                            if not w:
                                break
                            ws.append(w[0])
                        else:
                            best = min(best, max(ws))
                assert A.minimax_path(n, edges, mst, a, b) == pytest.approx(best)


def test_cayley_count_of_spanning_trees_of_the_complete_graph():
    for n in range(2, 9):
        edges = [(u, v, 1.0) for u in range(n) for v in range(u + 1, n)]
        assert round(A.count_spanning_trees(n, edges)[0]) == n ** (n - 2)
    assert A.count_spanning_trees(1, [])[0] == 1.0
    assert A.count_spanning_trees(3, [(0, 1, 1.0)])[0] == 0.0


# --- 2. Gültigkeit ----------------------------------------------------------------------------------------------------------------------------------


def test_every_output_is_a_spanning_tree_that_meets_its_limit():
    for n, edges in _graphs(150):
        for delta in (2, 3):
            g = A.degree_greedy(n, edges, delta)
            lg = A.lagrange_degree(n, edges, delta, g.tree if g.feasible else None)
            ex = A.exact_tree(n, edges, "degree", delta, pi=lg.pi, ub_tree=lg.tree or None)
            for t in ([g.tree] if g.feasible else []) + ([lg.tree] if lg.tree else []) + ([ex.tree] if ex.feasible else []):
                assert A.is_spanning_tree(n, edges, t) and max(A.degrees(n, edges, t)) <= delta
            if not g.feasible:
                assert g.tree == [] and g.cost == math.inf
        for hops in (1, 2, 3):
            g = A.hop_greedy(n, edges, hops)
            ex = A.exact_tree(n, edges, "hops", hops, ub_tree=g.tree if g.feasible else None)
            for t in ([g.tree] if g.feasible else []) + ([ex.tree] if ex.feasible else []):
                assert A.is_spanning_tree(n, edges, t) and max(A.depths_from(n, edges, t)) <= hops
                assert min(A.depths_from(n, edges, t)) >= 0


def test_failed_capped_runs_never_claim_a_tree():
    n, edges = DEG_FAIL
    for r in (A.kruskal_capped(n, edges, 2), A.prim_capped(n, edges, 2)):
        assert not r.feasible and len(r.tree) < n - 1
    n, edges = HOP_FAIL
    p = A.prim_hop(n, edges, 2)
    assert not p.feasible and len(p.tree) < n - 1


# --- 3. Exaktheit der Referenz ----------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("delta", [1, 2, 3])
def test_exact_degree_matches_brute_force_including_infeasible_cases(delta):
    for n, edges in _graphs(150):
        ref = brute_best(n, edges, _deg_ok(n, edges, delta))
        ex = A.exact_tree(n, edges, "degree", delta)
        assert ex.proved
        if ref is None:
            assert not ex.feasible and ex.cost == math.inf
        else:
            assert ex.feasible and ex.cost == pytest.approx(ref)
        lg = A.lagrange_degree(n, edges, delta)
        ex2 = A.exact_tree(n, edges, "degree", delta, pi=lg.pi, ub_tree=lg.tree or None)
        assert ex2.proved and ex2.feasible == (ref is not None) and (ref is None or ex2.cost == pytest.approx(ref))


@pytest.mark.parametrize("hops", [1, 2, 3, 4])
def test_exact_hops_matches_brute_force_including_infeasible_cases(hops):
    for n, edges in _graphs(150):
        ref = brute_best(n, edges, _hop_ok(n, edges, hops))
        ex = A.exact_tree(n, edges, "hops", hops)
        assert ex.proved
        if ref is None:
            assert not ex.feasible
        else:
            assert ex.feasible and ex.cost == pytest.approx(ref)


def test_hamiltonian_path_special_case_delta_two_on_a_complete_graph():
    rng = np.random.default_rng(11)
    for _ in range(25):
        n = 6
        edges = [(u, v, float(np.round(rng.uniform(1, 9), 3))) for u in range(n) for v in range(u + 1, n)]
        w = {(u, v): c for u, v, c in edges}
        best = min(sum(w[tuple(sorted(p))] for p in zip(perm, perm[1:])) for perm in permutations(range(n)))
        ex = A.exact_tree(n, edges, "degree", 2)
        assert ex.feasible and ex.cost == pytest.approx(best)


# --- 4. Sonderfälle ---------------------------------------------------------------------------------------------------------------------------------


def test_loose_limits_give_the_mst():
    for n, edges in _graphs(120):
        mst = A.kruskal(n, edges)
        deg = max(A.degrees(n, edges, mst.tree))
        depth = max(A.depths_from(n, edges, mst.tree))
        assert sorted(A.kruskal_capped(n, edges, deg).tree) == sorted(mst.tree)
        assert A.exact_tree(n, edges, "degree", deg).cost == pytest.approx(mst.cost)
        assert A.degree_greedy(n, edges, deg).cost == pytest.approx(mst.cost)
        assert A.exact_tree(n, edges, "hops", depth).cost == pytest.approx(mst.cost)
        assert A.hop_greedy(n, edges, depth).cost == pytest.approx(mst.cost)
        assert A.exact_tree(n, edges, "degree", n - 1).cost == pytest.approx(mst.cost)
        assert A.exact_tree(n, edges, "hops", n - 1).cost == pytest.approx(mst.cost)


def test_h_equals_one_is_the_star_and_needs_every_depot_edge():
    for seed in range(20):
        n = 6
        rng = np.random.default_rng(seed)
        edges = [(u, v, float(np.round(rng.uniform(1, 9), 3))) for u in range(n) for v in range(u + 1, n)]
        star = A.star_tree(n, edges)
        assert star is not None and A.tree_cost(edges, star) == pytest.approx(sum(w for u, v, w in edges if u == 0))
        ex = A.exact_tree(n, edges, "hops", 1)
        assert ex.cost == pytest.approx(A.tree_cost(edges, star)) and sorted(ex.tree) == sorted(star)
        assert A.hop_greedy(n, edges, 1).cost == pytest.approx(A.tree_cost(edges, star))
    n, edges = 4, [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)]
    assert A.star_tree(n, edges) is None
    assert A.exact_tree(n, edges, "hops", 1).feasible is False and A.hop_greedy(n, edges, 1).feasible is False


def test_degree_limit_one_is_only_possible_for_two_nodes():
    assert not A.exact_tree(3, [(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)], "degree", 1).feasible
    ex = A.exact_tree(2, [(0, 1, 2.5)], "degree", 1)
    assert ex.feasible and ex.cost == 2.5


def test_tiny_and_degenerate_graphs():
    assert A.exact_tree(1, [], "degree", 2).feasible and A.exact_tree(1, [], "hops", 1).feasible
    mst = A.kruskal(1, [])
    assert mst.tree == [] and mst.connected
    edges = [(0, 1, 3.0)]
    assert A.exact_tree(2, edges, "hops", 1).cost == 3.0 and A.exact_tree(2, edges, "degree", 2).cost == 3.0
    assert A.bottleneck_threshold(2, edges) == 3.0 and A.bottleneck_threshold(1, []) == 0.0
    split = [(0, 1, 1.0), (2, 3, 1.0)]
    assert A.bottleneck_threshold(4, split) is None and A.bottleneck_optimal_trees(4, split) == {}
    assert not A.kruskal(4, split).connected
    assert not A.exact_tree(4, split, "degree", 3).feasible
    with pytest.raises(ValueError):
        A.exact_tree(30, [(u, v, 1.0) for u in range(30) for v in range(u + 1, 30)], "degree", 3)


def test_equal_costs_everywhere_and_chain_and_star_inputs():
    n = 6
    complete = [(u, v, 1.0) for u in range(n) for v in range(u + 1, n)]
    assert A.exact_tree(n, complete, "degree", 2).cost == n - 1 and A.exact_tree(n, complete, "hops", 1).cost == n - 1
    chain = [(i, i + 1, 2.0) for i in range(n - 1)]
    assert A.exact_tree(n, chain, "degree", 2).cost == 2.0 * (n - 1) and not A.exact_tree(n, chain, "hops", n - 2).feasible
    star = [(0, i, 1.0 + i) for i in range(1, n)]
    assert not A.exact_tree(n, star, "degree", 2).feasible and A.exact_tree(n, star, "hops", 1).feasible
    assert A.degree_greedy(n, star, 2).feasible is False


# --- 5. Euklidischer Satz ---------------------------------------------------------------------------------------------------------------------------


def test_mst_degree_on_euclidean_instances_is_at_most_five_and_terrain_can_break_it():
    plain = ev.degree_profile("depot", 30, 0.0)
    assert sum(plain.values()) == 300 and max(plain) <= 5
    assert max(ev.degree_profile("hubs", 30, 0.0, 5)) <= 6
    assert max(ev.degree_profile("depot", 30, 1.0)) >= 6


def test_mst_degree_bound_holds_on_the_cross_fixture():
    t = S.textbook_instance()
    assert max(A.degrees(t.n, t.edges, A.kruskal(t.n, t.edges).tree)) == 4


# --- 6. Heuristiken und Schranken -------------------------------------------------------------------------------------------------------------------


def test_bound_chain_lower_bound_le_exact_le_heuristics():
    for n, edges in _graphs(150):
        mst = A.kruskal(n, edges).cost
        for delta in (2, 3):
            ref = brute_best(n, edges, _deg_ok(n, edges, delta))
            g = A.degree_greedy(n, edges, delta)
            lg = A.lagrange_degree(n, edges, delta, g.tree if g.feasible else None)
            if ref is None:
                assert not g.feasible and not lg.tree
                continue
            assert mst - 1e-9 <= lg.lb <= ref + 1e-7
            if g.feasible:
                assert g.cost >= ref - 1e-9
            if lg.tree:
                assert lg.ub >= ref - 1e-9 and lg.ub >= lg.lb - 1e-9
            if lg.optimal:
                assert lg.ub == pytest.approx(lg.lb, rel=1e-6) and lg.ub == pytest.approx(ref, rel=1e-6)
        for hops in (2, 3):
            ref = brute_best(n, edges, _hop_ok(n, edges, hops))
            g = A.hop_greedy(n, edges, hops)
            if ref is not None:
                assert g.feasible and g.cost >= ref - 1e-9 and ref >= mst - 1e-9


def test_lagrange_bound_is_valid_for_arbitrary_penalties():
    rng = np.random.default_rng(5)
    for n, edges in list(_graphs(80)):
        ref = brute_best(n, edges, _deg_ok(n, edges, 2))
        if ref is None:
            continue
        for _ in range(3):
            pi = [float(x) for x in rng.uniform(0, 3, size=n)]
            _tree, mod = A._mst_modified(n, edges, pi)
            assert mod - 2 * math.fsum(pi) <= ref + 1e-9


def test_lagrange_is_deterministic_and_history_is_monotone():
    n, edges = DEG_WORSE
    a, b = A.lagrange_degree(n, edges, 2), A.lagrange_degree(n, edges, 2)
    assert (a.lb, a.pi, a.history, a.tree) == (b.lb, b.pi, b.history, b.tree)
    bests = [h[2] for h in a.history]
    assert bests == sorted(bests) and a.iterations == len(a.history)
    ubs = [h[3] for h in a.history]
    assert ubs == sorted(ubs, reverse=True)


def _swap_neighbourhood(n, edges, tree):
    in_tree = set(tree)
    for e in range(len(edges)):
        if e in in_tree:
            continue
        for f in A.tree_path(A.adjacency(n, edges, tree), edges[e][0], edges[e][1]):
            yield [x for x in tree if x != f] + [e]


def test_exchange_moves_never_worsen_and_end_in_a_local_optimum():
    checked = 0
    for n, edges in _graphs(150):
        for delta in (2, 3):
            g = A.kruskal_capped(n, edges, delta)
            if not g.feasible:
                continue
            tree, _ops = A.exchange_degree(n, edges, g.tree, delta)
            assert A.is_spanning_tree(n, edges, tree) and max(A.degrees(n, edges, tree)) <= delta
            assert A.tree_cost(edges, tree) <= g.cost + 1e-9
            for cand in _swap_neighbourhood(n, edges, tree):
                if max(A.degrees(n, edges, cand)) <= delta:
                    assert A.tree_cost(edges, cand) >= A.tree_cost(edges, tree) - 1e-9
            checked += 1
        for hops in (2, 3):
            lt, depth = A.layer_tree(n, edges)
            if lt is None or depth > hops:
                continue
            tree, _ops = A.exchange_hop(n, edges, lt, hops)
            assert A.is_spanning_tree(n, edges, tree) and max(A.depths_from(n, edges, tree)) <= hops
            assert A.tree_cost(edges, tree) <= A.tree_cost(edges, lt) + 1e-9
            for cand in _swap_neighbourhood(n, edges, tree):
                if max(A.depths_from(n, edges, cand)) <= hops and min(A.depths_from(n, edges, cand)) >= 0:
                    assert A.tree_cost(edges, cand) >= A.tree_cost(edges, tree) - 1e-9
            checked += 1
    assert checked > 100


def test_layer_tree_has_the_smallest_possible_depth_and_exists_iff_a_hop_tree_does():
    for n, edges in _graphs(150):
        lt, depth = A.layer_tree(n, edges)
        assert lt is not None and A.is_spanning_tree(n, edges, lt)
        best = min(max(A.depths_from(n, edges, t)) for t in all_spanning_trees(n, edges))
        assert depth == best == max(A.depths_from(n, edges, lt))
    assert A.layer_tree(4, [(0, 1, 1.0), (2, 3, 1.0)]) == (None, None)


# --- 7. Schwierige Fixtures -------------------------------------------------------------------------------------------------------------------------


def test_greedy_can_fail_although_a_degree_two_tree_exists():
    n, edges = DEG_FAIL
    ref = brute_best(n, edges, _deg_ok(n, edges, 2))
    assert ref == pytest.approx(35.289)
    g = A.degree_greedy(n, edges, 2)
    assert not g.feasible and not g.detail["kruskal_ok"] and not g.detail["prim_ok"]
    ex = A.exact_tree(n, edges, "degree", 2)
    assert ex.feasible and ex.cost == pytest.approx(ref)
    lg = A.lagrange_degree(n, edges, 2)
    assert lg.tree and lg.ub == pytest.approx(ref)


def test_greedy_can_be_valid_but_strictly_worse_than_exact():
    n, edges = DEG_WORSE
    ref = brute_best(n, edges, _deg_ok(n, edges, 2))
    g = A.degree_greedy(n, edges, 2)
    assert ref == pytest.approx(21.818) and g.feasible and g.cost == pytest.approx(24.207)
    assert A.exact_tree(n, edges, "degree", 2).cost == pytest.approx(ref)


def test_prim_with_a_depth_limit_can_get_stuck_although_a_hop_tree_exists():
    n, edges = HOP_FAIL
    ref = brute_best(n, edges, _hop_ok(n, edges, 2))
    assert ref == pytest.approx(20.119)
    assert not A.prim_hop(n, edges, 2).feasible
    g = A.hop_greedy(n, edges, 2)
    assert g.feasible and g.detail["prim_ok"] is False and g.detail["layer_ok"] is True and g.cost >= ref - 1e-9
    assert A.exact_tree(n, edges, "hops", 2).cost == pytest.approx(ref)


def test_hop_greedy_can_be_valid_but_strictly_worse_than_exact():
    n, edges = HOP_WORSE
    ref = brute_best(n, edges, _hop_ok(n, edges, 2))
    g = A.hop_greedy(n, edges, 2)
    assert ref == pytest.approx(27.823) and g.feasible and g.cost == pytest.approx(27.841)
    assert A.exact_tree(n, edges, "hops", 2).cost == pytest.approx(ref)


def test_cross_fixture_by_hand():
    t = S.textbook_instance()
    n, E = t.n, t.edges
    mst = A.kruskal(n, E)
    assert mst.cost == pytest.approx(80.0) and A.degrees(n, E, mst.tree) == [1, 4, 1, 1, 1]
    assert A.exact_tree(n, E, "degree", 3).cost == pytest.approx(80.0 - 20.0 + 20.0 * math.sqrt(2))
    assert A.exact_tree(n, E, "degree", 2).cost == pytest.approx(96.5685424949238)
    assert A.exact_tree(n, E, "hops", 2).cost == pytest.approx(80.0)
    assert A.exact_tree(n, E, "hops", 1).cost == pytest.approx(20.0 + 40.0 + 2 * 20.0 * math.sqrt(2))


# --- 8. Buchführung, Branch-and-Bound, Kopie treu ---------------------------------------------------------------------------------------------------


def test_node_cap_makes_the_result_unproved_but_valid():
    n, edges = DEG_WORSE
    ex = A.exact_tree(n, edges, "degree", 2, node_cap=1)
    assert not ex.proved and ex.nodes >= 1
    if ex.feasible:
        assert A.is_spanning_tree(n, edges, ex.tree) and max(A.degrees(n, edges, ex.tree)) <= 2
    full = A.exact_tree(n, edges, "degree", 2)
    assert full.proved and full.nodes > 1


def test_upper_bound_start_does_not_change_the_optimum_and_saves_nodes():
    for n, edges in _graphs(80):
        g = A.degree_greedy(n, edges, 2)
        plain = A.exact_tree(n, edges, "degree", 2)
        seeded = A.exact_tree(n, edges, "degree", 2, ub_tree=g.tree if g.feasible else None)
        assert seeded.proved and seeded.cost == pytest.approx(plain.cost) and seeded.nodes <= plain.nodes


def test_kruskal_copy_reproduces_the_sibling_numbers():
    inst = S.generate(30, 6, 0.3, False, 35)
    res = A.kruskal(inst.n, inst.edges)
    assert round(res.cost, 2) == 466.63 and res.ops == 929
    assert A.is_spanning_tree(inst.n, inst.edges, res.tree)


def test_capped_greedy_with_large_limit_equals_the_unconstrained_answers():
    for n, edges in _graphs(60):
        mst = A.kruskal(n, edges)
        ps = A.prim_capped(n, edges, n)
        assert ps.feasible and ps.cost == pytest.approx(mst.cost) and sorted(ps.tree) == sorted(mst.tree)
        ph = A.prim_hop(n, edges, n)
        assert ph.feasible and ph.cost == pytest.approx(mst.cost)
