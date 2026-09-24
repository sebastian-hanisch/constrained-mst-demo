"""Auswertung: was kostet eine Nebenbedingung gegenüber dem MST?

Drei Nebenbedingungen (`constraint`): "bottleneck" (längste Kante), "degree" (Knotengrad <= delta), "hops" (Tiefe <= hops vom Depot). Verfahren auf derselben Instanz:
- Grad: Greedy (Kruskal/Prim mit Gradgrenze + Kantentausch), Lagrange (Knotenstrafen; liefert Untergrenze UND zulässige Bäume), exakt (Branch-and-Bound, nur kleine Instanzen).
- Hops: Greedy (Prim mit Tiefengrenze, Schichtenbaum, beide + Kantentausch), exakt (Branch-and-Bound, nur kleine Instanzen).
- Bottleneck: der MST und die Extreme der bottleneck-optimalen Bäume (billigster = MST, teuerster, zufälliger).

Kennzahlen laufen über 5 feste Instanzen (Seeds 100000-100004), Median mit 10./90. Perzentil. Alles ist deterministisch.
- **Preis** = Kosten / MST-Kosten - 1 in Prozent (der Baum muss die Nebenbedingung erfüllen; nur über Instanzen, in denen das Verfahren einen gültigen Baum findet).
- **Aufschlag** (`excess_<Verfahren>`) = Kosten des Verfahrens / Kosten des besten gefundenen Baums - 1: die Lücke der Heuristik, gemessen auf denselben Instanzen (der Preis `gap_<Verfahren>` nur über die, in denen
  das Verfahren einen Baum fand - die Ausfälle sind meist die schweren Instanzen, deshalb sind Preise verschiedener Verfahren nicht direkt vergleichbar).
- **Ausfall** = das Verfahren findet keinen gültigen Baum, obwohl der beste bekannte Verfahrensbaum (Lagrange/exakt/anderer Greedy) einen fand.
- **Untergrenze** = Lagrange-Untergrenze (Grad) bzw. MST-Kosten (Hops); exakte Werte nur, wo das Branch-and-Bound innerhalb der Knotengrenze fertig wurde (`proved`)."""

import math
from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np

import cons_algorithm as A
import cons_constants as C
import cons_scenario as S

INF = float("inf")
CONSTRAINTS = ("bottleneck", "degree", "hops")


@dataclass(frozen=True)
class Settings:
    kind: str = "depot"
    n: int = C.DEFAULT_N
    k: int = C.DEFAULT_K
    terrain: float = C.DEFAULT_TERRAIN
    round_costs: bool = False
    sats: int = C.DEFAULT_SATS
    seed: int = C.DEFAULT_SEED
    constraint: str = "degree"
    delta: int = C.DEFAULT_DELTA
    hops: int = C.DEFAULT_HOPS


@lru_cache(maxsize=256)
def instance_of(settings):
    if settings.kind == "textbook":
        return S.textbook_instance()
    return S.generate(settings.n, settings.k, settings.terrain, settings.round_costs, settings.seed, settings.kind, settings.sats)


@dataclass
class Found:
    name: str
    tree: list
    cost: float
    feasible: bool
    note: str = ""


def detour(inst, tree):
    """Umweg-Faktor je Filiale: Weglänge im Baum vom Depot / Luftlinie. Gibt (Median, Maximum) zurück."""
    adj = A.adjacency(inst.n, inst.edges, tree)
    dist = [None] * inst.n
    dist[0] = 0.0
    queue = [0]
    for u in queue:
        for v, i in adj[u]:
            if dist[v] is None:
                dist[v] = dist[u] + inst.edges[i][2]
                queue.append(v)
    f = []
    for v in range(1, inst.n):
        air = float(np.hypot(*(inst.xy[v] - inst.xy[0])))
        if dist[v] is not None and air > 1e-9:
            f.append(dist[v] / air)
    return (float(np.median(f)), float(max(f))) if f else (1.0, 1.0)


@dataclass
class Analysis:
    settings: Settings
    inst: object
    mst: object                        # KruskalResult
    b_star: float
    trees: dict                        # Name -> Found
    lagrange: object = None            # LagrangeResult (Grad)
    exact: object = None               # ExactResult, wenn angeboten
    exact_offered: bool = False
    min_hops: int = 0                  # kleinste mögliche Tiefe (Schichtenbaum), -1 = Depot erreicht nicht alle
    count_log10: float = 0.0           # log10 der Zahl bottleneck-optimaler Bäume
    threshold_edges: int = 0
    hop_detail: dict = None            # Greedy-Details der Hop-Verfahren (prim_ok, layer_ok, min_hops)

    @property
    def constraint(self):
        return self.settings.constraint

    @property
    def limit(self):
        return self.settings.delta if self.constraint == "degree" else self.settings.hops

    @property
    def mst_cost(self):
        return self.mst.cost

    @property
    def mst_max_degree(self):
        return max(A.degrees(self.inst.n, self.inst.edges, self.mst.tree), default=0)

    @property
    def mst_depth(self):
        return max(A.depths_from(self.inst.n, self.inst.edges, self.mst.tree), default=0)

    @property
    def mst_violates(self):
        if self.constraint == "degree":
            return self.mst_max_degree > self.settings.delta
        if self.constraint == "hops":
            return self.mst_depth > self.settings.hops
        return False

    def gap(self, name):
        f = self.trees.get(name)
        if f is None or not f.feasible or self.mst_cost <= 0:
            return None
        return 100.0 * (f.cost / self.mst_cost - 1.0)

    @property
    def best_name(self):
        cands = [f for f in self.trees.values() if f.feasible and f.name not in ("mst", "max", "random", "min")]
        return min(cands, key=lambda f: (f.cost, f.name)).name if cands else None

    @property
    def infeasible(self):
        """True, wenn BEWIESEN kein gültiger Baum existiert."""
        if self.constraint == "hops":
            return self.min_hops < 0 or self.min_hops > self.settings.hops
        if self.constraint == "degree":
            return bool(self.exact is not None and self.exact.proved and not self.exact.feasible)
        return False

    @property
    def lower_bound(self):
        if self.constraint == "degree" and self.lagrange is not None and math.isfinite(self.lagrange.lb):
            return max(self.mst_cost, self.lagrange.lb)
        return self.mst_cost

    @property
    def lb_gap(self):
        return 100.0 * (self.lower_bound / self.mst_cost - 1.0) if self.mst_cost > 0 else 0.0

    @property
    def proved(self):
        return bool(self.exact is not None and self.exact.proved and self.exact.feasible)

    def bottleneck_of(self, name):
        f = self.trees.get(name)
        return A.bottleneck(self.inst.edges, f.tree) if f is not None and f.feasible else None


def _found(name, tree, feasible, edges, note=""):
    cost = A.tree_cost(edges, tree) if feasible else INF
    return Found(name, list(tree) if feasible else [], cost, feasible, note)


def exact_offered(settings, inst):
    limit = C.N_EXACT.get(settings.constraint, 0)
    return settings.constraint != "bottleneck" and inst.n - 1 <= limit and inst.m <= A.MAX_EXACT_EDGES


def analyse(settings):
    inst = instance_of(settings)
    n, E = inst.n, inst.edges
    mst = A.kruskal(n, E)
    b_star = A.bottleneck_threshold(n, E)
    a = Analysis(settings, inst, mst, b_star if b_star is not None else 0.0, {"mst": Found("mst", list(mst.tree), mst.cost, True)})
    if settings.constraint == "bottleneck":
        trees = A.bottleneck_optimal_trees(n, E, settings.seed)
        for name in ("min", "max", "random"):
            if name in trees:
                a.trees[name] = _found(name, trees[name], True, E)
        subset = A.threshold_edges(E, b_star)
        a.threshold_edges = len(subset)
        a.count_log10 = A.count_spanning_trees(n, E, subset)[1]
        return a
    limit = settings.delta if settings.constraint == "degree" else settings.hops
    a.exact_offered = exact_offered(settings, inst)
    if settings.constraint == "degree":
        g = A.degree_greedy(n, E, limit)
        a.trees["greedy"] = _found("greedy", g.tree, g.feasible, E, "Kruskal" if g.detail.get("from") == "kruskal" else "Prim")
        lg = A.lagrange_degree(n, E, limit, g.tree if g.feasible else None)
        a.lagrange = lg
        if lg.tree:
            polished, _ = A.exchange_degree(n, E, lg.tree, limit)
            a.trees["lagrange"] = _found("lagrange", polished, True, E)
        else:
            a.trees["lagrange"] = _found("lagrange", [], False, E)
        ub = min((f for f in a.trees.values() if f.name in ("greedy", "lagrange") and f.feasible), key=lambda f: f.cost, default=None)
        if a.exact_offered:
            ex = A.exact_tree(n, E, "degree", limit, pi=lg.pi, ub_tree=ub.tree if ub else None)
            a.exact = ex
            a.trees["exact"] = _found("exact", ex.tree, ex.feasible, E, "bewiesen" if ex.proved else "nicht bewiesen")
        return a
    lt, depth = A.layer_tree(n, E)
    a.min_hops = depth if lt is not None else -1
    g = A.hop_greedy(n, E, limit)
    a.trees["greedy"] = _found("greedy", g.tree, g.feasible, E, g.detail.get("from", ""))
    a.hop_detail = g.detail
    if a.exact_offered and not a.infeasible:
        ex = A.exact_tree(n, E, "hops", limit, ub_tree=g.tree if g.feasible else None)
        a.exact = ex
        a.trees["exact"] = _found("exact", ex.tree, ex.feasible, E, "bewiesen" if ex.proved else "nicht bewiesen")
    return a


# --- Kennzahlen über feste Instanzen ------------------------------------------------------------------------------------------------------------


def _stats(values):
    values = [v for v in values if v is not None and not np.isnan(v) and v != INF]
    if not values:
        return float("nan"), float("nan"), float("nan")
    return float(np.median(values)), float(np.percentile(values, 10)), float(np.percentile(values, 90))


def run_config(base, seeds=C.SWEEP_SEEDS, **changes):
    s0 = replace(base, **changes)
    rows = [analyse(replace(s0, seed=seed)) for seed in seeds]
    out = {"n_runs": len(rows), "viol_mst": 100.0 * sum(r.mst_violates for r in rows) / len(rows)}
    cols = [("cost", [r.mst_cost for r in rows]), ("mst_maxdeg", [float(r.mst_max_degree) for r in rows]), ("mst_depth", [float(r.mst_depth) for r in rows]), ("b_star", [r.b_star for r in rows]),
            ("arcs", [float(r.inst.m) for r in rows])]
    if s0.constraint == "bottleneck":
        cols += [("log10_count", [r.count_log10 for r in rows]), ("gap_max", [r.gap("max") for r in rows]), ("gap_random", [r.gap("random") for r in rows]),
                 ("threshold_share", [100.0 * r.threshold_edges / max(1, r.inst.m) for r in rows])]
    else:
        names = ("greedy", "lagrange", "exact") if s0.constraint == "degree" else ("greedy", "exact")
        found = [{nm: r.gap(nm) for nm in names} for r in rows]
        best = [min((v for v in (r.gap(nm) for nm in r.trees if nm not in ("mst",)) if v is not None), default=None) for r in rows]
        out["feasible_share"] = 100.0 * sum(b is not None for b in best) / len(rows)
        out["infeasible_share"] = 100.0 * sum(r.infeasible for r in rows) / len(rows)
        for nm in names:
            out[f"fail_{nm}"] = 100.0 * sum(1 for r, f, b in zip(rows, found, best) if b is not None and f[nm] is None and (nm != "exact" or r.exact_offered)) / len(rows)
        if s0.constraint == "hops":
            out["fail_prim"] = 100.0 * sum(1 for r, b in zip(rows, best) if b is not None and not r.hop_detail["prim_ok"]) / len(rows)
        out["proved_share"] = 100.0 * sum(r.proved for r in rows) / len(rows)
        out["offered_share"] = 100.0 * sum(r.exact_offered for r in rows) / len(rows)
        cols += [(f"gap_{nm}", [f[nm] for f in found]) for nm in names]
        cols += [(f"excess_{nm}", [100.0 * (r.trees[nm].cost / r.trees[r.best_name].cost - 1.0) if r.best_name and nm in r.trees and r.trees[nm].feasible else None for r in rows]) for nm in names]
        cols += [("gap_best", best), ("gap_exact_proved", [r.gap("exact") if r.proved else None for r in rows]), ("gap_lb", [r.lb_gap for r in rows])]
        cols += [("bott_gap", [100.0 * (A.bottleneck(r.inst.edges, r.trees[r.best_name].tree) / r.b_star - 1.0) if r.best_name and r.b_star > 0 else None for r in rows])]
        um_m = [detour(r.inst, r.trees["mst"].tree)[1] for r in rows]
        um_b = [detour(r.inst, r.trees[r.best_name].tree)[1] if r.best_name else None for r in rows]
        cols += [("detour_mst", um_m), ("detour_best", um_b)]
        cols += [("depth_best", [float(max(A.depths_from(r.inst.n, r.inst.edges, r.trees[r.best_name].tree))) if r.best_name else None for r in rows])]
        cols += [("deg_best", [float(max(A.degrees(r.inst.n, r.inst.edges, r.trees[r.best_name].tree))) if r.best_name else None for r in rows])]
    for key, values in cols:
        out[key], out[f"{key}_lo"], out[f"{key}_hi"] = _stats(values)
    return out


SWEEP_VALUES = {"delta": C.DELTA_SWEEP, "hops": C.HOPS_SWEEP, "n": C.N_SWEEP, "k": C.K_SWEEP, "sats": C.SATS_SWEEP}
SWEEP_LABELS = {"delta": "Gradgrenze Δ", "hops": "Hop-Grenze H", "n": "Filialen n", "k": "Nächste Nachbarn k", "sats": "Anschlüsse je Verteiler"}


def sweep_params(constraint, kind):
    """Welche Regler ein Sweep für diese Nebenbedingung durchfahren kann."""
    first = {"degree": "delta", "hops": "hops", "bottleneck": "n"}[constraint]
    rest = [p for p in ("n", "k") if p != first]
    if kind == "hubs":
        rest.append("sats")
    return [first] + rest


def sweep(param, base=Settings(), values=None):
    values = SWEEP_VALUES[param] if values is None else values
    return [{"value": v, **run_config(base, **{param: v})} for v in values]


def feasibility(base, seeds=C.FEAS_SEEDS):
    """Wie oft gibt es einen gültigen Baum, und wie oft finden die Greedy-Verfahren ihn nicht? Über `seeds` Instanzen mit den Einstellungen von `base` (nur der Seed wechselt; nicht für "bottleneck")."""
    an = [analyse(replace(base, seed=seed)) for seed in seeds]
    found = [a.best_name is not None for a in an]
    out = {"n_runs": len(an), "found_share": 100.0 * sum(found) / len(an), "infeasible_share": 100.0 * sum(a.infeasible for a in an) / len(an),
           "mst_violates": 100.0 * sum(a.mst_violates for a in an) / len(an)}
    if base.constraint == "degree":
        out["greedy_fail"] = 100.0 * sum(1 for a, f in zip(an, found) if f and not a.trees["greedy"].feasible) / len(an)
        out["lagrange_rescue"] = 100.0 * sum(1 for a in an if not a.trees["greedy"].feasible and a.trees["lagrange"].feasible) / len(an)
        out["lagrange_fail"] = 100.0 * sum(1 for a, f in zip(an, found) if f and not a.trees["lagrange"].feasible) / len(an)
        out["unknown_share"] = 100.0 * sum(1 for a, f in zip(an, found) if not f and not a.infeasible) / len(an)
    else:
        out["prim_fail"] = 100.0 * sum(1 for a in an if a.min_hops >= 0 and a.min_hops <= base.hops and not a.hop_detail["prim_ok"]) / len(an)
        out["layer_fail"] = 100.0 * sum(1 for a in an if a.min_hops >= 0 and a.min_hops <= base.hops and not a.hop_detail["layer_ok"]) / len(an)
        out["unknown_share"] = 0.0
    return out


def bottleneck_counts(base, ns=(10, 20, 30, 60)):
    """Zahl bottleneck-optimaler Bäume (log10) und Mehrkosten des teuersten/zufälligen gegen den MST über n, mit den Einstellungen von `base`."""
    return [{"n": n, **run_config(replace(base, constraint="bottleneck"), n=n)} for n in ns]


def degree_profile(kind="depot", n=30, terrain=0.0, sats=C.DEFAULT_SATS, seeds=range(300)):
    """Wie groß wird der größte Grad im MST? Zählt über `seeds` gleich große dichte Instanzen, wie oft welcher größte Grad vorkommt: {Grad: Anzahl}. Auf euklidischen Instanzen ohne Geländezuschlag
    ist er höchstens 6 (in Allgemeinlage höchstens 5)."""
    counts = {}
    for seed in seeds:
        inst = S.generate(n, 1000, terrain, False, seed, kind, sats)
        d = max(A.degrees(inst.n, inst.edges, A.kruskal(inst.n, inst.edges).tree))
        counts[d] = counts.get(d, 0) + 1
    return dict(sorted(counts.items()))
