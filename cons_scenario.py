"""Die Instanz dieser Demo: ein Depot (Werk) und n Filialen (Kunden) als Punkte auf einer Karte; gesucht wird ein Leitungsnetz (Fernwärme, Glasfaser,
Sammelleitung), das alle verbindet - jetzt mit Nebenbedingungen (Knotengrad, Hops, Bottleneck). Zwei Layouts: gleichverteilt ("depot") und "Ortschaften" ("hubs": Verteiler mit
`sats` Anschlussnehmern im Kreis - dort hat der MST von Natur aus Knoten hohen Grades, damit die Gradgrenze etwas kostet). Kantenkosten = Trassenlänge = euklidischer Abstand mal Geländefaktor u in [1, 1 + Zuschlag] (damit der Baum nicht rein
geometrisch ist). Der Faktor hängt nur vom Knotenpaar und vom Seed ab, NICHT vom Kandidatengraphen - wer k ändert, streicht Kanten, ändert aber keine Kosten.

Kandidatengraph: `k >= n_gesamt - 1` = vollständig (dicht); sonst die k nächsten Nachbarn je Knoten (Vereinigung beider Richtungen), bei Bedarf um die kürzeste
Zwischenkante ergänzt, bis der Graph zusammenhängt (ohne diese Garantie wäre "kein Spannbaum" der Normalfall bei kleinem k, nicht die Ausnahme).

`round_costs` rundet die Kosten auf ganze km (mindestens 1): das erzeugt bewusst viele Gleichstände. Ein handgebautes Lehrbuchbeispiel (5 Knoten)."""

import math
from dataclasses import dataclass

import numpy as np

import cons_constants as C

INSTANCE_KINDS = ("depot", "hubs", "textbook")


@dataclass(frozen=True)
class Instance:
    xy: np.ndarray                 # (N, 2), Knoten 0 = Depot
    edges: tuple                   # ((u, v, w), ...) mit u < v, sortiert nach (u, v)
    kind: str = "depot"
    k: int = 0
    terrain: float = 0.0
    round_costs: bool = False
    seed: int = 0
    labels: object = None          # Knotennamen der Fixtures
    sats: int = 0                  # Anschlussnehmer je Verteiler (nur Layout "hubs")

    @property
    def n(self):
        return len(self.xy)

    @property
    def m(self):
        return len(self.edges)

    @property
    def depot(self):
        return 0


def _components_connect(n, edge_set, dist):
    """Ergänzt `edge_set` um die kürzeste Zwischenkante zweier Komponenten, bis alles zusammenhängt."""
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for u, v in edge_set:
        parent[find(u)] = find(v)
    while True:
        roots = [find(x) for x in range(n)]
        if len(set(roots)) == 1:
            return
        best = None
        for u in range(n):
            for v in range(u + 1, n):
                if roots[u] != roots[v] and (best is None or (dist[u, v], u, v) < best[0]):
                    best = ((dist[u, v], u, v), u, v)
        _, u, v = best
        edge_set.add((u, v))
        parent[find(u)] = find(v)


def _hub_points(n_customers, sats, seed):
    """Verteiler mit `sats` Anschlussnehmern im Kreis (Radius ~6); Verteiler mit Mindestabstand zufällig auf der Fläche. Genau `n_customers` Punkte (der letzte Verteiler ggf. mit weniger)."""
    rng = np.random.default_rng([int(seed), 910])
    n_hubs = max(1, -(-int(n_customers) // (sats + 1)))
    centers = []
    min_d = 24.0
    tries = 0
    while len(centers) < n_hubs:
        c = rng.uniform(12.0, C.AREA - 12.0, size=2)
        tries += 1
        if all(np.hypot(*(c - o)) >= min_d for o in centers) or tries > 500:
            centers.append(c)
        if tries > 500:
            min_d = 0.0
    pts = []
    for c in centers:
        pts.append(c)
        phase = rng.uniform(0.0, 2.0 * math.pi)
        for j in range(sats):
            ang = phase + 2.0 * math.pi * j / sats + rng.uniform(-0.15, 0.15)
            r = 6.0 * rng.uniform(0.9, 1.1)
            pts.append(c + r * np.array([math.cos(ang), math.sin(ang)]))
    return np.array(pts[: int(n_customers)])


def generate(n_customers=C.DEFAULT_N, k=C.DEFAULT_K, terrain=C.DEFAULT_TERRAIN, round_costs=False, seed=C.DEFAULT_SEED, kind="depot", sats=C.DEFAULT_SATS):
    """Depot + `n_customers` Filialen; Knoten 0 ist das Depot."""
    if kind not in ("depot", "hubs"):
        raise ValueError(f"unbekanntes Layout {kind}")
    n = int(n_customers) + 1
    rng = np.random.default_rng([int(seed), 909])
    if kind == "depot":
        xy = np.vstack([np.array([C.DEPOT_XY]), rng.uniform(0.0, C.AREA, size=(n - 1, 2))])
    else:
        xy = np.vstack([np.array([C.DEPOT_XY]), _hub_points(n - 1, int(sats), seed)])
    factor = rng.uniform(1.0, 1.0 + float(terrain), size=(n, n))
    factor = np.triu(factor, 1) + np.triu(factor, 1).T
    diff = xy[:, None, :] - xy[None, :, :]
    dist = np.hypot(diff[:, :, 0], diff[:, :, 1])
    cost = dist * np.where(factor > 0, factor, 1.0)
    if round_costs:
        cost = np.maximum(C.ROUND_UNIT, np.round(cost / C.ROUND_UNIT) * C.ROUND_UNIT)
    if k >= n - 1:
        edge_set = {(u, v) for u in range(n) for v in range(u + 1, n)}
    else:
        edge_set = set()
        order = np.argsort(dist + np.diag(np.full(n, np.inf)), axis=1, kind="stable")
        for u in range(n):
            for v in order[u, :k]:
                edge_set.add((min(u, int(v)), max(u, int(v))))
        _components_connect(n, edge_set, dist)
    edges = tuple((u, v, float(cost[u, v])) for u, v in sorted(edge_set))
    return Instance(xy, edges, kind, int(k), float(terrain), bool(round_costs), int(seed), None, int(sats) if kind == "hubs" else 0)


# --- Handgebaute Fixtures -----------------------------------------------------------------------------------------------------------------------

TEXTBOOK_XY = [(20.0, 50.0), (40.0, 50.0), (60.0, 50.0), (40.0, 70.0), (40.0, 30.0)]


def textbook_instance():
    """Fünf Knoten in Kreuzform, vollständiger Graph mit euklidischen Längen: W (Depot, links), C (Mitte), E, N, S. Der MST ist der Stern um C (vier Kanten der Länge 20, Kosten 80, Grad von C = 4);
    Grad <= 3 kostet 88.28 (S hängt stattdessen an W), Grad <= 2 kostet 96.57 (Pfad N-W-C-E-S), H = 1 (alle direkt an W) kostet 116.57."""
    xy = np.array(TEXTBOOK_XY)
    edges = []
    for u in range(5):
        for v in range(u + 1, 5):
            edges.append((u, v, float(np.hypot(*(xy[u] - xy[v])))))
    return Instance(xy, tuple(edges), "textbook", labels=("W", "C", "E", "N", "S"))
