"""Spannbäume mit Nebenbedingungen: Bottleneck, Knotengrad, Hop-Grenze - und Kruskal als Ausgangspunkt.

Der MST (Kruskal, wortgleich aus den Geschwister-Demos) minimiert die SUMME der Kantenkosten. Drei Nebenbedingungen, die reale Netze haben:
- **Bottleneck**: die längste Kante soll klein sein. Der MST ist dafür schon optimal (Satz, im Test gegen Brute-Force geprüft); die bottleneck-optimalen Bäume sind genau die Spannbäume
  des Graphen aller Kanten <= b* (b* = kleinste Schwelle, ab der der Graph zusammenhängt), und ihre Zahl (Kirchhoff, Determinante) ist meist riesig - der MST ist der billigste von ihnen.
- **Grad**: jeder Knoten hat höchstens Delta Anschlüsse. Für jedes Delta >= 2 NP-schwer (Delta = 2 ist der billigste Hamiltonpfad). Verfahren hier: Kruskal/Prim mit Gradgrenze (können
  scheitern), Kantentausch-Lokalsuche, Lagrange-Untergrenze mit Knotenstrafen (Volgenant 1989), exaktes Branch-and-Bound mit derselben Schranke (nur kleine Instanzen).
- **Hops**: jeder Knoten liegt höchstens H Kanten vom Depot (Wurzel, Knoten 0) entfernt. H = 1 ist der Stern, großes H der MST. Verfahren: Prim mit Tiefengrenze (kann scheitern), Schichtenbaum
  (Breitensuche, scheitert nie, wenn überhaupt ein Baum existiert) mit Umhänge-Lokalsuche, exaktes Branch-and-Bound (nur kleine Instanzen).

Kanten sind (u, v, w) mit u < v; der Schlüssel jeder Kante ist (Kosten, Kantenindex) - eine strikte Gesamtordnung. Ergebnisse tragen `feasible`: "kein gültiger Baum gefunden" wird nie als
Baum ausgegeben. Aufwand der Greedy-Verfahren in Elementarschritten (Vergleiche, Union-Find-Suchen, Nachbarschaftsprüfungen), NICHT in Laufzeit; das Branch-and-Bound zählt Knoten."""

import math
from dataclasses import dataclass, field

import numpy as np

from cons_unionfind import UnionFind

EPS = 1e-9


# --- Kruskal (wortgleich aus den Geschwistern, ohne Filter-Variante) ------------------------------------------------------------------------------


def counted_sort(items):
    """Eigener Mergesort mit Vergleichszähler; gibt (sortierte Liste, Vergleiche) zurück."""
    count = 0

    def merge(a, b):
        nonlocal count
        out, i, j = [], 0, 0
        while i < len(a) and j < len(b):
            count += 1
            if b[j] < a[i]:
                out.append(b[j])
                j += 1
            else:
                out.append(a[i])
                i += 1
        return out + a[i:] + b[j:]

    def sort(xs):
        if len(xs) <= 1:
            return xs
        mid = len(xs) // 2
        return merge(sort(xs[:mid]), sort(xs[mid:]))

    return sort(list(items)), count


@dataclass
class KruskalResult:
    tree: list
    cost: float
    examined: int = 0
    sort_comparisons: int = 0
    finds: int = 0
    find_steps: int = 0
    connected: bool = True

    @property
    def ops(self):
        return self.sort_comparisons + self.finds + self.find_steps


def kruskal(n, edges, uf_mode="full"):
    """Kruskal mit dem Schlüssel (Kosten, Kantenindex)."""
    keyed, comparisons = counted_sort([(edges[i][2], i) for i in range(len(edges))])
    uf = UnionFind(n, uf_mode)
    res = KruskalResult([], 0.0, sort_comparisons=comparisons)
    for _w, i in keyed:
        if len(res.tree) >= n - 1:
            break
        res.examined += 1
        u, v, w = edges[i]
        if uf.union(u, v):
            res.tree.append(i)
            res.cost += w
    res.finds, res.find_steps = uf.finds, uf.find_steps
    res.connected = uf.components == 1
    return res


# --- Baum-Werkzeuge -------------------------------------------------------------------------------------------------------------------------------


def tree_cost(edges, tree):
    return math.fsum(edges[i][2] for i in tree)


def bottleneck(edges, tree):
    """Längste Kante des Baums (0.0 für den leeren Baum)."""
    return max((edges[i][2] for i in tree), default=0.0)


def adjacency(n, edges, tree):
    adj = [[] for _ in range(n)]
    for i in tree:
        u, v, _w = edges[i]
        adj[u].append((v, i))
        adj[v].append((u, i))
    return adj


def degrees(n, edges, tree):
    deg = [0] * n
    for i in tree:
        u, v, _w = edges[i]
        deg[u] += 1
        deg[v] += 1
    return deg


def is_spanning_tree(n, edges, tree):
    """n-1 verschiedene Kanten, die alle n Knoten zusammenhängend verbinden (dann automatisch azyklisch)."""
    if len(tree) != max(n - 1, 0) or len(set(tree)) != len(tree):
        return False
    uf = UnionFind(n, "full")
    return all(uf.union(edges[i][0], edges[i][1]) for i in tree)


def depths_from(n, edges, tree, root=0):
    """Hop-Tiefe jedes Knotens von `root` im Baum bzw. Wald (-1 = nicht erreichbar)."""
    adj = adjacency(n, edges, tree)
    depth = [-1] * n
    if n == 0:
        return depth
    depth[root] = 0
    queue = [root]
    for u in queue:
        for v, _i in adj[u]:
            if depth[v] < 0:
                depth[v] = depth[u] + 1
                queue.append(v)
    return depth


def tree_path(adj, a, b):
    """Kantenindizes des Baumpfads a -> b (leer, falls a == b oder nicht verbunden)."""
    if a == b:
        return []
    prev = {a: (None, None)}
    queue = [a]
    for u in queue:
        if u == b:
            break
        for v, i in adj[u]:
            if v not in prev:
                prev[v] = (u, i)
                queue.append(v)
    if b not in prev:
        return []
    path, x = [], b
    while x != a:
        x, i = prev[x][0], prev[x][1]
        path.append(i)
    return path[::-1]


def minimax_path(n, edges, tree, a, b):
    """Größte Kantenlänge auf dem Baumpfad a -> b (0.0 für a == b)."""
    return max((edges[i][2] for i in tree_path(adjacency(n, edges, tree), a, b)), default=0.0)


@dataclass
class ConsResult:
    tree: list
    cost: float
    feasible: bool
    method: str = ""
    ops: int = 0
    detail: dict = field(default_factory=dict)


# --- Bottleneck -----------------------------------------------------------------------------------------------------------------------------------


def bottleneck_threshold(n, edges):
    """Kleinstes b*, ab dem der Graph aus allen Kanten <= b* zusammenhängt (= Bottleneck jedes MST); None, wenn er nie zusammenhängt."""
    if n <= 1:
        return 0.0
    res = kruskal(n, edges)
    return edges[res.tree[-1]][2] if res.connected else None


def threshold_edges(edges, b):
    return [i for i, e in enumerate(edges) if e[2] <= b + 1e-12]


def count_spanning_trees(n, edges, subset=None):
    """Anzahl der Spannbäume (Kirchhoff, Matrix-Baum-Satz) des Graphen aus den Kanten `subset` (Standard: alle) als (Betrag, log10) - der Betrag ist bei großen Zahlen nur eine Näherung."""
    idx = range(len(edges)) if subset is None else subset
    if n <= 1:
        return 1.0, 0.0
    lap = np.zeros((n, n))
    for i in idx:
        u, v, _w = edges[i]
        lap[u, u] += 1
        lap[v, v] += 1
        lap[u, v] -= 1
        lap[v, u] -= 1
    sign, logdet = np.linalg.slogdet(lap[1:, 1:])
    if sign <= 0:
        return 0.0, float("-inf")
    return float(np.exp(logdet)), float(logdet / math.log(10.0))


_MASK = (1 << 64) - 1


def _mix(seed, i):
    """Ganzzahliger Mischwert (SplitMix64) für die "zufällige" Reihenfolge: hängt nur von Seed und Kantenindex ab, nicht von der numpy-Version."""
    x = (int(seed) * 0x9E3779B97F4A7C15 + (i + 1) * 0xBF58476D1CE4E5B9) & _MASK
    x ^= x >> 30
    x = (x * 0xBF58476D1CE4E5B9) & _MASK
    x ^= x >> 27
    x = (x * 0x94D049BB133111EB) & _MASK
    return x ^ (x >> 31)


def bottleneck_optimal_trees(n, edges, seed=0):
    """Drei bottleneck-optimale Bäume (Spannbäume des Schwellwertgraphen): `min` = MST (billigster), `max` = teuerster (Kruskal mit absteigender Ordnung), `random` = zufällige Reihenfolge.
    Gibt {name: Kantenliste} zurück, leer wenn der Graph nicht zusammenhängt."""
    b = bottleneck_threshold(n, edges)
    if b is None:
        return {}
    cand = threshold_edges(edges, b)
    orders = {
        "min": sorted(cand, key=lambda i: (edges[i][2], i)),
        "max": sorted(cand, key=lambda i: (-edges[i][2], i)),
        "random": sorted(cand, key=lambda i: (_mix(seed, i), i)),
    }
    out = {}
    for name, order in orders.items():
        uf = UnionFind(n, "full")
        out[name] = [i for i in order if uf.union(edges[i][0], edges[i][1])]
    return out


# --- Grad: Greedy-Verfahren mit Gradgrenze --------------------------------------------------------------------------------------------------------


def kruskal_capped(n, edges, delta, weights=None):
    """Kruskal, aber eine Kante wird nur angenommen, wenn beide Endknoten noch weniger als `delta` Anschlüsse haben. Kann einen Wald liefern (`feasible=False`).
    `weights` (optional) sind abweichende Kosten für die Reihenfolge (Lagrange)."""
    w = [e[2] for e in edges] if weights is None else weights
    keyed, comparisons = counted_sort([(w[i], i) for i in range(len(edges))])
    uf = UnionFind(n, "full")
    deg = [0] * n
    tree = []
    for _k, i in keyed:
        if len(tree) >= n - 1:
            break
        u, v, _c = edges[i]
        if deg[u] >= delta or deg[v] >= delta:
            continue
        if uf.union(u, v):
            tree.append(i)
            deg[u] += 1
            deg[v] += 1
    ops = comparisons + uf.finds + uf.find_steps
    ok = len(tree) == n - 1
    return ConsResult(tree, tree_cost(edges, tree), ok, "kruskal", ops)


def prim_capped(n, edges, delta, start=0):
    """Prim ab `start` (Array-Variante): ein Knoten wird nur über einen Baumknoten mit weniger als `delta` Anschlüssen angeschlossen. Kann steckenbleiben (`feasible=False`)."""
    adj = [[] for _ in range(n)]
    for i, (u, v, w) in enumerate(edges):
        adj[u].append((v, w, i))
        adj[v].append((u, w, i))
    intree = [False] * n
    deg = [0] * n
    tree, ops = [], 0
    if n == 0:
        return ConsResult([], 0.0, True, "prim")
    intree[start] = True
    while len(tree) < n - 1:
        best = None
        for v in range(n):
            if intree[v]:
                continue
            for u, w, i in adj[v]:
                ops += 1
                if intree[u] and deg[u] < delta:
                    key = (w, i)
                    if best is None or key < best[0]:
                        best = (key, u, v)
        if best is None:
            break
        (_w, i), u, v = best
        intree[v] = True
        deg[u] += 1
        deg[v] += 1
        tree.append(i)
    return ConsResult(tree, tree_cost(edges, tree), len(tree) == n - 1, "prim", ops)


def exchange_degree(n, edges, tree, delta):
    """Kantentausch-Lokalsuche: eine Nichtbaumkante e kommt hinein, eine Kante f des von ihr geschlossenen Kreises fliegt raus, wenn das billiger ist und keine Gradgrenze verletzt.
    Beste Verbesserung je Runde, streng fallende Kosten, endet im lokalen Optimum. Erwartet einen gültigen Baum. Gibt (Baum, Prüfungen) zurück."""
    tree = list(tree)
    ops = 0
    deg = degrees(n, edges, tree)
    in_tree = set(tree)
    non_tree = [i for i in range(len(edges)) if i not in in_tree]
    while True:
        adj = adjacency(n, edges, tree)
        best = None
        for e in non_tree:
            a, b, w = edges[e]
            path = tree_path(adj, a, b)
            ops += len(path) + 1
            for f in path:
                fu, fv, fw = edges[f]
                gain = fw - w
                if gain <= EPS:
                    continue
                da = deg[a] + 1 - (1 if a in (fu, fv) else 0)
                db = deg[b] + 1 - (1 if b in (fu, fv) else 0)
                if da > delta or db > delta:
                    continue
                cand = (gain, -e, -f)
                if best is None or cand > best[0]:
                    best = (cand, e, f)
        if best is None:
            return tree, ops
        _cand, e, f = best
        tree.remove(f)
        tree.append(e)
        in_tree.discard(f)
        in_tree.add(e)
        non_tree.remove(e)
        non_tree.append(f)
        for x in edges[f][:2]:
            deg[x] -= 1
        for x in edges[e][:2]:
            deg[x] += 1


def degree_greedy(n, edges, delta):
    """Beste der beiden Greedy-Verfahren (Kruskal/Prim mit Gradgrenze), nach dem Kantentausch. `detail['kruskal_ok']`/`['prim_ok']` sagen, welche überhaupt einen Baum fanden."""
    ks = kruskal_capped(n, edges, delta)
    ps = prim_capped(n, edges, delta)
    cands, ops = [], ks.ops + ps.ops
    for r in (ks, ps):
        if r.feasible:
            tree, xo = exchange_degree(n, edges, r.tree, delta)
            ops += xo
            cands.append((tree_cost(edges, tree), r.method, tree))
    detail = {"kruskal_ok": ks.feasible, "prim_ok": ps.feasible, "kruskal_cost": ks.cost if ks.feasible else None, "prim_cost": ps.cost if ps.feasible else None}
    if not cands:
        return ConsResult([], math.inf, False, "greedy", ops, detail)
    cost, method, tree = min(cands, key=lambda c: (c[0], c[1]))
    detail["from"] = method
    return ConsResult(tree, cost, True, "greedy", ops, detail)


# --- Grad: Lagrange-Untergrenze (Knotenstrafen) ---------------------------------------------------------------------------------------------------


def _mst_modified(n, edges, pi):
    """MST für die Kosten c_uv + pi_u + pi_v; gibt (Baum, Summe der geänderten Kosten) zurück."""
    keyed = sorted((edges[i][2] + pi[edges[i][0]] + pi[edges[i][1]], i) for i in range(len(edges)))
    uf = UnionFind(n, "full")
    tree, mod = [], 0.0
    for wm, i in keyed:
        if uf.union(edges[i][0], edges[i][1]):
            tree.append(i)
            mod += wm
            if len(tree) == n - 1:
                break
    return tree, mod


@dataclass
class LagrangeResult:
    lb: float                                     # beste Untergrenze
    pi: list                                      # Strafen zur besten Untergrenze
    history: list = field(default_factory=list)   # je Iteration (Iteration, L(pi), beste Untergrenze, beste Obergrenze)
    tree: list = field(default_factory=list)      # bester gefundener zulässiger Baum (leer, falls keiner)
    ub: float = math.inf
    optimal: bool = False                         # Untergrenze == Obergrenze bewiesen
    iterations: int = 0


def lagrange_degree(n, edges, delta, ub_tree=None, iters=150):
    """Lagrange-Relaxation der Gradgrenze (Volgenant 1989): L(pi) = MST(c_uv + pi_u + pi_v) - delta * sum(pi), pi >= 0, ist für jedes pi eine Untergrenze der optimalen Kosten;
    maximiert per Subgradientenverfahren (Schrittweite nach Polyak, halbiert bei Stillstand; deterministisch). Jeder gefundene Baum mit Grad <= delta - auch der Kruskal-mit-Gradgrenze auf den
    geänderten Kosten - ist eine zulässige Lösung (Obergrenze)."""
    m = len(edges)
    if n <= 1 or m == 0:
        return LagrangeResult(0.0, [0.0] * n, [], [], 0.0, True, 0)
    pi = [0.0] * n
    best_lb, best_pi = -math.inf, pi[:]
    best_tree, best_ub = ([], math.inf)
    if ub_tree:
        best_tree, best_ub = list(ub_tree), tree_cost(edges, ub_tree)
    hist, theta, stall = [], 2.0, 0
    it = 0
    for it in range(1, iters + 1):
        tree, mod = _mst_modified(n, edges, pi)
        if len(tree) != n - 1:
            return LagrangeResult(math.inf, pi, hist, [], math.inf, False, it)
        lval = mod - delta * math.fsum(pi)
        deg = degrees(n, edges, tree)
        if max(deg) <= delta:
            c = tree_cost(edges, tree)
            if c < best_ub - EPS:
                best_ub, best_tree = c, list(tree)
        else:
            wmod = [e[2] + pi[e[0]] + pi[e[1]] for e in edges]
            kr = kruskal_capped(n, edges, delta, wmod)
            if kr.feasible and kr.cost < best_ub - EPS:
                best_ub, best_tree = kr.cost, list(kr.tree)
        if lval > best_lb + 1e-12:
            best_lb, best_pi, stall = lval, pi[:], 0
        else:
            stall += 1
            if stall >= 8:
                theta, stall = theta / 2.0, 0
        hist.append((it, lval, best_lb, best_ub))
        g = [deg[v] - delta if (deg[v] > delta or pi[v] > 0.0) else 0 for v in range(n)]
        norm = sum(x * x for x in g)
        if norm == 0:
            break
        if best_ub - best_lb <= 1e-9 * max(1.0, abs(best_lb)):
            break
        target = best_ub if math.isfinite(best_ub) else best_lb * 1.1 + 1.0
        step = theta * max(target - lval, 1e-9) / norm
        pi = [max(0.0, pi[v] + step * g[v]) for v in range(n)]
    optimal = math.isfinite(best_ub) and best_ub - best_lb <= 1e-7 * max(1.0, abs(best_lb))
    return LagrangeResult(best_lb, best_pi, hist, best_tree, best_ub, optimal, it)


# --- Hops: Verfahren mit Tiefengrenze -------------------------------------------------------------------------------------------------------------


def prim_hop(n, edges, hops, root=0):
    """Prim ab der Wurzel; ein Knoten wird nur unter einen Baumknoten mit Tiefe < hops gehängt. Kann steckenbleiben (`feasible=False`)."""
    adj = [[] for _ in range(n)]
    for i, (u, v, w) in enumerate(edges):
        adj[u].append((v, w, i))
        adj[v].append((u, w, i))
    depth = [-1] * n
    tree, ops = [], 0
    if n == 0:
        return ConsResult([], 0.0, True, "prim")
    depth[root] = 0
    while len(tree) < n - 1:
        best = None
        for v in range(n):
            if depth[v] >= 0:
                continue
            for u, w, i in adj[v]:
                ops += 1
                if depth[u] >= 0 and depth[u] < hops:
                    key = (w, i)
                    if best is None or key < best[0]:
                        best = (key, u, v)
        if best is None:
            break
        (_w, i), u, v = best
        depth[v] = depth[u] + 1
        tree.append(i)
    return ConsResult(tree, tree_cost(edges, tree), len(tree) == n - 1, "prim", ops)


def layer_tree(n, edges, root=0):
    """Schichtenbaum: Breitensuche von der Wurzel, jeder Knoten hängt an der billigsten Kante zu einem Knoten der Schicht darüber. Hat die kleinstmögliche Tiefe
    (= Hop-Abstand); gibt (Baum, Tiefe) zurück oder (None, None), wenn nicht alle Knoten erreichbar sind."""
    adj = [[] for _ in range(n)]
    for i, (u, v, w) in enumerate(edges):
        adj[u].append((v, w, i))
        adj[v].append((u, w, i))
    level = [-1] * n
    if n == 0:
        return [], 0
    level[root] = 0
    queue = [root]
    for u in queue:
        for v, _w, _i in adj[u]:
            if level[v] < 0:
                level[v] = level[u] + 1
                queue.append(v)
    if min(level) < 0:
        return None, None
    tree = []
    for v in range(n):
        if v == root:
            continue
        best = min(((w, i) for u, w, i in adj[v] if level[u] == level[v] - 1))
        tree.append(best[1])
    return tree, max(level)


def _rooted(n, edges, tree, root):
    """Elternzeiger, Elternkante, Tiefe, Ein-/Austrittszeit und größte Tiefe im Teilbaum (absolut) eines gültigen Baums."""
    adj = adjacency(n, edges, tree)
    parent, pedge, depth = [-1] * n, [-1] * n, [-1] * n
    depth[root] = 0
    order = [root]
    for u in order:
        for v, i in adj[u]:
            if depth[v] < 0:
                parent[v], pedge[v], depth[v] = u, i, depth[u] + 1
                order.append(v)
    children = [[] for _ in range(n)]
    for v in order[1:]:
        children[parent[v]].append(v)
    tin, tout, sub_max = [0] * n, [0] * n, depth[:]
    clock, stack = 0, [(root, 0)]
    while stack:
        u, state = stack.pop()
        if state == 0:
            tin[u] = clock
            clock += 1
            stack.append((u, 1))
            for c in reversed(children[u]):
                stack.append((c, 0))
        else:
            tout[u] = clock
            for c in children[u]:
                sub_max[u] = max(sub_max[u], sub_max[c])
    return parent, pedge, depth, tin, tout, sub_max


def exchange_hop(n, edges, tree, hops, root=0):
    """Kantentausch-Lokalsuche mit Tiefengrenze: eine Nichtbaumkante e = (a, b) kommt hinein, eine Kante f = (p, c) des von ihr geschlossenen Kreises fliegt raus; der abgetrennte Teilbaum von c
    hängt dann mit a als neuer Spitze an b (also auch "Teilbaum umdrehen", nicht nur "Teilbaum umhängen"). Erlaubt, wenn e billiger ist als f und die größte Tiefe im umgehängten Teilbaum
    (Tiefe von b + 1 + größter Abstand von a im Teilbaum) <= hops bleibt. Beste Verbesserung je Runde, streng fallende Kosten. Erwartet einen gültigen Baum. Gibt (Baum, Prüfungen) zurück."""
    tree = list(tree)
    ops = 0
    while True:
        parent, pedge, depth, tin, tout, _sub_max = _rooted(n, edges, tree, root)
        adj = adjacency(n, edges, tree)
        in_tree = set(tree)
        best = None
        for e in range(len(edges)):
            if e in in_tree:
                continue
            a, b, w = edges[e]
            path = tree_path(adj, a, b)
            ops += len(path) + 1
            for f in path:
                gain = edges[f][2] - w
                if gain <= EPS:
                    continue
                c = edges[f][0] if pedge[edges[f][0]] == f else edges[f][1]
                inside_a = tin[c] <= tin[a] < tout[c]
                x, y = (a, b) if inside_a else (b, a)
                ecc = 0
                dist = {x: 0}
                queue = [x]
                for u in queue:
                    for v, i in adj[u]:
                        if i != f and v not in dist and tin[c] <= tin[v] < tout[c]:
                            dist[v] = dist[u] + 1
                            ecc = max(ecc, dist[v])
                            queue.append(v)
                ops += len(queue)
                if depth[y] + 1 + ecc > hops:
                    continue
                cand = (gain, -e, -f)
                if best is None or cand > best[0]:
                    best = (cand, e, f)
        if best is None:
            return tree, ops
        _c, e, f = best
        tree.remove(f)
        tree.append(e)


def hop_greedy(n, edges, hops):
    """Prim mit Tiefengrenze und Schichtenbaum, beide nach der Umhänge-Lokalsuche; nimmt den billigeren. `detail['prim_ok']` sagt, ob Prim überhaupt einen Baum fand."""
    ps = prim_hop(n, edges, hops)
    ops = ps.ops
    cands = []
    if ps.feasible:
        tree, xo = exchange_hop(n, edges, ps.tree, hops)
        ops += xo
        cands.append((tree_cost(edges, tree), "prim", tree))
    lt, depth = layer_tree(n, edges)
    layer_ok = lt is not None and depth <= hops
    if layer_ok:
        tree, xo = exchange_hop(n, edges, lt, hops)
        ops += xo
        cands.append((tree_cost(edges, tree), "layer", tree))
    detail = {"prim_ok": ps.feasible, "prim_cost": ps.cost if ps.feasible else None, "layer_ok": layer_ok, "min_hops": depth}
    if not cands:
        return ConsResult([], math.inf, False, "greedy", ops, detail)
    cost, method, tree = min(cands, key=lambda c: (c[0], c[1]))
    detail["from"] = method
    return ConsResult(tree, cost, True, "greedy", ops, detail)


def star_tree(n, edges, root=0):
    """Der Stern (H = 1): jeder Knoten direkt an der Wurzel; None, wenn eine Wurzelkante fehlt."""
    direct = {}
    for i, (u, v, _w) in enumerate(edges):
        if u == root:
            direct[v] = i
        elif v == root:
            direct[u] = i
    if any(v not in direct for v in range(n) if v != root):
        return None
    return [direct[v] for v in range(n) if v != root]


# --- Exaktes Branch-and-Bound (kleine Instanzen) --------------------------------------------------------------------------------------------------

NODE_CAP = 20000
MAX_EXACT_EDGES = 300


@dataclass
class ExactResult:
    tree: list
    cost: float
    feasible: bool                                # ein gültiger Baum wurde gefunden
    proved: bool                                  # Suche vollständig: Baum optimal bzw. (feasible=False) kein gültiger Baum existiert
    nodes: int = 0


def _hop_partial_ok(n, edges, included, hops, root):
    """Notwendige Bedingung für die bisher gewählten Kanten: Tiefe der Wurzelkomponente <= hops, und jedes Fragment ohne Wurzel muss (Radius + 1) <= hops erlauben."""
    adj = adjacency(n, edges, included)
    seen = [False] * n
    depth = {root: 0}
    seen[root] = True
    queue = [root]
    for u in queue:
        for v, _i in adj[u]:
            if not seen[v]:
                seen[v] = True
                depth[v] = depth[u] + 1
                if depth[v] > hops:
                    return False
                queue.append(v)
    for s in range(n):
        if seen[s]:
            continue
        comp, q = [s], [s]
        seen[s] = True
        for u in q:
            for v, _i in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    comp.append(v)
                    q.append(v)
        if len(comp) == 1:
            continue
        radius = math.inf
        for a in comp:
            dist = {a: 0}
            qq = [a]
            for u in qq:
                for v, _i in adj[u]:
                    if v not in dist:
                        dist[v] = dist[u] + 1
                        qq.append(v)
            radius = min(radius, max(dist.values()))
        if radius + 1 > hops:
            return False
    return True


def exact_tree(n, edges, kind, limit, root=0, pi=None, ub_tree=None, node_cap=NODE_CAP):
    """Branch-and-Bound über Kanten (Entscheidung nehmen/lassen in Schlüsselreihenfolge). Schranke je Knoten: gewählte Kosten + Kruskal-Ergänzung über die noch offenen Kanten (für den Grad mit
    den Lagrange-Kosten `pi`, sonst mit den echten Kosten), ohne die Nebenbedingung - eine gültige Untergrenze. `kind` = 'degree' (limit = delta) oder 'hops' (limit = H, Wurzel `root`).
    `ub_tree` (zulässig, z. B. aus dem Greedy) startet die obere Schranke. `proved=False`, wenn `node_cap` Suchknoten nicht reichen (dann ist das Ergebnis nur die beste gefundene Lösung)."""
    m = len(edges)
    if n <= 1:
        return ExactResult([], 0.0, True, True, 0)
    if m > MAX_EXACT_EDGES:
        raise ValueError("zu viele Kanten für das exakte Verfahren")
    if kind == "degree" and limit < 1:
        return ExactResult([], math.inf, False, True, 0)
    if kind == "degree" and limit < 2 and n > 2:
        return ExactResult([], math.inf, False, True, 0)
    if kind == "hops":
        lt, depth = layer_tree(n, edges, root)
        if lt is None or depth > limit:
            return ExactResult([], math.inf, False, True, 0)
    pi = [0.0] * n if (kind != "degree" or pi is None) else list(pi)
    offset = -limit * math.fsum(pi) if kind == "degree" else 0.0
    order = sorted(range(m), key=lambda i: (edges[i][2], i))
    rank = [0] * m
    for pos, i in enumerate(order):
        rank[i] = pos
    mod = [edges[i][2] + pi[edges[i][0]] + pi[edges[i][1]] for i in range(m)]
    mod_order = sorted(range(m), key=lambda i: (mod[i], i))
    best = {"tree": list(ub_tree) if ub_tree else [], "cost": tree_cost(edges, ub_tree) if ub_tree else math.inf}
    nodes = 0
    aborted = False

    def valid(tree):
        if kind == "degree":
            return max(degrees(n, edges, tree)) <= limit
        d = depths_from(n, edges, tree, root)
        return max(d) <= limit and min(d) >= 0

    def dfs(pos, included, deg):
        nonlocal nodes, aborted
        if aborted:
            return
        nodes += 1
        if nodes > node_cap:
            aborted = True
            return
        uf = UnionFind(n, "full")
        inc_mod = 0.0
        for i in included:
            uf.union(edges[i][0], edges[i][1])
            inc_mod += mod[i]
        completion = []
        for i in mod_order:
            if rank[i] >= pos and uf.union(edges[i][0], edges[i][1]):
                completion.append(i)
                inc_mod += mod[i]
        if len(included) + len(completion) != n - 1:
            return
        lb = inc_mod + offset
        if lb >= best["cost"] - EPS:
            return
        full = included + completion
        if valid(full):
            c = tree_cost(edges, full)
            if c < best["cost"] - EPS:
                best["tree"], best["cost"] = list(full), c
            if abs(c - lb) <= EPS:
                return
        if pos >= m:
            return
        e = order[pos]
        u, v, _w = edges[e]
        uf2 = UnionFind(n, "full")
        for i in included:
            uf2.union(edges[i][0], edges[i][1])
        if uf2.find(u) != uf2.find(v):
            ok = True
            if kind == "degree":
                ok = deg[u] < limit and deg[v] < limit
            if ok:
                if kind == "degree":
                    deg[u] += 1
                    deg[v] += 1
                inc = included + [e]
                if kind == "degree" or _hop_partial_ok(n, edges, inc, limit, root):
                    dfs(pos + 1, inc, deg)
                if kind == "degree":
                    deg[u] -= 1
                    deg[v] -= 1
        dfs(pos + 1, included, deg)

    dfs(0, [], [0] * n)
    feasible = bool(best["tree"])
    return ExactResult(best["tree"], best["cost"] if feasible else math.inf, feasible, not aborted, nodes)
