"""Konstanten der Demo zum beschränkten Spannbaum: Instanz-Geometrie (wortgleich zu den Geschwistern), Regler, gemessene Werte, Presets."""
AREA = 100.0
DEPOT_XY = (15.0, 50.0)
N_MIN, N_MAX, DEFAULT_N, N_STEP = 5, 60, 12, 1
K_MIN, DEFAULT_K = 3, 6
TERRAIN_MIN, TERRAIN_MAX, DEFAULT_TERRAIN, TERRAIN_STEP = 0.0, 1.0, 0.0, 0.05
SEED_MAX = 999999
DEFAULT_SEED = 35
ROUND_UNIT = 1.0
KINDS = ("depot", "hubs", "textbook")
SATS_MIN, SATS_MAX, DEFAULT_SATS = 3, 8, 5
SWEEP_SEEDS = tuple(range(100000, 100005))
CONSTRAINTS = ("bottleneck", "degree", "hops")
CONSTRAINT_LABELS = {"bottleneck": "Bottleneck (längste Kante)", "degree": "Knotengrad Δ", "hops": "Hops H vom Depot"}
DEFAULT_CONSTRAINT = "degree"
DELTA_MIN, DELTA_MAX, DEFAULT_DELTA = 2, 6, 2
HOPS_MIN, HOPS_MAX, DEFAULT_HOPS = 1, 12, 4
K_OPTIONS = (3, 4, 5, 6, 8, 10, 15, 20, 40, 1000)
TERRAIN_OPTIONS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0)
N_EXACT = {"degree": 24, "hops": 12}
DELTA_SWEEP = (2, 3, 4, 5, 6)
HOPS_SWEEP = (1, 2, 3, 4, 5, 6, 8, 10)
N_SWEEP = (8, 12, 16, 20, 30)
K_SWEEP = (3, 4, 6, 10, 1000)
SATS_SWEEP = (3, 4, 5, 6, 8)
FEAS_SEEDS = tuple(range(200000, 200050))
TREE_OPTIONS = {"bottleneck": ("mst", "max", "random"), "degree": ("best", "greedy", "lagrange", "exact", "mst"), "hops": ("best", "greedy", "exact", "mst")}
ALL_TREES = ("best", "greedy", "lagrange", "exact", "mst", "max", "random")
TREE_LABELS = {"best": "Bester Fund", "greedy": "Greedy", "lagrange": "Lagrange", "exact": "Exakt", "mst": "MST", "max": "Teuerster bottleneck-optimaler", "random": "Zufälliger bottleneck-optimaler"}
DEFAULT_TREE = "best"
KIND_LABELS = {"depot": "Depot und Filialen (Karte)", "hubs": "Ortschaften (Verteiler mit Anschlüssen)", "textbook": "Lehrbuchbeispiel (Kreuz, 5 Knoten)"}

_BASE = {"kind": "depot", "n": 12, "k": 6, "terrain": 0.0, "round_costs": False, "sats": 5, "seed": 35, "constraint": "degree", "delta": 2, "hops": 4, "tree": "best"}
PRESETS = {
    "Standardfall (Voreinstellung)": dict(_BASE),
    "Greedy scheitert (Grad ≤ 2)": {**_BASE, "seed": 100001},
    "Ortschaften: Grad ≤ 3": {**_BASE, "kind": "hubs", "n": 20, "delta": 3},
    "Kostenlose Grenze (Grad ≤ 3)": {**_BASE, "n": 20, "delta": 3},
    "Hop-Grenze H = 3": {**_BASE, "constraint": "hops", "hops": 3},
    "Alle direkt ans Depot (H = 1)": {**_BASE, "constraint": "hops", "hops": 1, "k": 1000},
    "Bottleneck: Milliarden Bäume": {**_BASE, "n": 20, "constraint": "bottleneck", "tree": "mst"},
    "Lehrbuchbeispiel (Grad ≤ 2)": {**_BASE, "kind": "textbook"},
}
PRESET_HELP = {
    "Standardfall (Voreinstellung)": "12 Filialen, k = 6, Seed 35, Grad ≤ 2: der MST (Kosten 215.65, größter Grad 3) muss zum Pfad werden - Kosten 224.37, +4.04 %. Greedy, Lagrange und das exakte Verfahren finden dieselben 224.37; die Lagrange-Untergrenze schließt die Lücke schon nach 3 Iterationen.",
    "Greedy scheitert (Grad ≤ 2)": "Seed 100001: Kruskal und Prim mit Gradgrenze finden gar keinen Baum, obwohl einer existiert. Lagrange und das exakte Verfahren finden ihn: Kosten 267.66, +9.06 % gegen den MST (245.43).",
    "Ortschaften: Grad ≤ 3": "20 Filialen in Ortschaften mit je 5 Anschlussnehmern: der MST hat Knoten mit Grad 5 (Kosten 189.43). Grad ≤ 3 kostet +1.07 % (exakt, gleich der Lagrange-Untergrenze); Greedy mit Kantentausch liegt bei +1.89 %.",
    "Kostenlose Grenze (Grad ≤ 3)": "20 Filialen gleichverteilt: der MST hat höchstens Grad 3, die Grenze ändert nichts (+0.00 %). Ohne Geländezuschlag hatte der MST in allen 300 getesteten gleichverteilten Instanzen (n = 30, dicht) höchstens Grad 4; mit Zuschlag 1.0 kommen Grad 5 und 6 vor.",
    "Hop-Grenze H = 3": "12 Filialen: der MST ist bis zu 9 Hops tief. H ≤ 3 kostet +18.42 % (exakt, bewiesen), Greedy mit Kantentausch +33.91 %. Der größte Umweg sinkt von 2.74 auf 1.90 (Luftlinie = 1), aber die längste Kante wächst von 36.93 auf 42.43.",
    "Alle direkt ans Depot (H = 1)": "Vollständiger Graph, H = 1 ist der Stern: Kosten 555.17 gegen 215.65, +157.44 %, dafür kein Umweg mehr (Faktor 1.00). Der Depotknoten hat Grad 12.",
    "Bottleneck: Milliarden Bäume": "20 Filialen: 4.6 Milliarden Spannbäume haben denselben minimalen Bottleneck 33.30; der MST ist der billigste (316.00), ein beliebiger (per Zufallsreihenfolge gezogener) bottleneck-optimaler Baum kostet +49.08 %, der teuerste +74.57 %.",
    "Lehrbuchbeispiel (Grad ≤ 2)": "Fünf Punkte im Kreuz: der MST ist der Stern um C (Grad 4, Kosten 80). Grad ≤ 2 verlangt einen Pfad und kostet 96.57 (+20.71 %), Grad ≤ 3 kostet 88.28 (+10.36 %), H = 1 kostet 116.57 (+45.71 %).",
}
# Beobachtete Spannweite der Kennzahl (MEDIAN über die 5 festen Instanzen Seeds 100000-100004) je Preset, mit Sicherheitsabstand: (Kennzahl, untere, obere Grenze).
PRESET_EXPECTED_BANDS = {
    "Standardfall (Voreinstellung)": ("gap_best", 0.3, 1.2),
    "Greedy scheitert (Grad ≤ 2)": ("gap_best", 0.3, 1.2),
    "Ortschaften: Grad ≤ 3": ("gap_best", 0.6, 1.6),
    "Kostenlose Grenze (Grad ≤ 3)": ("gap_best", -0.01, 0.05),
    "Hop-Grenze H = 3": ("gap_best", 12.0, 18.0),
    "Alle direkt ans Depot (H = 1)": ("gap_best", 110.0, 160.0),
    "Bottleneck: Milliarden Bäume": ("log10_count", 9.5, 11.5),
}
