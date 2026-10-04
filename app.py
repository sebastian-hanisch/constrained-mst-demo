"""Beschränkte Spannbäume – Bottleneck, Knotengrad, Hops - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Sechstes Stück der Spannbaum-Reihe der "Konzepte"-Reihe: der MST ist das Optimum ohne Nebenbedingung. Reale Netze haben Grenzen: ein Verteiler hat nur Δ Anschlüsse (Grad), ein Signal darf nur H Sprünge vom Depot
laufen (Hops), das schwächste Glied soll kurz sein (Bottleneck). Der MST ist bottleneck-optimal, aber nicht grad- oder hop-optimal; Grad- und Hop-Grenze machen das Problem NP-schwer. Gemessen werden der Preis der
Nebenbedingung, das Scheitern der Greedy-Verfahren, die Lagrange-Untergrenze und der Umweg-Gewinn.

Lauffähig mit: streamlit run app.py
"""

from dataclasses import replace

import streamlit as st

import cons_algorithm as A
import cons_constants as C
from cons_evaluation import SWEEP_LABELS, Settings, analyse, bottleneck_counts, degree_profile, detour, feasibility, sweep, sweep_params
from cons_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    store_from_widget,
    sync_query_params,
)
from cons_visualization import (
    METHOD_LABELS,
    build_bottleneck_counts,
    build_cost_bars,
    build_degree_view,
    build_hop_view,
    build_instance,
    build_lagrange,
    build_sweep,
    build_threshold,
    build_tree,
)

st.set_page_config(page_title="Beschränkte Spannbäume – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner=False)
def _sweep(param, base):
    return sweep(param, base)


@st.cache_data(show_spinner=False)
def _feasibility(base):
    return feasibility(base)


@st.cache_data(show_spinner=False)
def _bottleneck_counts(base):
    return bottleneck_counts(base)


@st.cache_data(show_spinner=False)
def _profile(kind, terrain, sats):
    return degree_profile(kind, 30, terrain, sats)


def pct(x):
    return "kein Baum" if x is None else f"{x:+.2f} %"


def num(x):
    return f"{x:.2f}"


st.title("🌲 Beschränkte Spannbäume – Grad, Hops, Bottleneck")
st.markdown(
    """
**Sechstes Stück der Spannbaum-Reihe.** Der minimale Spannbaum ist das Optimum ohne Nebenbedingung. Reale Netze haben Grenzen: ein **Verteiler hat nur Δ Anschlüsse** (Knotengrad), ein **Signal darf nur H Sprünge
vom Depot** laufen (Hops), das **schwächste Glied** - die längste Leitung - soll kurz sein (Bottleneck). Der MST ist für den Bottleneck schon optimal, aber weder grad- noch hop-optimal; die beiden letzten Nebenbedingungen machen
das Problem **NP-schwer** (Grad Δ = 2 ist ein Hamiltonpfad).

Hier wird gemessen, **was jede Grenze kostet**, ob die einfachen Greedy-Verfahren überhaupt einen gültigen Baum finden, wie nah eine **Lagrange-Untergrenze** (Knotenstrafen, Volgenant 1989) und ein exaktes
**Branch-and-Bound** (nur kleine Instanzen) an den besten Baum herankommen und was die Hop-Grenze am **Umweg** ändert.
"""
)
st.caption(
    "Setzt auf [kruskal-demo](https://github.com/sebastian-hanisch/kruskal-demo), [prim-demo](https://github.com/sebastian-hanisch/prim-demo) und [arborescence-demo](https://github.com/sebastian-hanisch/arborescence-demo) auf "
    "(der Kruskal-Baum ist hier der Ausgangspunkt). Nachfolger in der Reihe: [cmst-demo](https://github.com/sebastian-hanisch/cmst-demo) (Kapazitierter MST), [steiner-tree-demo](https://github.com/sebastian-hanisch/steiner-tree-demo), [pcst-demo](https://github.com/sebastian-hanisch/pcst-demo) (Prize-Collecting Steiner-Baum), [mst-sensitivity-demo](https://github.com/sebastian-hanisch/mst-sensitivity-demo) (Sensitivität), [random-spanning-tree-demo](https://github.com/sebastian-hanisch/random-spanning-tree-demo) (zufällige Spannbäume)."
)

with st.expander("Die drei Nebenbedingungen", expanded=True):
    st.markdown(
        """
1. **Bottleneck:** die längste Kante des Baums soll möglichst klein sein. **Der MST erreicht das bereits** (Kruskal nimmt die Kanten nach Länge; ab der Länge b*, bei der der Graph zusammenhängt, ist der Baum fertig). Aber jeder
   Spannbaum aus Kanten ≤ b* hat denselben Bottleneck - das sind meist gewaltig viele, und der MST ist der billigste.
2. **Knotengrad Δ:** jeder Knoten hat höchstens Δ Anschlüsse. Kruskal/Prim mit Gradgrenze können **scheitern** (sie enden im Wald oder stecken fest, obwohl ein Baum existiert). Besser: **Lagrange** - jeder Knoten bekommt eine Strafe π, der MST
   auf den Kosten c + π_u + π_v ist eine **Untergrenze**; die Strafen werden per Subgradient angepasst - und ein **exaktes** Branch-and-Bound mit derselben Schranke.
3. **Hops H:** kein Knoten liegt mehr als H Kanten vom Depot entfernt. H = 1 ist der Stern, großes H der MST. Prim mit Tiefengrenze kann steckenbleiben; der **Schichtenbaum** (Breitensuche) scheitert nie, wenn überhaupt ein Baum existiert;
   beide werden per **Kantentausch** verbessert. Exakt geht das Branch-and-Bound nur für sehr kleine Instanzen.
        """
    )

if C.PRESETS:
    st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
    preset_names = list(C.PRESETS.keys())
    for row in (preset_names[:4], preset_names[4:]):
        if not row:
            continue
        cols = st.columns(len(row))
        for col, name in zip(cols, row):
            with col:
                st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

ss = st.session_state
with st.sidebar:
    st.header("⚙️ Einstellungen")
    kind = st.radio("Instanz", options=list(C.KINDS), format_func=lambda v: C.KIND_LABELS[v], key="kind_select",
                    help="Ortschaften: Verteiler mit Anschlussnehmern im Kreis - dort hat der MST von Natur aus Knoten hohen Grades. Lehrbuchbeispiel: fünf Punkte im Kreuz, von Hand nachzurechnen.")
    constraint = st.radio("Nebenbedingung", options=list(C.CONSTRAINTS), format_func=lambda v: C.CONSTRAINT_LABELS[v], key="constraint_select",
                          help="Bottleneck: der MST ist optimal. Grad und Hops: NP-schwer, der MST verletzt die Grenze oft.")
    if constraint == "degree":
        delta = st.select_slider("Gradgrenze Δ", options=list(range(C.DELTA_MIN, C.DELTA_MAX + 1)), value=int(ss["delta_select"]), key="delta_widget", on_change=store_from_widget, args=("delta_select",),
                                 help="Höchstens Δ Anschlüsse je Knoten. Δ = 2 verlangt einen Pfad (Hamiltonpfad, NP-schwer) und kostet auf 20 Filialen +7.29 %; ab Δ = 3 ist es auf gleichverteilten Instanzen meist gratis.")
    else:
        delta = C.DEFAULT_DELTA
    if constraint == "hops":
        hops = st.select_slider("Hop-Grenze H", options=list(range(C.HOPS_MIN, C.HOPS_MAX + 1)), value=int(ss["hops_select"]), key="hops_widget", on_change=store_from_widget, args=("hops_select",),
                                help="Kein Knoten liegt mehr als H Kanten vom Depot. H = 1 ist der Stern (braucht eine Kante Depot-Knoten für alle: k = 1000). Der MST des Standardfalls (12 Filialen, Seed 35) ist 9 Hops tief.")
    else:
        hops = C.DEFAULT_HOPS
    if kind != "textbook":
        n = st.slider("Filialen n", *bounds("n_slider"), value=int(ss["n_slider"]), key="n_widget", on_change=store_from_widget, args=("n_slider",),
                      help=f"Das exakte Verfahren wird angeboten bis n = {C.N_EXACT['degree']} (Grad) bzw. n = {C.N_EXACT['hops']} (Hops); größere Instanzen zeigen nur die Heuristiken und die Untergrenze.")
        k = st.select_slider("Kandidaten: nächste Nachbarn k", options=list(C.K_OPTIONS), value=int(ss["k_select"]), key="k_widget", on_change=store_from_widget, args=("k_select",),
                             help="Je Knoten die k nächsten Nachbarn als Kandidaten. Bei kleinem k kann es gar keinen Baum mit Grenze geben (z. B. H = 1 braucht k = 1000).")
        terrain = st.select_slider("Geländezuschlag", options=list(C.TERRAIN_OPTIONS), value=float(ss["terrain_select"]), key="terrain_widget", on_change=store_from_widget, args=("terrain_select",),
                                   format_func=lambda v: "0 (euklidisch)" if v == 0 else f"{v:g}",
                                   help="Kosten = Länge x Geländefaktor in [1, 1 + Zuschlag]. Ohne Zuschlag ist der MST-Grad auf gleichverteilten Instanzen höchstens 4; mit Zuschlag 1.0 kommen Grad 5 und 6 vor.")
        round_costs = st.checkbox("Kosten auf ganze Einheiten runden", value=bool(ss["round_select"]), key="round_widget", on_change=store_from_widget, args=("round_select",),
                                  help="Erzeugt Gleichstände; die Bäume bleiben durch den Schlüssel (Kosten, Kantenindex) eindeutig.")
        if kind == "hubs":
            sats = st.slider("Anschlüsse je Verteiler", *bounds("sats_select"), value=int(ss["sats_select"]), key="sats_widget", on_change=store_from_widget, args=("sats_select",),
                             help="Anschlussnehmer im Kreis um jeden Verteiler. Bei 5 hat der MST Knoten mit Grad 5, bei 6 liegen die Nachbarn im Kreis etwa so weit voneinander wie vom Verteiler, ab 7 näher beieinander.")
        else:
            sats = C.DEFAULT_SATS
        seed = st.number_input("Zufalls-Seed der Instanz", *bounds("seed_input"), value=int(ss["seed_input"]), key="seed_widget", step=1, on_change=store_from_widget, args=("seed_input",))
        st.button("🎲 Neue Instanz generieren", width="stretch", on_click=randomize_seed)
    else:
        n, k, terrain, round_costs, sats, seed = C.DEFAULT_N, C.DEFAULT_K, C.DEFAULT_TERRAIN, False, C.DEFAULT_SATS, C.DEFAULT_SEED

sync_query_params({"kind_select": kind, "n_slider": int(ss["n_slider"]), "k_select": int(ss["k_select"]), "terrain_select": float(ss["terrain_select"]), "round_select": bool(ss["round_select"]),
                   "sats_select": int(ss["sats_select"]), "seed_input": int(ss["seed_input"]), "constraint_select": constraint, "delta_select": int(ss["delta_select"]), "hops_select": int(ss["hops_select"]),
                   "tree_select": ss["tree_select"]})

settings = Settings(kind, int(n), int(k), float(terrain), bool(round_costs), int(sats), int(seed), constraint, int(delta), int(hops))
with st.spinner("Rechne..."):
    a = _analysis(settings)
inst = a.inst
E = inst.edges
limit = a.limit
names = inst.labels


def label(v):
    return names[v] if names else str(v)


# --- In Aktion ---------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Der beschränkte Baum in Aktion")
STEP_LABELS = {1: "1 · Instanz und MST", 2: "2 · Die Nebenbedingung", 3: "3 · Der beschränkte Baum"}
step = st.select_slider("Schritt", options=list(STEP_LABELS), key="cons_step", format_func=lambda s: STEP_LABELS[s])
mst_tree = a.mst.tree

if step == 1:
    st.markdown(f"**{inst.n} Standorte** (Depot ⭐ + {inst.n - 1} Filialen), **{inst.m} Kandidatenkanten**. Der MST (grün) kostet **{num(a.mst_cost)}**, seine längste Kante ist **{num(a.b_star)}**, sein größter Grad {a.mst_max_degree} "
                f"und er ist bis zu {a.mst_depth} Hops tief.")
    st.plotly_chart(build_instance(inst, mst_tree, delta if constraint == "degree" else None, hops if constraint == "hops" else None), width="stretch", key="s1_map")
    st.caption("Punktgröße wächst mit dem Grad im MST; rote Punkte verletzen die gewählte Grenze (Grad > Δ bzw. Tiefe > H).")
elif step == 2:
    if constraint == "bottleneck":
        order = sorted(range(inst.m), key=lambda i: (E[i][2], i))
        uf = A.UnionFind(inst.n, "full")
        star_j = next(j for j, i in enumerate(order, 1) if uf.union(E[i][0], E[i][1]) and uf.components == 1)
        if "cons_j" in ss:
            ss["cons_j"] = min(max(1, int(ss["cons_j"])), inst.m)
        elif star_j is not None:
            ss["cons_j"] = star_j
        j = st.slider("Die j billigsten Kanten", 1, inst.m, key="cons_j", help="Kanten in Kruskal-Reihenfolge; die orange Kante ist die zuletzt hinzugenommene.") if inst.m > 1 else 1
        uf2 = A.UnionFind(inst.n, "full")
        for i in order[:j]:
            uf2.union(E[i][0], E[i][1])
        comp = [uf2.find(v) for v in range(inst.n)]
        b_j = E[order[j - 1]][2]
        if uf2.components == 1:
            note = f"Der Graph aus allen Kanten ≤ {num(b_j)} **hängt zusammen**" + (" - erstmals bei j = " + str(star_j) + " (Schwelle b* = " + num(a.b_star) + ")." if j == star_j else f" (schon ab j = {star_j}, b* = {num(a.b_star)}).")
        else:
            note = f"Bei Schwelle {num(b_j)} zerfällt der Graph noch in **{uf2.components} Komponenten** - ein Spannbaum mit dieser längsten Kante ist unmöglich."
        st.markdown(f"**j = {j} von {inst.m}:** {note}")
        st.plotly_chart(build_threshold(inst, order, j, comp), width="stretch", key=f"s2_thr_{j}")
        st.markdown(f"**Der MST ist bottleneck-optimal:** kleiner als b* = {num(a.b_star)} geht die längste Kante nicht (der Graph hängt dann nicht zusammen). Aber **jeder** Spannbaum des Graphen aus den {a.threshold_edges} Kanten ≤ b* hat denselben Bottleneck - "
                    f"das sind etwa **10^{a.count_log10:.1f}** Bäume (Matrix-Baum-Satz von Kirchhoff); der MST ist der billigste von ihnen.")
    elif constraint == "degree":
        viol = [v for v, d in enumerate(A.degrees(inst.n, E, mst_tree)) if d > delta]
        if viol:
            st.markdown(f"**Der MST verletzt die Gradgrenze Δ = {delta}:** {len(viol)} Knoten haben mehr Anschlüsse (rot), höchster Grad {a.mst_max_degree}. Der Baum muss umgebaut werden.")
        else:
            st.markdown(f"**Der MST erfüllt die Gradgrenze Δ = {delta} schon** (größter Grad {a.mst_max_degree}): die Grenze kostet nichts.")
        st.plotly_chart(build_degree_view(inst, mst_tree, delta), width="stretch", key="s2_deg")
        st.caption("Zahl am Knoten = Grad im MST; rot: über Δ, orange: genau Δ.")
        lg = a.lagrange
        if lg is not None and lg.history:
            ub_txt = f"beste zulässige Lösung {num(lg.ub)}" if lg.tree else "noch keine zulässige Lösung"
            st.markdown(f"**Lagrange-Untergrenze:** jeder Knoten bekommt eine Strafe π; der MST auf c + π_u + π_v ergibt nach {lg.iterations} Iterationen die Untergrenze **{num(lg.lb)}** (MST ohne Strafen: {num(a.mst_cost)}), {ub_txt}."
                        + (" Untergrenze = zulässige Lösung: **optimal bewiesen**." if lg.optimal else ""))
            st.plotly_chart(build_lagrange(lg.history, a.mst_cost), width="stretch", key="s2_lag")
    else:
        dep = A.depths_from(inst.n, E, mst_tree)
        over = sum(1 for d in dep if d > hops)
        if over:
            st.markdown(f"**Der MST verletzt die Hop-Grenze H = {hops}:** er ist bis zu {a.mst_depth} Hops tief, {over} Knoten liegen tiefer als erlaubt (rot). Kleinste überhaupt mögliche Tiefe: {a.min_hops} Hops." if a.min_hops >= 0
                        else f"**Der Depotknoten erreicht nicht alle Standorte** über die Kandidatenkanten - keine Hop-Grenze ist erfüllbar.")
        else:
            st.markdown(f"**Der MST erfüllt die Hop-Grenze H = {hops} schon** (Tiefe {a.mst_depth}).")
        st.plotly_chart(build_hop_view(inst, mst_tree, hops), width="stretch", key="s2_hop")
        um = detour(inst, mst_tree)
        st.caption(f"Zahl am Knoten = Hops vom Depot im MST. Größter Umweg im MST: Faktor {um[1]:.2f} (Weglänge im Baum / Luftlinie, Median {um[0]:.2f}).")
else:
    options = C.TREE_OPTIONS[constraint]
    if ss.get("tree_widget") not in options:
        ss.pop("tree_widget", None)
    cur = ss["tree_select"] if ss["tree_select"] in options else options[0]
    tree_key = st.radio("Baum zeigen", options=list(options), format_func=lambda v: C.TREE_LABELS[v], key="tree_widget", horizontal=True, index=list(options).index(cur), on_change=store_from_widget, args=("tree_select",),
                        help="Grün: wie der MST, orange: weicht ab, grau gestrichelt: MST-Kante fehlt. Bester Fund = billigster gültiger Baum unter Greedy, Lagrange und exakt.")
    name = a.best_name if tree_key == "best" else tree_key
    found = a.trees.get(name) if name else None
    if constraint == "bottleneck":
        f = a.trees.get(tree_key)
        st.plotly_chart(build_tree(inst, f.tree, mst_tree, True), width="stretch", key=f"s3_map_{tree_key}")
        st.markdown(f"**{METHOD_LABELS[{'mst': 'min'}.get(tree_key, tree_key)]}:** längste Kante {num(A.bottleneck(E, f.tree))} (= b*), Kosten **{num(f.cost)}**, {pct(a.gap(tree_key))} gegen den MST.")
    elif found is None or not found.feasible:
        if name is None or a.infeasible:
            why = "Es gibt **keinen** gültigen Baum" + (" (bewiesen)." if a.infeasible else ".") if a.infeasible else "Kein Verfahren fand einen gültigen Baum."
        else:
            why = f"{C.TREE_LABELS.get(tree_key, tree_key)} findet hier **keinen** gültigen Baum."
        st.warning(why + (" Lagrange und das exakte Verfahren sind teils erfolgreich, wo Greedy scheitert." if constraint == "degree" and not a.infeasible else ""))
        st.plotly_chart(build_tree(inst, [], mst_tree, False), width="stretch", key="s3_none")
    else:
        st.plotly_chart(build_tree(inst, found.tree, mst_tree, True), width="stretch", key=f"s3_map_{name}")
        ndiff = len([i for i in found.tree if i not in set(mst_tree)])
        extra = f" ({found.note})" if found.note else ""
        st.markdown(f"**{C.TREE_LABELS[name]}{extra}:** Kosten **{num(found.cost)}**, **{pct(a.gap(name))}** gegen den MST ({num(a.mst_cost)}); {ndiff} Kanten weichen ab. Größter Grad {max(A.degrees(inst.n, E, found.tree))}, "
                    f"Tiefe {max(A.depths_from(inst.n, E, found.tree))}, längste Kante {num(A.bottleneck(E, found.tree))} (MST: {num(a.b_star)}).")
    order_bars = ("mst", "min", "max", "random") if constraint == "bottleneck" else (("mst", "greedy", "lagrange", "exact") if constraint == "degree" else ("mst", "greedy", "exact"))
    costs = {k: (a.trees[k].cost if k in a.trees and a.trees[k].feasible else None) for k in order_bars if k in a.trees or k == "mst"}
    if constraint == "bottleneck":
        costs.pop("min", None)
    shown = [k for k in order_bars if k in costs]
    st.plotly_chart(build_cost_bars(costs, a.mst_cost, shown), width="stretch", key="cost_bars")
    st.caption("Mehrkosten gegen den MST. Ein fehlender Balken heißt: das Verfahren liefert keinen gültigen Baum bzw. wird bei dieser Größe nicht angeboten.")

st.markdown("---")

# --- Kennzahlen --------------------------------------------------------------------------------------------------------------------------------

st.markdown("## ⚙️ Was kostet die Nebenbedingung?")
if constraint == "degree":
    ex_txt = "zu groß" if not a.exact_offered else ("bewiesen" if a.proved else "nicht bewiesen")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Preis", pct(a.gap(a.best_name)) if a.best_name else "kein Baum", delta=f"Fund: {C.TREE_LABELS.get(a.best_name, '-')}" if a.best_name else None, delta_color="off")
    m2.metric("Untergrenze", pct(a.lb_gap), delta="Lagrange", delta_color="off")
    m3.metric("Greedy", pct(a.gap("greedy")) if a.trees["greedy"].feasible else "scheitert", delta="gegen MST", delta_color="off")
    m4.metric("Exakt", pct(a.gap("exact")) if a.exact_offered and "exact" in a.trees and a.trees["exact"].feasible else "-", delta=ex_txt, delta_color="off")
    st.caption(f"Preis = billigster gültiger Baum gegen den MST ({num(a.mst_cost)}); Untergrenze = Lagrange-Untergrenze (für jedes π gültig). "
               + (f"Das Branch-and-Bound brauchte {a.exact.nodes} Suchknoten." if a.exact is not None else "Bei dieser Größe wird das exakte Verfahren nicht angeboten."))
elif constraint == "hops":
    ex_txt = "zu groß" if not a.exact_offered else ("bewiesen" if a.proved else "nicht bewiesen")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Preis", pct(a.gap(a.best_name)) if a.best_name else "kein Baum", delta=f"Fund: {C.TREE_LABELS.get(a.best_name, '-')}" if a.best_name else None, delta_color="off")
    m2.metric("Tiefe des MST", str(a.mst_depth), delta=f"Grenze {hops}", delta_color="off")
    m3.metric("Greedy", pct(a.gap("greedy")) if a.trees["greedy"].feasible else "scheitert", delta="gegen MST", delta_color="off")
    m4.metric("Exakt", pct(a.gap("exact")) if a.exact_offered and "exact" in a.trees and a.trees["exact"].feasible else "-", delta=ex_txt, delta_color="off")
    um_b = detour(inst, a.trees[a.best_name].tree)[1] if a.best_name else None
    st.caption(f"Preis = billigster gültiger Baum gegen den MST ({num(a.mst_cost)}). Größter Umweg: MST {detour(inst, mst_tree)[1]:.2f}" + (f", beschränkter Baum {um_b:.2f}" if um_b else "")
               + f". Die Hop-Untergrenze ist hier nur der MST selbst (kein Lagrange für Hops). Kleinste mögliche Tiefe: {a.min_hops}.")
else:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Bottleneck b*", num(a.b_star), delta="längste MST-Kante", delta_color="off")
    m2.metric("Optimale Bäume", f"10^{a.count_log10:.1f}", delta=f"{a.threshold_edges} Kanten ≤ b*", delta_color="off")
    m3.metric("Teuerster", pct(a.gap("max")), delta="gegen MST", delta_color="off")
    m4.metric("Zufälliger", pct(a.gap("random")), delta="gegen MST", delta_color="off")
    st.caption("Alle Bäume dieser Tabelle haben denselben, kleinstmöglichen Bottleneck b*; sie unterscheiden sich nur in der Summe.")

st.markdown("---")

# --- Experimente auf Abruf ---------------------------------------------------------------------------------------------------------------------

base = replace(settings, seed=0)
if kind != "textbook":
    if constraint in ("degree", "hops"):
        st.subheader("🎲 Wie oft scheitern die Greedy-Verfahren?")
        st.caption("50 Instanzen mit den Einstellungen der Seitenleiste (nur der Seed wechselt): wie oft gibt es einen gültigen Baum, und wie oft findet ihn der Greedy nicht?")
        if st.button("Machbarkeits-Experiment über 50 Instanzen", key="feas_start"):
            ss["feas_done"] = ss.get("feas_done", set()) | {base}
        if base in ss.get("feas_done", set()):
            with st.spinner("Rechne..."):
                res = _feasibility(base)
            f1, f2, f3, f4 = st.columns(4)
            f1.metric("Baum gefunden", f"{res['found_share']:.0f} %", delta=f"unmöglich: {res['infeasible_share']:.0f} %", delta_color="off")
            f2.metric("MST verletzt Grenze", f"{res['mst_violates']:.0f} %", delta="der Instanzen", delta_color="off")
            if constraint == "degree":
                f3.metric("Greedy scheitert", f"{res['greedy_fail']:.0f} %", delta=f"Lagrange rettet {res['lagrange_rescue']:.0f} %", delta_color="off")
                f4.metric("Lagrange scheitert", f"{res['lagrange_fail']:.0f} %", delta=f"unklar: {res['unknown_share']:.0f} %", delta_color="off")
            else:
                f3.metric("Prim scheitert", f"{res['prim_fail']:.0f} %", delta="mit Tiefengrenze", delta_color="off")
                f4.metric("Schichtenbaum scheitert", f"{res['layer_fail']:.0f} %", delta="wenn ein Baum existiert", delta_color="off")
            st.caption(f"Über {res['n_runs']} Instanzen. \"Scheitert\" zählt nur Instanzen, in denen ein anderes Verfahren einen gültigen Baum fand.")
        st.markdown("---")

    if constraint == "bottleneck":
        st.subheader("♾️ Wie viele bottleneck-optimale Bäume gibt es?")
        st.caption("Über n mit den Einstellungen der Seitenleiste (5 feste Instanzen je n): Zahl der Bäume (Kirchhoff) und Mehrkosten des teuersten und eines zufälligen gegen den MST.")
        if st.button("Zahl der bottleneck-optimalen Bäume berechnen", key="bn_start"):
            ss["bn_done"] = ss.get("bn_done", set()) | {base}
        if base in ss.get("bn_done", set()):
            with st.spinner("Rechne..."):
                rows_b = _bottleneck_counts(base)
            st.plotly_chart(build_bottleneck_counts(rows_b), width="stretch", key="bn_counts")
            st.caption("n = " + ", ".join(str(r["n"]) for r in rows_b) + ": 10^" + ", 10^".join(f"{r['log10_count']:.1f}" for r in rows_b) + " Bäume (Median).")
        st.markdown("---")

    st.subheader("📐 Sweeps")
    params = sweep_params(constraint, kind)
    if ss.get("sweep_select") not in params:
        ss.pop("sweep_select", None)
    sweep_param = st.selectbox("Welcher Regler soll durchgefahren werden?", params, format_func=lambda v: SWEEP_LABELS[v], key="sweep_select")
    metric_opts = {"degree": {"gap": "Preis gegen den MST", "fail": "Ausfall der Greedy-Verfahren", "struct": "Größter Grad"},
                   "hops": {"gap": "Preis gegen den MST", "fail": "Ausfall / unmöglich", "struct": "Größter Umweg"},
                   "bottleneck": {"count": "Zahl der Bäume (log10)", "gap": "Mehrkosten teuerster/zufälliger"}}[constraint]
    if ss.get("sweep_metric") not in metric_opts:
        ss.pop("sweep_metric", None)
    metric = st.radio("Kennzahl", options=list(metric_opts), format_func=lambda v: metric_opts[v], key="sweep_metric", horizontal=True)
    if st.button("Sweep über 5 feste Instanzen berechnen (kann einige Sekunden dauern)", key="sweep_start"):
        ss["sweep_done"] = ss.get("sweep_done", set()) | {(sweep_param, base)}
    if (sweep_param, base) in ss.get("sweep_done", set()):
        with st.spinner("Rechne den Sweep über 5 feste Instanzen..."):
            rows_s = _sweep(sweep_param, base)
        lab = SWEEP_LABELS[sweep_param]
        series = {
            ("degree", "gap"): ([("gap_best", "bester Fund", "#2F6B65"), ("gap_greedy", "Greedy", "#4c78a8"), ("gap_lb", "Lagrange-Untergrenze", "#e8a13a")], "Mehrkosten gegen den MST (%)"),
            ("degree", "fail"): ([("fail_greedy", "Greedy", "#4c78a8"), ("fail_lagrange", "Lagrange", "#e8a13a")], "Instanzen ohne Baum (%)"),
            ("degree", "struct"): ([("mst_maxdeg", "MST", "#7b3fbf"), ("deg_best", "beschränkter Baum", "#2F6B65")], "Größter Grad"),
            ("hops", "gap"): ([("gap_best", "bester Fund", "#2F6B65"), ("gap_greedy", "Greedy", "#4c78a8"), ("gap_exact_proved", "exakt (bewiesen)", "#e8a13a")], "Mehrkosten gegen den MST (%)"),
            ("hops", "fail"): ([("fail_prim", "Prim mit Tiefengrenze", "#4c78a8"), ("infeasible_share", "kein Baum möglich", "#d62728")], "Instanzen (%)"),
            ("hops", "struct"): ([("detour_mst", "MST", "#7b3fbf"), ("detour_best", "beschränkter Baum", "#2F6B65")], "Größter Umweg (Faktor)"),
            ("bottleneck", "count"): ([("log10_count", "log10 Zahl der Bäume", "#2F6B65")], "log10 der Anzahl"),
            ("bottleneck", "gap"): ([("gap_max", "teuerster", "#d62728"), ("gap_random", "zufälliger", "#e8a13a")], "Mehrkosten gegen den MST (%)"),
        }[(constraint, metric)]
        st.plotly_chart(build_sweep(rows_s, lab, series[0], series[1]), width="stretch", key="sweep_chart")
        st.caption("Median über 5 feste Instanzen (Seeds 100000–100004), Band = 10. bis 90. Perzentil; Preise nur über Instanzen mit gültigem Baum, das exakte Verfahren nur, wo es innerhalb der Suchknoten fertig wurde. Die übrigen Regler stehen wie in der Seitenleiste.")
    st.markdown("---")

    st.subheader("📏 Wie groß wird der Grad des MST?")
    st.caption("Auf euklidischen Instanzen ist der MST-Grad durch eine geometrische Tatsache beschränkt (höchstens 6, in Allgemeinlage höchstens 5). 300 dichte Instanzen mit n = 30 Filialen:")
    if st.button("Grad-Verteilung über 300 Instanzen berechnen", key="prof_start"):
        ss["prof_done"] = True
    if ss.get("prof_done"):
        with st.spinner("Rechne..."):
            p0, p1, ph = _profile("depot", 0.0, 5), _profile("depot", 1.0, 5), _profile("hubs", 0.0, 5)
        fmt = lambda p: ", ".join(f"Grad {d}: {c}" for d, c in p.items())
        st.markdown(f"- gleichverteilt, ohne Geländezuschlag: {fmt(p0)}\n- gleichverteilt, Geländezuschlag 1.0: {fmt(p1)}\n- Ortschaften (5 Anschlüsse), ohne Zuschlag: {fmt(ph)}")
    st.markdown("---")

# --- Grenzen -----------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Der MST reicht** | Nur für den Bottleneck. Grad Δ = 2 kostet auf 20 gleichverteilten Filialen im Median +7.29 %, Δ = 3 nichts; in Ortschaften kostet Δ = 3 +1.06 %, Δ = 4 +0.23 %. Die Hop-Grenze H = 3 kostet auf 12 Filialen +15.06 %, H = 6 +1.40 %. | - |
| **Kruskal/Prim mit Grenze finden einen Baum** | Nein. Bei Δ = 2 scheitern sie in 20 % der Instanzen (Ortschaften: 60 %), obwohl ein Baum existiert; Lagrange scheitert dort nie. Prim mit Tiefengrenze scheitert bei H = 4 auf 20 Filialen in 80 % der Instanzen. | - |
| **Greedy ist nah am Optimum** | Nur bei lockerer Grenze. Bei H = 4 auf 12 Filialen liegt Greedy +9.4 % über dem MST, der exakte Wert nur +6.49 %; die Lokalsuche bleibt in Nachbarschaften stecken. | Stärkere Metaheuristiken (nicht gebaut) |
| **Exakt lösbar** | Nur klein: das Branch-and-Bound wird bis n = 24 (Grad) bzw. n = 12 (Hops) angeboten und ist bei enger Hop-Grenze (H = 2) auch dort nicht immer fertig. Große Instanzen brauchen Branch-and-Cut (nicht gebaut). | Layered-Graph-/Schnittformulierungen (Gouveia u. a., nicht gebaut) |
| **Für Hops gibt es eine Untergrenze** | Hier nur der MST selbst - viel zu schwach; die Lücke der Hop-Heuristik ist ab n = 13 nicht mehr messbar. | Lagrange/LP für Hops (nicht gebaut) |
| **Euklidisch** | Der Satz "MST-Grad ≤ 6" gilt nur ohne Geländezuschlag; mit Zuschlag 1.0 kamen Grad 5 und 6 vor, in Ortschaften mit 5 Anschlüssen liegt der MST-Grad fast immer bei 5. | - |
| **Synthetisches Modell** | Punkte im Quadrat, k nächste Nachbarn, ein Depot; keine Kapazitäten (Folgestück Kapazitierter MST), keine echten Netze. Approximationsalgorithmen für Grad- und Längenbeschränkung (Fürer-Raghavachari, Haeupler u. a.) sind nicht gebaut. | Echte Netze, Approximationsverfahren |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Bottleneck.** Gesucht ist ein Spannbaum $T$ mit minimalem $\max_{e \in T} c_e$. Sei $b^*$ die kleinste Schwelle, für die der Graph aus allen Kanten mit $c_e \le b^*$ zusammenhängt. Jeder MST hat Bottleneck $b^*$ (Kruskal
nimmt die Kanten nach Länge und ist bei $b^*$ fertig), und die bottleneck-optimalen Bäume sind genau die Spannbäume des Graphen $G_{\le b^*}$. Deren Zahl liefert der Matrix-Baum-Satz: die Determinante der Laplace-Matrix ohne eine
Zeile und Spalte.

**Knotengrad.** $\min \sum_{e \in T} c_e$ unter $\deg_T(v) \le \Delta$ für alle $v$. **Lagrange-Relaxation** der Gradgrenzen mit $\pi \ge 0$:
$L(\pi) = \min_T \sum_{uv \in T}(c_{uv} + \pi_u + \pi_v) - \Delta \sum_v \pi_v$ - das Minimum ist ein MST auf geänderten Kosten. $L(\pi)$ ist für jedes $\pi$ eine Untergrenze der optimalen Kosten und in $\pi$ konkav; das
Subgradientenverfahren mit Subgradient $\deg_T(v) - \Delta$ maximiert sie. Ist der MST zu $\pi$ zulässig und gilt komplementärer Schlupf, ist er optimal.

**Hops.** Kein Knoten liegt mehr als $H$ Kanten vom Depot: $\mathrm{depth}_T(v) \le H$. Der Schichtenbaum (Breitensuche) hat die kleinstmögliche Tiefe; er ist also zulässig genau dann, wenn irgendein zulässiger Baum existiert.

**Komplexität.** Der Gradbeschränkte Spannbaum ist für jedes $\Delta \ge 2$ NP-vollständig (Garey & Johnson 1979); für $\Delta = 2$ ist er das Hamiltonpfad-Problem. Der hop-beschränkte Spannbaum ist für allgemeine Kosten NP-schwer (schon für $H = 2$),
für euklidische Instanzen wird das hier nicht behauptet.

**Literatur.** Camerini, P. M. (1978). *The min-max spanning tree problem and some extensions.* Information Processing Letters 7(1), 10-14. Gabow, H. N., & Tarjan, R. E. (1988). *Algorithms for two bottleneck optimization problems.*
Journal of Algorithms 9(3), 411-417. Volgenant, A. (1989). *A Lagrangean approach to the degree-constrained minimum spanning tree problem.* European Journal of Operational Research 39(3), 325-331. Garey, M. R., & Johnson, D. S.
(1979). *Computers and Intractability.* Freeman. Gouveia, L. und Mitarbeiter: Modellierung des hop-beschränkten MST über geschichtete Graphen (Mathematical Programming). Zur längenbeschränkten Variante: *Simple Length-Constrained
Minimum Spanning Trees* (arXiv 2410.08170, 2024).

Implementiert in `cons_algorithm.py` (Verfahren), `cons_scenario.py` (Instanzen), `cons_evaluation.py` (Kennzahlen, Sweeps, Experimente).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Spannbäume: vom Kruskal bis zum Zufallsbaum](https://sebastianhanisch.net/konzepte-spannbaum.html)."
)
