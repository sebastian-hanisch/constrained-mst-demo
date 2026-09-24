"""AppTest-Rauchtests: Voreinstellung, jedes Preset, jeder Schritt in jeder Nebenbedingung, alle Instanztypen und Bäume, Randwerte, Würfel-Knopf, Permalink-Grenzen, Instanzwechsel, Experimente und Sweeps auf Abruf,
Footer."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import cons_constants as C

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _run(step=1, **state):
    at = AppTest.from_file(APP, default_timeout=240)
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    if step != 1:
        at.select_slider(key="cons_step").set_value(step).run()
    return at


def _ok(at):
    assert not at.exception, [e.value for e in at.exception]


def _metric(at, label):
    return next(m for m in at.metric if m.label.startswith(label))


def test_default_run_has_no_exception_and_shows_the_four_metrics():
    at = _run()
    _ok(at)
    assert {"Preis", "Untergrenze", "Greedy", "Exakt"} <= {m.label for m in at.metric}
    assert _metric(at, "Preis").value == "+4.04 %" and _metric(at, "Untergrenze").value == "+4.04 %" and _metric(at, "Greedy").value == "+4.04 %" and _metric(at, "Exakt").delta == "bewiesen"


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_button_runs(name):
    at = _run()
    next(b for b in at.button if b.key == f"preset_{name}").click().run()
    _ok(at)
    p = C.PRESETS[name]
    ss = at.session_state
    assert (ss["kind_select"], ss["constraint_select"], ss["delta_select"], ss["hops_select"], ss["n_slider"], ss["k_select"], ss["seed_input"]) == (p["kind"], p["constraint"], p["delta"], p["hops"], p["n"], p["k"], p["seed"])
    assert at.metric and at.get("plotly_chart")


@pytest.mark.parametrize("constraint", C.CONSTRAINTS)
@pytest.mark.parametrize("step", [1, 2, 3])
def test_every_step_runs_for_every_constraint_and_kind(constraint, step):
    for kind in C.KINDS:
        at = _run(kind_select=kind, constraint_select=constraint, n_slider=10, k_select=1000 if constraint == "hops" else 6, cons_step=step)
        _ok(at)
        assert at.get("plotly_chart") and at.session_state["cons_step"] == step


@pytest.mark.parametrize("constraint", C.CONSTRAINTS)
def test_every_tree_view_runs(constraint):
    for tree in C.TREE_OPTIONS[constraint]:
        at = _run(step=3, constraint_select=constraint, tree_select=tree, n_slider=10, k_select=1000)
        _ok(at)
        assert at.get("plotly_chart") and any("Kosten" in m.value for m in at.markdown)


def test_failed_greedy_shows_a_warning_instead_of_a_tree():
    at = _run(step=3, seed_input=100001, tree_select="greedy")
    _ok(at)
    assert any("findet hier **keinen** gültigen Baum" in w.value for w in at.warning)
    at2 = _run(step=3, seed_input=100001, tree_select="exact")
    _ok(at2)
    assert not at2.warning


def test_infeasible_hop_limit_shows_a_warning():
    at = _run(step=3, constraint_select="hops", hops_select=1, k_select=6)
    _ok(at)
    assert any("keinen" in w.value for w in at.warning)


def test_bottleneck_threshold_slider_walks_through_the_edges():
    at = _run(step=2, constraint_select="bottleneck")
    _ok(at)
    m = int(at.slider(key="cons_j").max)
    assert m == 45
    for j in (1, 20, m):
        at.slider(key="cons_j").set_value(j).run()
        _ok(at)
        assert any(x.value.startswith(f"**j = {j} von {m}:**") for x in at.markdown)
    assert any("hängt zusammen" in x.value for x in at.markdown)
    at.slider(key="cons_j").set_value(3).run()
    assert any("Komponenten" in x.value for x in at.markdown)
    at.session_state["kind_select"] = "textbook"
    at.run()
    _ok(at)
    assert at.session_state["cons_j"] <= 10


@pytest.mark.parametrize("kw", [
    dict(n_slider=C.N_MIN), dict(n_slider=C.N_MAX), dict(n_slider=C.N_MAX, constraint_select="hops", hops_select=C.HOPS_MAX), dict(k_select=C.K_OPTIONS[0]), dict(k_select=C.K_OPTIONS[-1]),
    dict(delta_select=C.DELTA_MAX), dict(hops_select=C.HOPS_MIN, constraint_select="hops", k_select=1000), dict(terrain_select=C.TERRAIN_OPTIONS[-1]), dict(round_select=True), dict(kind_select="hubs", sats_select=C.SATS_MAX),
    dict(kind_select="hubs", sats_select=C.SATS_MIN, delta_select=2), dict(n_slider=C.N_MIN, k_select=3, constraint_select="hops", hops_select=1),
])
def test_extreme_settings_run(kw):
    for step in (1, 2, 3):
        _ok(_run(step=step, **kw))


def test_dice_button_changes_the_seed():
    at = _run()
    old = at.session_state["seed_input"]
    next(b for b in at.button if b.label == "🎲 Neue Instanz generieren").click().run()
    _ok(at)
    assert at.session_state["seed_input"] != old


def test_permalink_values_are_clamped_and_invalid_choices_fall_back_to_the_default():
    at = AppTest.from_file(APP, default_timeout=240)
    for k, v in dict(n="9999", k="7", terrain="0.35", round="maybe", sats="99", delta="9", hops="0", constraint="x", tree="nope", kind="nope").items():
        at.query_params[k] = v
    at.run()
    _ok(at)
    ss = at.session_state
    assert (ss["n_slider"], ss["k_select"], ss["terrain_select"], ss["round_select"]) == (C.N_MAX, C.DEFAULT_K, C.DEFAULT_TERRAIN, False)
    assert (ss["sats_select"], ss["delta_select"], ss["hops_select"]) == (C.SATS_MAX, C.DELTA_MAX, C.HOPS_MIN)
    assert (ss["constraint_select"], ss["tree_select"], ss["kind_select"]) == (C.DEFAULT_CONSTRAINT, C.DEFAULT_TREE, "depot")


def test_permalink_accepts_valid_values():
    at = AppTest.from_file(APP, default_timeout=240)
    for k, v in dict(kind="hubs", n="20", k="10", terrain="0.4", round="true", sats="6", seed="7", constraint="hops", delta="4", hops="5", tree="greedy").items():
        at.query_params[k] = v
    at.run()
    _ok(at)
    ss = at.session_state
    assert (ss["kind_select"], ss["n_slider"], ss["k_select"], ss["terrain_select"], ss["round_select"], ss["sats_select"], ss["seed_input"]) == ("hubs", 20, 10, 0.4, True, 6, 7)
    assert (ss["constraint_select"], ss["delta_select"], ss["hops_select"], ss["tree_select"]) == ("hops", 4, 5, "greedy")


def test_sidebar_shows_only_the_controls_that_matter():
    deg = _run()
    assert any(w.key == "delta_widget" for w in deg.select_slider) and not any(w.key == "hops_widget" for w in deg.select_slider)
    assert any(w.key == "n_widget" for w in deg.slider) and not any(w.key == "sats_widget" for w in deg.slider)
    hops = _run(constraint_select="hops")
    assert any(w.key == "hops_widget" for w in hops.select_slider) and not any(w.key == "delta_widget" for w in hops.select_slider)
    hubs = _run(kind_select="hubs")
    assert any(w.key == "sats_widget" for w in hubs.slider)
    tb = _run(kind_select="textbook")
    assert not any(w.key == "n_widget" for w in tb.slider) and not any(n.key == "seed_widget" for n in tb.number_input) and any(w.key == "delta_widget" for w in tb.select_slider)
    for at in (deg, hops, hubs, tb):
        assert any(r.key == "constraint_select" for r in at.radio) and any(r.key == "kind_select" for r in at.radio)


def test_changing_constraint_while_on_step_three_does_not_crash():
    at = _run(step=3, tree_select="lagrange")
    _ok(at)
    for con in ("hops", "bottleneck", "degree"):
        at.session_state["constraint_select"] = con
        at.run()
        _ok(at)
    at.session_state["kind_select"] = "textbook"
    at.run()
    _ok(at)


def test_bottleneck_metrics_and_labels():
    at = _run(constraint_select="bottleneck", n_slider=20, tree_select="mst")
    _ok(at)
    assert {"Bottleneck b*", "Optimale Bäume", "Teuerster", "Zufälliger"} <= {m.label for m in at.metric}
    assert _metric(at, "Optimale Bäume").value == "10^9.7" and _metric(at, "Teuerster").value == "+74.57 %"


def test_hop_metrics():
    at = _run(constraint_select="hops", hops_select=3)
    _ok(at)
    assert _metric(at, "Preis").value == "+18.42 %" and _metric(at, "Tiefe des MST").value == "9" and _metric(at, "Greedy").value == "+33.91 %" and _metric(at, "Exakt").value == "+18.42 %"


def test_large_instance_offers_no_exact_method():
    at = _run(n_slider=40)
    _ok(at)
    assert _metric(at, "Exakt").value == "-" and _metric(at, "Exakt").delta == "zu groß"


def test_feasibility_experiment_runs_on_demand_for_degree_and_hops():
    at = _run(n_slider=12)
    next(b for b in at.button if b.key == "feas_start").click().run()
    _ok(at)
    assert _metric(at, "Greedy scheitert").value == "6 %" and _metric(at, "Greedy scheitert").delta == "Lagrange rettet 6 %"
    at2 = _run(constraint_select="hops", hops_select=4, n_slider=12)
    next(b for b in at2.button if b.key == "feas_start").click().run()
    _ok(at2)
    assert _metric(at2, "Prim scheitert").value == "4 %" and _metric(at2, "Schichtenbaum scheitert").value == "0 %"


def test_bottleneck_count_experiment_runs_on_demand():
    at = _run(constraint_select="bottleneck")
    next(b for b in at.button if b.key == "bn_start").click().run()
    _ok(at)
    assert any("n = 10, 20, 30, 60: 10^3.2, 10^10.6, 10^14.2, 10^37.3 Bäume" in c.value for c in at.caption)


def test_degree_profile_experiment_runs_on_demand():
    at = _run()
    next(b for b in at.button if b.key == "prof_start").click().run()
    _ok(at)
    assert any("Grad 3: 250, Grad 4: 50" in m.value for m in at.markdown)


@pytest.mark.parametrize("constraint,params", [("degree", ["delta", "n", "k"]), ("hops", ["hops", "n", "k"]), ("bottleneck", ["n", "k"])])
def test_sweeps_run_on_demand_for_every_metric(constraint, params):
    metrics = {"degree": ["gap", "fail", "struct"], "hops": ["gap", "fail", "struct"], "bottleneck": ["count", "gap"]}[constraint]
    for metric in metrics:
        at = _run(n_slider=8, constraint_select=constraint, sweep_metric=metric, k_select=1000 if constraint == "hops" else 6)
        at.selectbox(key="sweep_select").set_value(params[0]).run()
        next(b for b in at.button if b.key == "sweep_start").click().run()
        _ok(at)
        assert at.get("plotly_chart")


def test_villages_offer_the_sats_sweep_and_the_textbook_has_no_experiments():
    at = _run(kind_select="hubs", n_slider=10)
    assert "Anschlüsse je Verteiler" in list(at.selectbox(key="sweep_select").options)
    tb = _run(kind_select="textbook")
    assert not any(b.key in ("feas_start", "sweep_start", "prof_start", "bn_start") for b in tb.button)


def test_footer_limits_and_literature_are_present():
    at = _run()
    assert any("Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net)" in c.value for c in at.caption)
    assert any("Wo die Annahmen enden" in s.value for s in at.subheader)
    assert any("Camerini, P. M. (1978)" in m.value and "Volgenant, A. (1989)" in m.value and "Gabow, H. N., & Tarjan, R. E. (1988)" in m.value for m in at.markdown)
