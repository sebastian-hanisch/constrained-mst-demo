"""Presets: gültige Einstellungen, Kennzahl-Bänder über die 5 festen Instanzen, Permalink-Spezifikation."""

from dataclasses import replace

import pytest

import cons_constants as C
import cons_evaluation as ev
import cons_presets as P

KEYS = {"kind", "n", "k", "terrain", "round_costs", "sats", "seed", "constraint", "delta", "hops", "tree"}


def _settings(p):
    return ev.Settings(p["kind"], p["n"], p["k"], p["terrain"], p["round_costs"], p["sats"], p["seed"], p["constraint"], p["delta"], p["hops"])


def test_presets_have_help_and_full_settings():
    assert len(C.PRESETS) == 8 and set(C.PRESET_HELP) == set(C.PRESETS)
    for name, p in C.PRESETS.items():
        assert set(p) == KEYS, name
        assert p["kind"] in C.KINDS and p["constraint"] in C.CONSTRAINTS and p["tree"] in C.ALL_TREES
        assert C.N_MIN <= p["n"] <= C.N_MAX and p["k"] in C.K_OPTIONS and p["terrain"] in C.TERRAIN_OPTIONS
        assert C.DELTA_MIN <= p["delta"] <= C.DELTA_MAX and C.HOPS_MIN <= p["hops"] <= C.HOPS_MAX and C.SATS_MIN <= p["sats"] <= C.SATS_MAX
        assert p["tree"] in C.TREE_OPTIONS[p["constraint"]], name
        assert p["seed"] <= C.SEED_MAX
        assert len(C.PRESET_HELP[name]) > 40


def test_default_preset_is_the_default_setting():
    assert _settings(C.PRESETS["Standardfall (Voreinstellung)"]) == ev.Settings()
    for key, state_key in P.PRESET_KEYS.items():
        assert P.SETTING_SPECS[state_key].default == C.PRESETS["Standardfall (Voreinstellung)"][key]


@pytest.mark.parametrize("name", list(C.PRESET_EXPECTED_BANDS))
def test_preset_key_metric_lies_in_its_band(name):
    metric, lo, hi = C.PRESET_EXPECTED_BANDS[name]
    r = ev.run_config(_settings(C.PRESETS[name]))
    assert lo <= r[metric] <= hi, (name, metric, r[metric])


def test_every_preset_analyses_without_error_and_finds_what_its_help_promises():
    for name, p in C.PRESETS.items():
        a = ev.analyse(_settings(p))
        if p["constraint"] != "bottleneck":
            assert a.best_name is not None, name
    assert not ev.analyse(_settings(C.PRESETS["Greedy scheitert (Grad ≤ 2)"])).trees["greedy"].feasible
    assert ev.analyse(_settings(C.PRESETS["Kostenlose Grenze (Grad ≤ 3)"])).gap("exact") == pytest.approx(0.0, abs=1e-9)
    assert not ev.analyse(_settings(C.PRESETS["Kostenlose Grenze (Grad ≤ 3)"])).mst_violates


def test_setting_specs_cover_all_widget_keys_and_permalink_names_are_unique():
    assert set(P.WIDGET_KEYS) <= set(P.SETTING_SPECS)
    urls = [s.url_param for s in P.SETTING_SPECS.values()]
    assert len(urls) == len(set(urls))
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


@pytest.mark.parametrize("state_key,bad", [("kind_select", "grid"), ("k_select", "7"), ("terrain_select", "0.35"), ("round_select", "yes"), ("constraint_select", "x"), ("tree_select", "nope")])
def test_permalink_casters_reject_invalid_choices(state_key, bad):
    with pytest.raises(ValueError):
        P.SETTING_SPECS[state_key].caster(bad)
    default = P.SETTING_SPECS[state_key].default
    assert P.SETTING_SPECS[state_key].caster(str(default)) == default


def test_permalink_casters_accept_valid_values():
    assert P.SETTING_SPECS["round_select"].caster("true") is True
    assert P.SETTING_SPECS["terrain_select"].caster("0.3") == 0.3
    assert P.SETTING_SPECS["k_select"].caster("1000") == 1000
    assert P.SETTING_SPECS["n_slider"].lo == C.N_MIN and P.SETTING_SPECS["hops_select"].hi == C.HOPS_MAX
