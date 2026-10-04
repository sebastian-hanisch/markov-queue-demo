"""Presets: Vollständigkeit, gültige Werte, Permalink-Konstanten, Formatierer."""

import pytest

import mkv_constants as C
import mkv_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP) == set(C.PRESET_ORDER) and len(C.PRESETS) == 4
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS) and C.PRESET_HELP[name]


def test_preset_values_are_valid_and_match_the_setting_specs():
    for preset in C.PRESETS.values():
        assert C.C_MIN <= preset["c"] <= C.C_MAX and C.K_MIN <= preset["k"] <= C.K_MAX and preset["k"] >= preset["c"]
        assert C.RHO_PCT_MIN <= preset["rho_pct"] <= C.RHO_PCT_MAX and P.snap_rho(preset["rho_pct"]) == preset["rho_pct"]
        for key, state_key in P.PRESET_KEYS.items():
            P.SETTING_SPECS[state_key].caster(preset[key])


def test_preset_names_state_the_values_they_set():
    assert C.PRESETS["Eine Spur, 6 Plätze"]["c"] == 1 and C.PRESETS["Eine Spur, 6 Plätze"]["k"] == 6
    assert C.PRESETS["Überlast (150 %)"]["rho_pct"] == 150
    assert C.PRESETS["Nur Spuren (K = c)"]["k"] == C.PRESETS["Nur Spuren (K = c)"]["c"]


def test_default_preset_equals_the_default_settings():
    p = C.PRESETS["Zwei Spuren, 6 Plätze"]
    assert (p["c"], p["k"], p["rho_pct"], p["seed"]) == (C.DEFAULT_C, C.DEFAULT_K, C.DEFAULT_RHO_PCT, C.DEFAULT_SEED)


def test_bounds_and_url_params():
    assert P.bounds("seed_input") == (0, C.SEED_MAX) and P.bounds("rho_slider") == (50, 150) and P.bounds("c_slider") == (1, 4) and P.bounds("k_slider") == (4, 10)
    assert len({spec.url_param for spec in P.SETTING_SPECS.values()}) == len(P.SETTING_SPECS)


def test_k_never_falls_below_the_largest_number_of_lanes():
    assert C.K_MIN >= C.C_MAX


@pytest.mark.parametrize("value,expected", [(0, 50), (52, 50), (53, 50), (57, 60), (87, 90), (99, 100), (200, 150)])
def test_load_snaps_to_the_step_inside_the_bounds(value, expected):
    assert P.snap_rho(value) == expected


def test_formatters():
    assert C.fmt_int(150000) == "150.000" and C.fmt_pct(0.2) == "20 %" and C.fmt_pct(0.076, 1) == "7.6 %"
    assert C.fmt_sci(5.46e-07) == "5.5·10⁻⁷" and C.fmt_sci(193000) == "1.9·10⁵" and C.fmt_sci(0) == "0"
