"""AppTest-Rauchtests: Voreinstellung, jedes Preset, Randwerte, Würfel-Knopf, Permalink-Grenzen, Abschnitte, lokale Regler, Footer."""

import random
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import mkv_constants as C

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _run(**state):
    at = AppTest.from_file(APP, default_timeout=600)
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    return at


def _ok(at):
    assert not at.exception, [e.value for e in at.exception]


def _metric(at, label):
    return next(m.value for m in at.metric if m.label == label)


def _table(at, start):
    return next(m.value for m in at.markdown if m.value.startswith(start))


def test_default_run_has_no_exception_and_shows_the_reference_values():
    at = _run()
    _ok(at)
    assert _metric(at, "Verlust (π_K)") == "7.60 %" and _metric(at, "Mittlere Zahl im System") == "2.453"
    assert _metric(at, "Wartezeit der Angenommenen") == "1.98 min" and _metric(at, "Auslastung der Spuren") == "73.9 %"
    assert len(at.get("plotly_chart")) == 6


def test_all_sections_are_present():
    at = _run()
    heads = [s.value for s in at.subheader]
    assert heads == ["📐 Drei Wege zum Gleichgewicht", "🔬 Wie lange dauert das Einschwingen?", "🔬 Wann kommt der erste Verlust?", "🔬 Nicht exponentielle Dauer: Phasen", "🚧 Wo die Annahmen enden"]
    assert any(m.value.startswith("## 🔗 Vom Gate zur Kette") for m in at.markdown)
    assert any("Diese Demo ist Teil des Portfolios" in c.value for c in at.caption)


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_button_runs(name):
    at = _run()
    next(b for b in at.button if b.key == f"preset_{name}").click().run()
    _ok(at)
    p = C.PRESETS[name]
    assert (at.session_state["c_slider"], at.session_state["k_slider"], at.session_state["rho_slider"]) == (p["c"], p["k"], p["rho_pct"])
    assert at.metric


@pytest.mark.parametrize("kw", [dict(c_slider=4, k_slider=4, rho_slider=150), dict(c_slider=1, k_slider=4, rho_slider=50), dict(c_slider=4, k_slider=10, rho_slider=150),
                                 dict(c_slider=1, k_slider=10, rho_slider=150), dict(c_slider=3, k_slider=5, rho_slider=100)])
def test_extreme_settings_run(kw):
    _ok(_run(**kw))


def test_generator_table_and_balance_table_have_the_right_size():
    at = _run(c_slider=1, k_slider=4, rho_slider=100)
    _ok(at)
    q_table = _table(at, "**Generatormatrix Q**")
    assert q_table.count("\n| **") == 5 and "| **0** | -1 | 1 | · | · | · |" in q_table and "| **4** | · | · | · | 1 | -1 |" in q_table
    assert any("erfüllt die 5 Gleichungen" in i.value for i in at.info)


def test_only_lanes_chain_is_erlang_b_and_nobody_waits():
    at = _run(c_slider=4, k_slider=4, rho_slider=80)
    _ok(at)
    assert _metric(at, "Verlust (π_K)") == "22.81 %" and _metric(at, "Wartezeit der Angenommenen") == "0.00 min"


def test_power_method_steps_slider_drives_the_table():
    at = _run(steps_slider=400)
    _ok(at)
    assert "| Potenzmethode nach 400 Schritten |" in _table(at, "| Weg |")
    at = _run(steps_slider=1)
    _ok(at)
    assert "| Potenzmethode nach 1 Schritten |" in _table(at, "| Weg |")


@pytest.mark.parametrize("key,value", [("relax_select", 50), ("relax_select", 95), ("hit_a_select", 1.0), ("hit_a_select", 5.0), ("phase_rho_select", 50), ("phase_rho_select", 95)])
def test_local_regulators_run(key, value):
    at = _run(**{key: value})
    _ok(at)
    assert len(at.get("plotly_chart")) == 6


def test_phase_table_shows_the_state_space_growth_and_agreement():
    at = _run()
    table = _table(at, "| Phasen k |")
    assert "| 1 | 1 | 4.0000 | 4.0000 | 301 |" in table
    last = table.strip().splitlines()[-1]
    assert last.startswith("| 10 | 0.1 | ") and last.endswith("| 3.001 |")


def test_dice_button_changes_the_seed(monkeypatch):
    """Der Würfel zieht sonst einen unseeded Zufalls-Seed; deshalb ist der gewürfelte Seed im Test fest."""
    monkeypatch.setattr(random, "randint", lambda a, b: 508145)
    at = _run()
    old_seed = at.session_state["seed_input"]
    next(b for b in at.button if b.label == "🎲 Neuen Lauf würfeln").click().run()
    _ok(at)
    assert at.session_state["seed_input"] == 508145 != old_seed


def test_permalink_values_are_clamped_and_snapped():
    at = AppTest.from_file(APP, default_timeout=600)
    at.query_params["rho"] = "87"
    at.query_params["c"] = "9"
    at.query_params["k"] = "1"
    at.run()
    _ok(at)
    assert at.session_state["rho_slider"] == 90 and at.session_state["c_slider"] == 4 and at.session_state["k_slider"] == 4


def test_permalink_ignores_garbage():
    at = AppTest.from_file(APP, default_timeout=600)
    at.query_params["rho"] = "viel"
    at.query_params["seed"] = "nan"
    at.run()
    _ok(at)
    assert at.session_state["rho_slider"] == C.DEFAULT_RHO_PCT and at.session_state["seed_input"] == C.DEFAULT_SEED


def test_no_sentence_wide_comma_replacement_in_the_app_source():
    """Regressionsschutz: `.replace(",", ".")` auf einem ganzen (verketteten) Satz macht aus Kommas im Fließtext Punkte; Tausender nur über `fmt_int`."""
    source = Path(APP).read_text(encoding="utf-8")
    assert '.replace(",", ".")' not in source
