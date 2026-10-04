"""JEDE Zahl aus README und App-Texten wird hier nachgerechnet. Zeiten in Abfertigungsdauern, wo nicht Minuten (3 min Mittel) dasteht."""

import numpy as np
import pytest

import mkv_chain as K
import mkv_constants as C
import mkv_simulation as S


def test_default_chain_numbers():
    """README: Zwei Spuren, 6 Plätze, Last 80 %: Verlust 7.60 %, 2.45 Lkw im System, Wartezeit 1.98 min, Auslastung 73.9 %."""
    m = K.metrics(2, 6, 1.6)
    assert C.fmt_pct(m["blocking"], 2) == "7.60 %" and round(m["L"], 2) == 2.45 and f"{K.to_minutes(m['Wq']):.2f}" == "1.98" and C.fmt_pct(m["utilisation"], 1) == "73.9 %"


def test_preset_numbers():
    """README und PRESET_HELP: eine Spur 6.6 % / 2.14 / 5.6 min; Überlast 36.0 %, 96 % ausgelastet, 4.0 min; vier Spuren ohne Warteplatz 22.8 % = Erlang B, niemand wartet."""
    one = K.metrics(1, 6, 0.8)
    assert C.fmt_pct(one["blocking"], 1) == "6.6 %" and round(one["L"], 2) == 2.14 and f"{K.to_minutes(one['Wq']):.1f}" == "5.6"
    two = K.metrics(2, 6, 1.6)
    assert C.fmt_pct(two["blocking"], 1) == "7.6 %" and round(two["L"], 2) == 2.45 and f"{K.to_minutes(two['Wq']):.1f}" == "2.0"
    over = K.metrics(2, 6, 3.0)
    assert C.fmt_pct(over["blocking"], 1) == "36.0 %" and C.fmt_pct(over["utilisation"]) == "96 %" and f"{K.to_minutes(over['Wq']):.1f}" == "4.0"
    lanes = K.metrics(4, 4, 3.2)
    assert C.fmt_pct(lanes["blocking"], 1) == "22.8 %" and lanes["blocking"] == pytest.approx(K.erlang_b(4, 3.2)) and lanes["Lq"] == 0.0
    for name, text in C.PRESET_HELP.items():
        assert text.startswith(("Eine Spur", "Zwei Spuren", "Vier Spuren")) and name in C.PRESETS


def test_power_method_steps():
    """README: eine Spur, 300 Plätze, L1-Abstand unter 1e-8: 219 / 2 076 / 9 296 / 38 587 Schritte bei Last 50 / 80 / 90 / 95 %; die Voreinstellung 149."""
    assert [K.steps_to_converge(K.generator(1, 300, r)) for r in (0.5, 0.8, 0.9, 0.95)] == [219, 2076, 9296, 38587]
    assert K.steps_to_converge(K.generator(2, 6, 1.6)) == 149


def test_default_simulation_and_closed_form_distance():
    """README: Seed 35, 100 000 Abfertigungsdauern: L1-Abstand der Simulation 0.0087; direkte Lösung gegen geschlossene Form unter 1e-15."""
    pi = K.stationary_direct(K.generator(2, 6, 1.6))
    frac = S.time_fractions(2, 6, 1.6, C.SIM_HORIZON, S.SplitMix64(C.DEFAULT_SEED))
    assert round(float(np.abs(frac - pi).sum()), 4) == 0.0087
    assert float(np.abs(pi - K.stationary_closed(2, 6, 1.6)).sum()) < 1e-15


def test_relaxation_times():
    """README: 14.9 / 102.1 / 418.9 / 1 696.3 Abfertigungsdauern (Last 50 / 80 / 90 / 95 %), rund 45 min / 5 h / 21 h / 3.5 Tage; Faustformel 11.7 / 89.7 / 379.7 / 1 559.7, Verhältnis 1.28 / 1.14 / 1.10 / 1.09."""
    rhos = (0.5, 0.8, 0.9, 0.95)
    ts = [K.relaxation_time(r) for r in rhos]
    assert [round(t, 1) for t in ts] == [14.9, 102.1, 418.9, 1696.3]
    assert [round(K.to_minutes(t)) for t in ts[:1]] == [45] and round(K.to_minutes(ts[1]) / 60) == 5 and round(K.to_minutes(ts[2]) / 60) == 21 and round(K.to_minutes(ts[3]) / 1440, 1) == 3.5
    assert [round(K.relaxation_heuristic(r), 1) for r in rhos] == [11.7, 89.7, 379.7, 1559.7]
    assert [round(t / K.relaxation_heuristic(r), 2) for t, r in zip(ts, rhos)] == [1.28, 1.14, 1.10, 1.09]


def test_first_loss_times():
    """README: Angebot 3: 5 Spuren 4.0, 8 Spuren 28.7, 10 Spuren 192.9, 15 Spuren 156 329 (rund 11 Monate), 20 Spuren 8.3·10⁸ (rund 4 760 Jahre)."""
    h = {c: K.hitting_time_loss(c, 3.0) for c in (5, 8, 10, 15, 20)}
    assert [round(h[5], 1), round(h[8], 1), round(h[10], 1), round(h[15]), f"{h[20]:.1e}"] == [4.0, 28.7, 192.9, 156329, "8.3e+08"]
    assert round(K.to_minutes(h[15]) / (30.44 * 24 * 60)) == 11 and round(K.to_minutes(h[20]) / (365 * 24 * 60), -1) == 4760
    assert C.fmt_sci(h[20]) == "8.3·10⁸"


def test_the_dense_system_fails_where_the_stable_solution_does_not():
    """README: Dichte Lösung bei Angebot 1 und 20 Spuren: Betrag in der Größenordnung 10¹⁷, aber mit falschem Vorzeichen bzw. mehr als 100 % daneben (3.5·10¹⁷ ist richtig); 15 Spuren: Fehler
    in der Größenordnung 10⁻⁶; Angebot 3, 20 Spuren: Fehler zwischen 10⁻⁹ und 10⁻⁶; Stufenrekursion gegen Bruchrechnung auf 1e-15. Die Stellen des schlecht konditionierten dichten Systems hängen
    von der LAPACK-Bibliothek ab (Windows 2e-8, Linux 4e-8), deshalb stehen hier Bänder statt Ziffern."""
    from fractions import Fraction

    def exact(c, a):
        d, tot = Fraction(0), Fraction(0)
        for i in range(c):
            d = (1 + i * d) / Fraction(a)
            tot += d
        return float(tot)

    assert f"{exact(20, 1):.1e}" == "3.5e+17"
    dense = K.hitting_time_dense(20, 1.0)
    assert abs(dense / exact(20, 1) - 1) > 1.0 and 1e16 < abs(dense) < 1e19
    err15 = abs(K.hitting_time_dense(15, 1.0) / exact(15, 1) - 1)
    err20 = abs(K.hitting_time_dense(20, 3.0) / exact(20, 3) - 1)
    assert 1e-7 < err15 < 1e-4 and 1e-9 < err20 < 1e-6
    assert max(abs(K.hitting_time_loss(c, a) / exact(c, a) - 1) for c in range(2, 21) for a in (1, 2, 3, 5)) < 1e-15


def test_phase_chain_numbers():
    """README: M/E_k/1, Last 80 %, k = 1 / 2 / 4 / 10: Wartezeit 4.0 / 3.0 / 2.5 / 2.2, Zustände 301 / 601 / 1 201 / 3 001 (Abschneiden bei 300 Aufträgen)."""
    out = [K.mek1_wait(0.8, k, C.PHASE_TRUNCATION) for k in (1, 2, 4, 10)]
    assert [round(w, 1) for w, _ in out] == [4.0, 3.0, 2.5, 2.2] and [n for _, n in out] == [301, 601, 1201, 3001]
    assert [round(K.pk_wait(0.8, 1 / k), 1) for k in (1, 2, 4, 10)] == [4.0, 3.0, 2.5, 2.2]


def test_app_texts_quote_the_measured_power_steps():
    """App-Text im Abschnitt Drei Wege nennt 219 / 2.076 / 9.296 / 38.587 Schritte; dieselben Zahlen wie die README."""
    from pathlib import Path
    source = (Path(__file__).resolve().parent.parent / "app.py").read_text(encoding="utf-8")
    assert "219 / 2.076 / 9.296 / 38.587" in source
