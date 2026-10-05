"""Simulation: Gillespie-Einheiten von Hand, Zeitanteile gegen das Gleichgewicht, erste Passage gegen die exakte Lösung, Reproduzierbarkeit."""

import numpy as np
import pytest

import mkv_chain as K
import mkv_simulation as S
from conftest import ScriptedRng


def test_rates_by_hand():
    assert S.rates(0, 2, 5, 1.5) == (1.5, 0.0)
    assert S.rates(1, 2, 5, 1.5) == (1.5, 1.0)
    assert S.rates(4, 2, 5, 1.5) == (1.5, 2.0)
    assert S.rates(5, 2, 5, 1.5) == (0.0, 2.0)


def test_jump_by_hand():
    """n = 1, c = 1, K = 2, a = 1: Gesamtrate 2; Zufall 0.3 < 1/2 → Ankunft (n = 2), Zufall 0.7 → Abgang (n = 0); Verweilzeit ist der Skriptwert."""
    assert S.jump(1, 1, 2, 1.0, ScriptedRng(exp_values=[0.4], uniform_values=[0.3])) == (0.4, 2)
    assert S.jump(1, 1, 2, 1.0, ScriptedRng(exp_values=[0.4], uniform_values=[0.7])) == (0.4, 0)


def test_the_full_state_never_goes_up_and_the_empty_state_never_goes_down():
    rng = ScriptedRng(exp_values=[1.0, 1.0], uniform_values=[0.0, 0.999])
    assert S.jump(2, 1, 2, 1.0, rng)[1] == 1                       # n = K: Ankunftsrate 0, auch Zufall 0.0 führt nach unten
    assert S.jump(0, 1, 2, 1.0, rng)[1] == 1                       # n = 0: Abgangsrate 0, auch Zufall 0.999 führt nach oben


def test_time_fractions_by_hand(gillespie_mini):
    """Siehe conftest: Zeiten n = 0: 0.75, n = 1: 1.25, n = 2: 0; Anteile 0.375 / 0.625 / 0 (Horizont 2.0 genau erreicht)."""
    frac = S.time_fractions(1, 2, 1.0, 2.0, gillespie_mini)
    assert frac == pytest.approx([0.375, 0.625, 0.0])


def test_the_last_jump_is_cut_off_at_the_horizon():
    rng = ScriptedRng(exp_values=[5.0], uniform_values=[0.0])
    frac = S.time_fractions(1, 2, 1.0, 2.0, rng)
    assert frac == pytest.approx([1.0, 0.0, 0.0])                  # der erste Sprung (5.0) wird bei 2.0 abgeschnitten, die ganze Zeit liegt bei n = 0


@pytest.mark.parametrize("c,k,a", [(1, 6, 0.8), (2, 6, 1.6), (3, 5, 4.0)])
def test_time_fractions_match_the_stationary_distribution(c, k, a):
    frac = S.time_fractions(c, k, a, 150_000.0, S.SplitMix64(11))
    assert np.abs(frac - K.stationary_direct(K.generator(c, k, a))).sum() < 0.03
    assert frac.sum() == pytest.approx(1.0)


def test_first_loss_time_by_hand():
    """c = 1, a = 2: die erste Ankunft belegt die Spur (n = 1), erst die zweite Ankunft ist der Verlust. Skript 0.7 (0 → 1), 0.3 (1 → 2 = Verlust, Zufall 0.0 < 2/3): Zeit 1.0.
    Mit Rückfall: 0.7 (0 → 1), 0.3 (Zufall 0.9 ≥ 2/3: Abgang, 1 → 0), 0.5 (0 → 1), 0.2 (1 → 2): Zeit 1.7 (bis die Spur zum ersten Mal belegt ist, wären es nur 0.7)."""
    assert S.first_loss_time(1, 2.0, ScriptedRng(exp_values=[0.7, 0.3], uniform_values=[0.0, 0.0])) == pytest.approx(1.0)
    assert S.first_loss_time(1, 2.0, ScriptedRng(exp_values=[0.7, 0.3, 0.5, 0.2], uniform_values=[0.0, 0.9, 0.0, 0.0])) == pytest.approx(1.7)


def test_mean_first_loss_time_matches_the_exact_solution():
    for c, a in ((2, 1.0), (5, 3.0)):
        mean = S.mean_first_loss_time(c, a, 4000, S.SplitMix64(7))
        assert mean == pytest.approx(K.hitting_time_loss(c, a), rel=0.06)


def test_same_seed_same_result():
    assert (S.time_fractions(2, 6, 1.6, 2_000.0, S.SplitMix64(5)) == S.time_fractions(2, 6, 1.6, 2_000.0, S.SplitMix64(5))).all()
    assert S.mean_first_loss_time(3, 2.0, 50, S.SplitMix64(5)) == S.mean_first_loss_time(3, 2.0, 50, S.SplitMix64(5))
    assert S.mean_first_loss_time(3, 2.0, 50, S.SplitMix64(5)) != S.mean_first_loss_time(3, 2.0, 50, S.SplitMix64(6))
