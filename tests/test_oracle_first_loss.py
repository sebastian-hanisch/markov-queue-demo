"""Orakel für die Zeit bis zum ersten Verlust: ein UNABHÄNGIGES Verlustsystem (Poisson-Ankünfte, jeder Lkw mit eigener Abfertigungsdauer, Erlang-B-Regel „alle Spuren belegt: abgewiesen“,
Python-`random`, keine Markov-Kette) und das Gleichungssystem der ersten Schritte per Gauß-Verfahren in exakter Bruchrechnung (anderer Rechenweg als die Stufenrekursion)."""

import math
import random
from fractions import Fraction

import pytest

import mkv_chain as K
import mkv_simulation as S


def _gauss_hitting_time(c, a):
    """(a + i)·h_i = 1 + a·h_{i+1} + i·h_{i−1} für i = 0 … c, h_{c+1} = 0 (Ankunft im Zustand c ist der Verlust), Gauß-Jordan mit Fraction."""
    n = c + 1
    a = Fraction(a)
    m = [[Fraction(0)] * (n + 1) for _ in range(n)]
    for i in range(n):
        m[i][i] = -(a + i)
        if i + 1 < n:
            m[i][i + 1] = a
        if i > 0:
            m[i][i - 1] = Fraction(i)
        m[i][n] = Fraction(-1)
    for col in range(n):
        piv = next(r for r in range(col, n) if m[r][col] != 0)
        m[col], m[piv] = m[piv], m[col]
        for r in range(n):
            if r != col and m[r][col] != 0:
                f = m[r][col] / m[col][col]
                m[r] = [x - f * y for x, y in zip(m[r], m[col])]
    return float(m[0][n] / m[0][0])


def _independent_first_loss_time(c, a, rng):
    """Verlustsystem Lkw für Lkw: Ankunft nach Exp(a); sind alle c Spuren belegt, ist das der erste Verlust, sonst bekommt der Lkw eine eigene Exp(1)-Abfertigungsdauer."""
    t, ends = 0.0, []
    while True:
        t += rng.expovariate(a)
        ends = [e for e in ends if e > t]
        if len(ends) == c:
            return t
        ends.append(t + rng.expovariate(1.0))


@pytest.mark.parametrize("c,a", [(1, 2.0), (2, 1.0), (3, 1.0), (4, 3.0), (5, 3.0), (8, 2.0), (12, 5.0)])
def test_first_loss_time_equals_the_exact_gauss_solution(c, a):
    assert K.hitting_time_loss(c, a) == pytest.approx(_gauss_hitting_time(c, a), rel=1e-12)


@pytest.mark.parametrize("c,a", [(1, 2.0), (2, 1.0), (3, 2.0), (4, 3.0)])
def test_first_loss_time_equals_an_independent_loss_system_simulation(c, a):
    """Der Mittelwert über 3000 unabhängige Läufe liegt innerhalb von vier Standardfehlern am exakten Wert; Zufall `random.Random` mit festem Seed."""
    rng = random.Random(20261005 + 31 * c)
    xs = [_independent_first_loss_time(c, a, rng) for _ in range(3000)]
    mean = sum(xs) / len(xs)
    se = math.sqrt(sum((x - mean) ** 2 for x in xs) / (len(xs) - 1) / len(xs))
    assert abs(mean - K.hitting_time_loss(c, a)) < 4 * se


def test_first_loss_is_later_than_the_first_time_all_lanes_are_busy():
    """c = 2, a = 1: bis beide Spuren zum ersten Mal belegt sind, dauert es 3, bis zum ersten abgewiesenen Lkw 8."""
    assert K.hitting_time_loss(2, 1.0) == pytest.approx(8.0)
    runs = [S.first_loss_time(2, 1.0, S.SplitMix64(500 + i)) for i in range(2000)]
    assert sum(runs) / len(runs) > 6.5
