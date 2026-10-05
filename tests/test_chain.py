"""Kette: Generator von Hand, Gleichgewicht (direkt, geschlossen, Potenzmethode), Kennzahlen, Little, Erlang B, Einschwingen (unabhängiger RK4-Vergleich), erste Passage von Hand, Phasen."""

import numpy as np
import pytest

import mkv_chain as K


def test_generator_by_hand():
    """c = 1, K = 2, a = 0.5: Zeilen (−0.5, 0.5, 0), (1, −1.5, 0.5), (0, 1, −1)."""
    q = K.generator(1, 2, 0.5)
    assert q == pytest.approx(np.array([[-0.5, 0.5, 0.0], [1.0, -1.5, 0.5], [0.0, 1.0, -1.0]]))


def test_generator_with_two_lanes_by_hand():
    """c = 2, K = 3, a = 1: Abgangsraten 1, 2, 2; Diagonale −(a + μ)."""
    q = K.generator(2, 3, 1.0)
    assert q[1, 0] == 1.0 and q[2, 1] == 2.0 and q[3, 2] == 2.0
    assert q[0, 0] == -1.0 and q[1, 1] == -2.0 and q[2, 2] == -3.0 and q[3, 3] == -2.0


@pytest.mark.parametrize("c,k,a", [(1, 5, 0.8), (2, 6, 1.6), (4, 10, 6.0), (3, 4, 0.3)])
def test_generator_rows_sum_to_zero_and_off_diagonals_are_nonnegative(c, k, a):
    q = K.generator(c, k, a)
    assert q.sum(axis=1) == pytest.approx(np.zeros(k + 1), abs=1e-12)
    assert (q - np.diag(np.diag(q)) >= 0).all()


def test_invalid_generator_input_is_rejected():
    for args in ((0, 4, 1.0), (3, 2, 1.0), (1, 4, 0.0)):
        with pytest.raises(ValueError):
            K.generator(*args)


def test_stationary_distribution_by_hand():
    """c = 1, K = 2, a = 1: Gewichte 1, 1, 1 → je 1/3; Verlust 1/3, Durchsatz 2/3, L = 1, Wq = Lq/Durchsatz = (1/3)/(2/3) = 0.5."""
    q = K.generator(1, 2, 1.0)
    assert K.stationary_direct(q) == pytest.approx([1 / 3] * 3)
    assert K.stationary_closed(1, 2, 1.0) == pytest.approx([1 / 3] * 3)
    m = K.metrics(1, 2, 1.0)
    assert m["blocking"] == pytest.approx(1 / 3) and m["throughput"] == pytest.approx(2 / 3)
    assert m["L"] == pytest.approx(1.0) and m["Wq"] == pytest.approx(0.5)


@pytest.mark.parametrize("c,k,a", [(1, 6, 0.8), (2, 6, 1.6), (4, 10, 6.0), (3, 7, 5.0)])
def test_direct_solution_matches_the_closed_form_and_balances(c, k, a):
    q = K.generator(c, k, a)
    pi = K.stationary_direct(q)
    assert pi == pytest.approx(K.stationary_closed(c, k, a), abs=1e-12)
    assert pi.sum() == pytest.approx(1.0) and (pi > 0).all()
    assert np.abs(pi @ q).max() < 1e-12                                                        # πQ = 0
    for n in range(k):                                                                         # detailed balance: π_n a = π_{n+1} μ_{n+1}
        assert pi[n] * a == pytest.approx(pi[n + 1] * K.service_rate(n + 1, c), rel=1e-9)


def test_large_mm1_chain_is_the_geometric_distribution():
    rho = 0.7
    pi = K.stationary_closed(1, 200, rho)
    geo = np.array([(1 - rho) * rho ** n for n in range(201)])
    assert pi == pytest.approx(geo / geo.sum(), abs=1e-14)
    assert K.metrics(1, 200, rho)["L"] == pytest.approx(rho / (1 - rho), rel=1e-6)


def test_littles_law_and_erlang_b_come_out_of_the_chain():
    m = K.metrics(2, 6, 1.6)
    assert m["W"] == pytest.approx(m["L"] / m["throughput"])
    assert m["W"] - m["Wq"] == pytest.approx(1.0)                                              # Verweilzeit = Wartezeit + eine Abfertigungsdauer
    for c, a in ((4, 3.2), (3, 2.0), (10, 7.0)):
        assert K.metrics(c, c, a)["blocking"] == pytest.approx(K.erlang_b(c, a), rel=1e-12)
    assert K.metrics(4, 4, 3.2)["Wq"] == pytest.approx(0.0)


def test_erlang_b_by_hand():
    assert K.erlang_b(1, 1.0) == pytest.approx(0.5) and K.erlang_b(2, 1.0) == pytest.approx(0.2)


def test_uniformized_chain_is_stochastic_and_keeps_the_stationary_distribution():
    q = K.generator(2, 6, 1.6)
    p, lam = K.uniformized(q)
    assert p.sum(axis=1) == pytest.approx(np.ones(7)) and (p >= 0).all() and lam == pytest.approx(3.6)
    pi = K.stationary_direct(q)
    assert pi @ p == pytest.approx(pi, abs=1e-14)


def test_power_method_converges_monotonically_to_the_direct_solution():
    q = K.generator(2, 6, 1.6)
    x, dists = K.power_method(q, 400)
    assert dists[0] == pytest.approx(2 * (1 - K.stationary_direct(q)[0]))
    assert dists[-1] < 1e-10 and abs(x.sum() - 1) < 1e-12
    assert all(b <= a + 1e-12 for a, b in zip(dists, dists[1:]))


def test_power_method_needs_more_steps_for_higher_load():
    steps = [K.steps_to_converge(K.generator(1, 60, rho)) for rho in (0.5, 0.8, 0.9)]
    assert steps[0] < steps[1] < steps[2]


def _rk4_transient(q, p0, t, n=4000):
    """Unabhängige Referenz: dp/dt = pQ numerisch mit Runge-Kutta 4. Ordnung."""
    p = np.array(p0, float)
    h = t / n
    for _ in range(n):
        k1 = p @ q
        k2 = (p + h / 2 * k1) @ q
        k3 = (p + h / 2 * k2) @ q
        k4 = (p + h * k3) @ q
        p = p + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return p


def test_transient_mean_matches_an_independent_runge_kutta_integration():
    """M/M/1, ρ = 0.5, 60 Zustände: E[N(t)] aus der Matrixexponentialfunktion gegen Runge-Kutta zu t = 3 und t = 10."""
    q = K.generator(1, 59, 0.5)
    p0 = np.zeros(60)
    p0[0] = 1.0
    for t, mean in zip((3.0, 10.0), K.transient_mean(0.5, [3.0, 10.0], size=60)):
        assert mean == pytest.approx(float((_rk4_transient(q, p0, t) * np.arange(60)).sum()), rel=1e-8)


def test_transient_mean_starts_at_zero_and_rises_to_the_steady_value():
    means = K.transient_mean(0.8, [0.0, 5.0, 50.0, 500.0])
    assert means[0] == pytest.approx(0.0, abs=1e-12)
    assert means[0] < means[1] < means[2] < means[3]
    assert means[3] == pytest.approx(4.0, rel=1e-3)


def test_relaxation_time_grows_with_load_and_follows_the_heuristic_in_order_of_magnitude():
    ts = [K.relaxation_time(r) for r in (0.5, 0.8, 0.9)]
    assert ts[0] < ts[1] < ts[2]
    for r, t in zip((0.5, 0.8, 0.9), ts):
        assert 1.0 < t / K.relaxation_heuristic(r) < 1.4


def test_relaxation_time_reaches_the_level_by_construction():
    t = K.relaxation_time(0.8)
    assert K.transient_mean(0.8, [t])[0] == pytest.approx(0.95 * 4.0, rel=1e-5)


def test_hitting_time_by_hand():
    """Zeit bis zum ersten VERLUST (Ankunft auf lauter belegte Spuren), von Hand über die Gleichungen der ersten Schritte:
    c = 1, a = 2: h₀ = 1/a + h₁, h₁ = 1/(a + 1) + h₀/(a + 1) (mit Wahrscheinlichkeit 1/(a + 1) geht der Lkw weg und das Gate ist wieder leer) → h₀ = 1.25.
    c = 2, a = 1: h₀ = 1 + h₁, h₁ = 1/2 + (h₂ + h₀)/2, h₂ = 1/3 + (2/3)h₁ → h₁ = 7, h₀ = 8 (nicht 3: so lange dauert es nur, bis beide Spuren zum ersten Mal belegt sind).
    c = 3, a = 1: Stufen 1, 2, 5, 16 → 24."""
    assert K.hitting_time_loss(1, 2.0) == pytest.approx(1.25)
    assert K.hitting_time_loss(2, 1.0) == pytest.approx(8.0)
    assert K.hitting_time_loss(3, 1.0) == pytest.approx(24.0)


def _exact_hitting_time(c, a):
    """Unabhängige Referenz in exakter Bruchrechnung (Stufenrekursion d_i = (1 + i·d_{i−1})/a mit Fraction, kein Rundungsfehler)."""
    from fractions import Fraction
    d, total = Fraction(0), Fraction(0)
    for i in range(c + 1):
        d = (1 + i * d) / Fraction(a)
        total += d
    return float(total)


@pytest.mark.parametrize("c", [2, 5, 8, 10, 15, 20])
@pytest.mark.parametrize("a", [1, 2, 3, 5])
def test_hitting_time_matches_exact_rational_arithmetic(c, a):
    assert K.hitting_time_loss(c, a) == pytest.approx(_exact_hitting_time(c, a), rel=1e-12)


def test_the_dense_system_agrees_for_small_c_and_loses_digits_for_large_c():
    """Dasselbe System dicht gelöst: für c ≤ 8 gleich (1e-9), bei Angebot 1 und c = 20 um mehr als die Hälfte daneben (Matrix schlecht konditioniert)."""
    for c in (2, 5, 8):
        for a in (1.0, 2.0, 3.0):
            assert K.hitting_time_dense(c, a) == pytest.approx(K.hitting_time_loss(c, a), rel=1e-9)
    assert abs(K.hitting_time_dense(20, 1.0) / K.hitting_time_loss(20, 1.0) - 1) > 0.5
    assert abs(K.hitting_time_dense(15, 3.0) / K.hitting_time_loss(15, 3.0) - 1) < 1e-9


def test_hitting_time_grows_with_the_number_of_lanes_and_falls_with_the_offer():
    h = [K.hitting_time_loss(c, 3.0) for c in range(2, 12)]
    assert all(a < b for a, b in zip(h, h[1:]))
    assert K.hitting_time_loss(8, 3.0) > K.hitting_time_loss(8, 5.0)


def test_hitting_time_rejects_invalid_input():
    for args in ((0, 1.0), (3, 0.0)):
        with pytest.raises(ValueError):
            K.hitting_time_loss(*args)


def test_phase_chain_reproduces_pollaczek_khinchine():
    """M/E_k/1 als Kette mit Phasen: Wartezeit gleich ρ(1 + 1/k)/(2(1 − ρ)); k = 1 ist M/M/1."""
    for rho in (0.5, 0.8):
        for k in (1, 2, 4):
            w, _ = K.mek1_wait(rho, k, 250)
            assert w == pytest.approx(K.pk_wait(rho, 1.0 / k), rel=1e-6)


def test_phase_chain_state_count_and_generator_properties():
    q, index = K.mek1_chain(0.8, 3, 20)
    assert len(index) == 1 + 3 * 20 and q.shape == (61, 61)
    assert np.abs(np.asarray(q.sum(axis=1)).ravel()).max() < 1e-12
    assert K.mek1_wait(0.8, 4, 20)[1] == 81


def test_pk_wait_by_hand_and_invalid_input():
    assert K.pk_wait(0.5, 1.0) == pytest.approx(1.0) and K.pk_wait(0.8, 0.0) == pytest.approx(2.0)
    with pytest.raises(ValueError):
        K.pk_wait(1.0, 1.0)


def test_minutes():
    assert K.to_minutes(2.0) == 6.0
