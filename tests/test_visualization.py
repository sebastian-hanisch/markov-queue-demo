"""Abbildungen: gesperrte Achsen, Zahl der Linien, Beschriftungen."""

import mkv_chain as K
import mkv_constants as C
import mkv_visualization as V


def _locked(fig):
    return all(ax.fixedrange for ax in fig.select_xaxes()) and all(ax.fixedrange for ax in fig.select_yaxes())


def test_all_charts_lock_their_axes():
    q = K.generator(2, 6, 1.6)
    pi = K.stationary_direct(q)
    x, dists = K.power_method(q, 50)
    figs = [V.build_state_diagram(2, 6, 1.6), V.build_stationary_chart(pi, x, pi, 50), V.build_convergence_chart(dists, 10, C.POWER_TOL),
            V.build_transient_chart(0.8, [0.0, 1.0, 2.0], [0.0, 0.5, 0.9], 4.0, 90.0), V.build_hitting_chart(3.0, [2, 3], [3.0, 8.0], [2], [3.1]),
            V.build_phases_chart(0.8, [1, 2], [2.0, 1.6], [2.0, 1.6])]
    assert all(_locked(f) for f in figs)


def test_state_diagram_has_two_arrows_and_two_labels_per_edge():
    fig = V.build_state_diagram(2, 6, 1.6)
    assert len(fig.layout.annotations) == 4 * 6 and len(fig.data[0].x) == 7
    labels = [a.text for a in fig.layout.annotations if a.text]
    assert labels.count("a = 1.6") == 6 and labels[1::2][:3] == ["1", "2", "2"]                    # Abgangsraten min(n, c): 1, 2, 2, …


def test_stationary_chart_has_three_layers():
    q = K.generator(1, 4, 0.8)
    pi = K.stationary_direct(q)
    assert [t.name.split(" ")[0] for t in V.build_stationary_chart(pi, pi, pi, 7).data] == ["direkte", "Potenzmethode", "Simulation"]


def test_hitting_chart_omits_the_simulation_when_there_are_no_points():
    assert len(V.build_hitting_chart(3.0, [2, 3], [3.0, 8.0], [], []).data) == 1
    assert len(V.build_hitting_chart(3.0, [2, 3], [3.0, 8.0], [2], [3.1]).data) == 2


def test_convergence_chart_marks_the_chosen_step():
    fig = V.build_convergence_chart([1.0, 0.5, 0.25, 0.0], 2, C.POWER_TOL)
    assert fig.data[1].x == (2,) and fig.data[1].y == (0.25,) and fig.data[0].y[3] is None          # exakte Null wird nicht auf der Log-Achse gezeichnet


def test_hitting_chart_labels_powers_of_ten_instead_of_si_prefixes():
    fig = V.build_hitting_chart(1.0, [2, 20], [3.0, 3.5e17], [], [])
    texts = fig.layout.yaxis.ticktext
    assert texts[0] == "1" and texts[-1] == "10¹⁸" and not any(t.endswith(("k", "M", "G")) for t in texts)
    assert fig.layout.yaxis.tickvals[-1] == 1e18 and len(texts) == len(fig.layout.yaxis.tickvals)


def test_power_ticks_by_hand():
    assert V.power_ticks([4.0, 8.3e8]) == ([1.0, 1e2, 1e4, 1e6, 1e8], ["1", "10²", "10⁴", "10⁶", "10⁸"])
    assert V.power_ticks([0.5, 30.0])[1] == ["10⁻¹", "1", "10¹", "10²"]
