"""Plotly-Abbildungen der Markov-Ketten-Demo: Zustandsdiagramm mit Raten, Gleichgewicht auf drei Wegen, Konvergenz der Potenzmethode, Einschwingen, erste Passage, Phasen. Achsen sind gesperrt
(fixedrange), damit Touch-Geräte beim Scrollen nicht zoomen."""

import math

import plotly.graph_objects as go

import mkv_chain as K

EXACT_COLOR = "#4c78a8"
POWER_COLOR = "#e45756"
SIM_COLOR = "#f58518"
BASE_COLOR = "#9d9d9d"
PHASE_COLOR = "#54a24b"


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height, legend_y=-0.25, top=10):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=top, b=10), legend=dict(orientation="h", y=legend_y),
                      plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


_SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def power_ticks(values):
    """Zehnerpotenzen als Achsenbeschriftung (Plotly würde sonst SI-Vorsilben wie 10k oder 1M schreiben): (Werte, Texte) für eine logarithmische Achse über `values`."""
    lo, hi = math.floor(math.log10(min(values))), math.ceil(math.log10(max(values)))
    step = max(1, math.ceil((hi - lo) / 7))
    exps = list(range(lo, hi + 1, step))
    return [10.0 ** e for e in exps], [("1" if e == 0 else "10" + str(e).translate(_SUP)) for e in exps]


def fmt_rate(x):
    return f"{x:g}"


def build_state_diagram(c, k, a):
    """Zustandsdiagramm der M/M/c/K-Kette: Knoten 0 … K in einer Reihe, Pfeile nach rechts (Ankunft, Rate a) und nach links (Abgang, Rate min(n, c))."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(k + 1)), y=[0] * (k + 1), mode="markers+text", marker=dict(size=34, color=[EXACT_COLOR if n <= c else "#9ecae9" for n in range(k + 1)],
                                                                                              line=dict(color="white", width=2)),
                             text=[str(n) for n in range(k + 1)], textfont=dict(color="white", size=14), name="Zustand n", hoverinfo="skip"))
    for n in range(k):
        fig.add_annotation(x=n + 0.78, y=0.22, ax=n + 0.22, ay=0.22, xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=3, arrowwidth=1.6, arrowcolor=POWER_COLOR)
        fig.add_annotation(x=n + 0.5, y=0.36, text=f"a = {fmt_rate(a)}", showarrow=False, font=dict(size=11, color=POWER_COLOR))
        fig.add_annotation(x=n + 0.22, y=-0.22, ax=n + 0.78, ay=-0.22, xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=3, arrowwidth=1.6, arrowcolor=SIM_COLOR)
        fig.add_annotation(x=n + 0.5, y=-0.36, text=f"{fmt_rate(K.service_rate(n + 1, c))}", showarrow=False, font=dict(size=11, color=SIM_COLOR))
    fig.update_xaxes(visible=False, range=[-0.5, k + 0.5])
    fig.update_yaxes(visible=False, range=[-0.6, 0.6])
    return _base(fig, 220, top=0)


def build_stationary_chart(pi_exact, pi_power, pi_sim, steps):
    """Gleichgewicht: direkte Lösung (Balken) gegen Potenzmethode nach `steps` Schritten (Linie) und Zeitanteile der Simulation (Punkte)."""
    ns = list(range(len(pi_exact)))
    fig = go.Figure()
    fig.add_trace(go.Bar(x=ns, y=pi_exact, marker_color=EXACT_COLOR, name="direkte Lösung (πQ = 0)"))
    fig.add_trace(go.Scatter(x=ns, y=pi_power, mode="lines+markers", line=dict(color=POWER_COLOR, width=2), name=f"Potenzmethode nach {steps} Schritten"))
    fig.add_trace(go.Scatter(x=ns, y=pi_sim, mode="markers", marker=dict(color=SIM_COLOR, size=9, symbol="diamond"), name="Simulation (Zeitanteile)"))
    fig.update_xaxes(title_text="Lkw im System n", dtick=1)
    fig.update_yaxes(title_text="Wahrscheinlichkeit", rangemode="tozero")
    return _base(fig, 340)


def build_convergence_chart(dists, steps, tol):
    """L1-Abstand der Potenzmethode zur exakten Lösung über die Schritte (logarithmisch); der Punkt markiert den gewählten Schritt."""
    xs = list(range(len(dists)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=[d if d > 0 else None for d in dists], mode="lines", line=dict(color=POWER_COLOR, width=2.5), name="L1-Abstand zur exakten Lösung"))
    fig.add_trace(go.Scatter(x=[steps], y=[max(dists[steps], 1e-16)], mode="markers", marker=dict(color=SIM_COLOR, size=11), name="gewählter Schritt"))
    fig.add_hline(y=tol, line=dict(color=BASE_COLOR, dash="dash"), annotation_text=f"Schwelle {tol:g}")
    fig.update_xaxes(title_text="Schritte der Potenzmethode")
    fig.update_yaxes(title_text="Abstand (logarithmisch)", type="log", range=[-16, 0.2], tickvals=[1e-15, 1e-12, 1e-9, 1e-6, 1e-3, 1],
                     ticktext=["10⁻¹⁵", "10⁻¹²", "10⁻⁹", "10⁻⁶", "0.001", "1"])
    return _base(fig, 340)


def build_transient_chart(rho, times, means, steady, t95):
    """Mittlere Zahl im System der M/M/1-Kette von leer aus über die Zeit (exakt über die Matrixexponentialfunktion): Gleichgewichtswert gestrichelt, 95 %-Zeitpunkt senkrecht."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=times, y=means, mode="lines", line=dict(color=EXACT_COLOR, width=2.5), name="E[N(t)] (exakt)"))
    fig.add_hline(y=steady, line=dict(color=BASE_COLOR, dash="dash"), annotation_text=f"Gleichgewicht {steady:.1f}")
    fig.add_vline(x=t95, line=dict(color=POWER_COLOR, dash="dot"), annotation_text=f"95 % bei t = {t95:.0f}")
    fig.update_xaxes(title_text="Zeit in Abfertigungsdauern")
    fig.update_yaxes(title_text="mittlere Zahl im System", rangemode="tozero")
    return _base(fig, 340)


def build_hitting_chart(a, cs, exact, sim_cs, sim_values):
    """Mittlere Zeit bis zum ersten Verlust über die Spurzahl (Abfertigungsdauern, logarithmisch): exakt aus der Kette, bis zehn Spuren zusätzlich simuliert."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cs, y=exact, mode="lines+markers", line=dict(color=EXACT_COLOR, width=2.5), name="exakt (lineares System)"))
    if sim_cs:
        fig.add_trace(go.Scatter(x=sim_cs, y=sim_values, mode="markers", marker=dict(color=SIM_COLOR, size=10, symbol="diamond"), name="Simulation"))
    fig.update_xaxes(title_text=f"Spuren c (Angebot {a:g} Erlang)", dtick=2)
    tickvals, ticktext = power_ticks(exact)
    fig.update_yaxes(title_text="Zeit bis zum ersten Verlust (logarithmisch)", type="log", tickvals=tickvals, ticktext=ticktext)
    return _base(fig, 340)


def build_phases_chart(rho, ks, chain_waits, pk_waits):
    """Wartezeit der M/E_k/1-Kette (Zustand = Zahl und Phase) gegen Pollaczek-Khinchine für cs² = 1/k."""
    labels = [f"k = {k} (cs² = {1 / k:g})" for k in ks]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=chain_waits, marker_color=PHASE_COLOR, name="Markov-Kette mit Phasen"))
    fig.add_trace(go.Scatter(x=labels, y=pk_waits, mode="markers", marker=dict(color=POWER_COLOR, size=12, symbol="diamond"), name="Pollaczek-Khinchine (Formel)"))
    fig.update_yaxes(title_text="mittlere Wartezeit in Abfertigungsdauern", rangemode="tozero")
    return _base(fig, 340)
