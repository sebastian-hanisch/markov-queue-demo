"""Warteschlangen als Markov-Ketten - Zusatzstück der Konzepte-Linie "Warteschlangentheorie und Simulation"
Sebastian Hanisch - Operations Research und Machine Learning

Einführung in das Rechenwerkzeug hinter den Formeln der Linie: Zustandsdiagramm, Generatormatrix, Gleichgewichtsgleichungen und drei Wege zum Gleichgewicht (direkt, Potenzmethode, Simulation) für ein
kleines Gate, dazu die exakte Einschwingzeit, die Zeit bis zum ersten Verlust und Phasen-Ketten für nicht exponentielle Dauer. Siehe README für die Einordnung in die Linie.

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import mkv_chain as K
import mkv_constants as C
import mkv_simulation as S
from mkv_presets import (apply_preset, bounds, init_session_state_defaults, load_permalink_settings, randomize_seed, sync_query_params)
from mkv_visualization import (build_convergence_chart, build_hitting_chart, build_phases_chart, build_state_diagram, build_stationary_chart, build_transient_chart)

st.set_page_config(page_title="Markov-Ketten – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _relaxation(rho):
    t95 = K.relaxation_time(rho)
    times = [t95 * f / 39 * 1.6 for f in range(40)]
    return t95, times, K.transient_mean(rho, times)


@st.cache_data(show_spinner=False)
def _relaxation_table():
    return {r: K.relaxation_time(r / 100) for r in C.RELAX_RHO_PCT}


@st.cache_data(show_spinner=False)
def _simulated_fractions(c, k, a, seed):
    return S.time_fractions(c, k, a, C.SIM_HORIZON, S.SplitMix64(seed))


@st.cache_data(show_spinner=False)
def _hitting(a, seed):
    exact = [K.hitting_time_loss(c, a) for c in C.HIT_C_RANGE]
    sim_cs = [c for c, e in zip(C.HIT_C_RANGE, exact) if c <= C.HIT_SIM_MAX_C and e <= 250 and c % 2 == 0]
    rng = S.SplitMix64(seed)
    sim = [S.mean_first_loss_time(c, a, 1000, rng) for c in sim_cs]
    return exact, sim_cs, sim


@st.cache_data(show_spinner=False)
def _phases(rho):
    out = []
    for k in C.PHASE_K_OPTIONS:
        w, n = K.mek1_wait(rho, k, C.PHASE_TRUNCATION)
        out.append((k, w, n, K.pk_wait(rho, 1.0 / k)))
    return out


def _time_text(x):
    """Zeit in Abfertigungsdauern → Text in der passenden Einheit bei 3 min Abfertigung."""
    minutes = K.to_minutes(x)
    if minutes < 120:
        return f"{minutes:.0f} min"
    if minutes < 48 * 60:
        return f"{minutes / 60:.1f} h"
    if minutes < 2 * 365 * 24 * 60:
        return f"{minutes / (24 * 60):.0f} Tage"
    return f"{minutes / (365 * 24 * 60):,.0f} Jahre".replace(",", " ")


st.title("🔗 Warteschlangen als Markov-Ketten")
st.markdown(
    """
Hinter fast allen Formeln dieser Linie steckt dasselbe Werkzeug: eine **Markov-Kette**. Der **Zustand** ist die Zahl der Lkw im Gate; von jedem Zustand führen **Pfeile mit Raten** zu den Nachbarn
(Ankunft nach rechts, Abgang nach links). Aus dem Diagramm folgt die **Generatormatrix Q**, aus Q die **Gleichgewichtsgleichungen** πQ = 0, und aus dem Gleichgewicht alle Kennzahlen. Diese Demo macht den Weg
sichtbar: vom Gate zum Diagramm, drei Wege zur Lösung, die Frage **wie lange das Einschwingen dauert** und **wann der erste Verlust kommt**, und die Grenze der Methode bei nicht exponentieller Dauer.
"""
)
st.caption(
    "Zusatzstück der Linie „Warteschlangentheorie und Simulation“ (nicht nummeriert), die Grundlage zu [mm1-queue-demo](https://sebastianhanisch-mm1-queue-demo.streamlit.app/) (Stück 1) und "
    "[mmc-queue-demo](https://sebastianhanisch-mmc-queue-demo.streamlit.app/) (Stück 3). Jedes Folgestück hebt eine der Annahmen unter „Wo die Annahmen enden“ auf."
)

with st.expander("So wird aus dem Gate eine Kette", expanded=True):
    st.markdown(
        """
- **Zustand:** n = Zahl der Lkw im System (0 bis K, K = Spuren plus Warteplätze). **Übergänge:** n → n + 1 mit der Ankunftsrate a (solange n < K), n → n − 1 mit der Abgangsrate min(n, c) (jede belegte Spur mit Rate 1).
- **Generator Q:** Zeile n enthält die Raten aus n heraus, die Diagonale macht jede Zeile zur Summe null. **Gleichgewicht:** πQ = 0, Σπ = 1 (zufließen = abfließen für jeden Zustand).
- **Drei Wege:** (1) **direkt** als lineares System; (2) **Potenzmethode**: die uniformisierte Kette P = I + Q/Λ immer wieder anwenden, bis sich nichts mehr ändert; (3) **Simulation**: Zeitanteile eines langen Laufs.
- **Mehr aus derselben Kette:** die exakte **Einschwingzeit** (Matrixexponentialfunktion), die **Zeit bis zum ersten Verlust** (lineares System über die übrigen Zustände), **Phasen** für nicht exponentielle Dauer.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
preset_cols = st.columns(len(C.PRESET_ORDER))
for i, name in enumerate(C.PRESET_ORDER):
    with preset_cols[i]:
        st.button(name, key=f"preset_{name}", width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name] or None)

st.caption(
    "🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, "
    "um ein Szenario zu teilen."
)

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    c = st.slider("Spuren c", *bounds("c_slider"), key="c_slider", help="So viele Lkw können gleichzeitig abgefertigt werden (Abfertigungsrate 1 je Spur).")
    k = st.slider("Plätze insgesamt K", *bounds("k_slider"), key="k_slider", help="Spuren plus Warteplätze; wer bei K Lkw im System ankommt, geht verloren.")
    rho_pct = st.slider("Last je Spur", *bounds("rho_slider"), step=C.RHO_PCT_STEP, key="rho_slider", format="%d %%",
                        help="Angebot a = c · Last; über 100 % ist Überlast, die endliche Kette bleibt dabei lösbar.")
    seed = st.number_input("Zufalls-Seed", min_value=bounds("seed_input")[0], max_value=bounds("seed_input")[1], step=1,
                           key="seed_input", help="Bestimmt alle Zufallszahlen der Simulationen.")
    st.button("🎲 Neuen Lauf würfeln", on_click=randomize_seed)

c, k, rho_pct, seed = int(c), int(k), int(rho_pct), int(seed)
a = c * rho_pct / 100.0
sync_query_params({"c_slider": c, "k_slider": k, "rho_slider": rho_pct, "seed_input": seed})

q = K.generator(c, k, a)
pi = K.stationary_direct(q)
m = K.metrics(c, k, a, pi)

st.markdown("---")
st.markdown("## 🔗 Vom Gate zur Kette")
st.caption(f"{c} Spur(en), {k} Plätze, Last {rho_pct} % (Angebot a = {a:g} Erlang). Zustände 0 bis {k}; Pfeile oben: Ankunft, unten: Abgang.")
st.plotly_chart(build_state_diagram(c, k, a), width="stretch", key=f"diagram_{c}_{k}_{rho_pct}")
r1 = st.columns(4)
r1[0].metric("Verlust (π_K)", C.fmt_pct(m["blocking"], 2), help="Wahrscheinlichkeit, dass das System voll ist (PASTA: auch der Anteil abgewiesener Ankünfte).")
r1[1].metric("Mittlere Zahl im System", f"{m['L']:.3f}")
r1[2].metric("Wartezeit der Angenommenen", f"{K.to_minutes(m['Wq']):.2f} min", help="Lq geteilt durch den Durchsatz a(1 − π_K) (Gesetz von Little).")
r1[3].metric("Auslastung der Spuren", C.fmt_pct(m["utilisation"], 1))
header = "| von \\ nach | " + " | ".join(str(j) for j in range(k + 1)) + " |\n|---|" + "---|" * (k + 1) + "\n"
body = ""
for i in range(k + 1):
    body += f"| **{i}** | " + " | ".join(f"{q[i, j]:g}" if q[i, j] != 0 else "·" for j in range(k + 1)) + " |\n"
st.markdown("**Generatormatrix Q** (Raten; die Diagonale ist minus die Summe der Zeile):\n\n" + header + body)
rows = []
for n in range(k + 1):
    inflow = (pi[n - 1] * a if n > 0 else 0.0) + (pi[n + 1] * K.service_rate(n + 1, c) if n < k else 0.0)
    outflow = pi[n] * ((a if n < k else 0.0) + K.service_rate(n, c))
    rows.append(f"| {n} | {pi[n]:.5f} | {inflow:.5f} | {outflow:.5f} | {abs(inflow - outflow):.0e} |")
st.markdown("**Gleichgewichtsgleichungen** (für jeden Zustand: Zufluss = Abfluss):\n\n| Zustand n | π(n) | Zufluss | Abfluss | Differenz |\n|---|---|---|---|---|\n" + "\n".join(rows))
st.info(
    f"Das Gleichgewicht erfüllt die {k + 1} Gleichungen auf Rundungsfehler genau. Daraus folgt: Verlust {C.fmt_pct(m['blocking'], 2)}, mittlere Zahl im System {m['L']:.3f}, Wartezeit der "
    f"Angenommenen {K.to_minutes(m['Wq']):.2f} min. Bei K = c ist das Erlang B (Stück 8)."
)

st.markdown("---")
st.subheader("📐 Drei Wege zum Gleichgewicht")
steps = st.slider("Schritte der Potenzmethode", C.STEPS_MIN, C.STEPS_MAX, C.DEFAULT_STEPS, key="steps_slider",
                  help="Wie oft die uniformisierte Kette auf den Startzustand „leer“ angewandt wird.")
pi_power, dists = K.power_method(q, steps)
_, all_dists = K.power_method(q, C.STEPS_MAX)
pi_sim = _simulated_fractions(c, k, a, seed)
needed = K.steps_to_converge(q)
st.markdown(
    f"Dieselbe Kette, drei Rechenwege: die **direkte Lösung** (Balken), die **Potenzmethode** nach {steps} Schritten (Linie, Start leer) und die **Simulation** (Punkte, {C.fmt_int(int(C.SIM_HORIZON))} Abfertigungsdauern)."
)
col_a, col_b = st.columns(2)
with col_a:
    st.plotly_chart(build_stationary_chart(pi, pi_power, pi_sim, steps), width="stretch", key=f"stat_{c}_{k}_{rho_pct}_{steps}_{seed}")
with col_b:
    st.plotly_chart(build_convergence_chart(all_dists, steps, C.POWER_TOL), width="stretch", key=f"conv_{c}_{k}_{rho_pct}_{steps}")
closed = K.stationary_closed(c, k, a)
st.markdown(
    "| Weg | Abstand zur direkten Lösung (L1) |\n|---|---|\n"
    f"| direkte Lösung gegen geschlossene Formel (Geburts-Sterbe-Kette) | {abs(pi - closed).sum():.0e} |\n"
    f"| Potenzmethode nach {steps} Schritten | {dists[-1]:.1e} |\n"
    f"| Simulation ({C.fmt_int(int(C.SIM_HORIZON))} Abfertigungsdauern) | {abs(pi_sim - pi).sum():.3f} |"
)
st.info(
    f"Die Potenzmethode braucht für diese Kette {C.fmt_int(needed)} Schritte bis zum Abstand {C.POWER_TOL:g}; bei großer Last werden es sehr viele (bei einer Spur und 300 Plätzen 219 / 2.076 / 9.296 / 38.587 Schritte "
    "für Last 50 / 80 / 90 / 95 %). Die Simulation ist die ungenaueste, aber die einzige, die ohne die Kette auskommt."
)

st.markdown("---")
st.subheader("🔬 Wie lange dauert das Einschwingen?")
relax_pct = st.select_slider("Auslastung der M/M/1-Kette", options=C.RELAX_RHO_PCT, value=C.DEFAULT_RELAX_RHO_PCT, key="relax_select", format_func=lambda v: f"{v} %",
                             help="Eine Spur, unbegrenzter Warteraum (bei 400 Zuständen abgeschnitten).")
with st.spinner("Berechne die Einschwingkurve über die Matrixexponentialfunktion …"):
    t95, times, means = _relaxation(relax_pct / 100)
    table = _relaxation_table()
steady = (relax_pct / 100) / (1 - relax_pct / 100)
st.markdown(
    "Die Kette beginnt **leer**; die mittlere Zahl im System steigt dem Gleichgewichtswert entgegen. Wie lange das dauert, ist die **Einschwingzeit**, und sie berechnet sich exakt aus der Kette "
    "(p(t) = p(0)·e^{Qt}), ohne Simulation. Genau diese Zeit müsste ein Simulationslauf als Warm-up verwerfen (Stück 2)."
)
st.plotly_chart(build_transient_chart(relax_pct / 100, times, means, steady, t95), width="stretch", key=f"trans_{relax_pct}")
rows = []
for r_pct, t in table.items():
    rho = r_pct / 100
    rows.append(f"| {r_pct} % | {rho / (1 - rho):.2f} | {t:.1f} | {_time_text(t)} | {K.relaxation_heuristic(rho):.1f} | {t * (1 - rho ** 0.5) ** 2:.2f} |")
st.markdown("| Auslastung | Gleichgewicht L | Zeit bis 95 % (Abfertigungsdauern) | in Zeit (3 min je Dauer) | 1/(1 − √ρ)² | Verhältnis |\n|---|---|---|---|---|---|\n" + "\n".join(rows))
st.info(
    f"Bei Auslastung {relax_pct} % dauert es {t95:.0f} Abfertigungsdauern ({_time_text(t95)}), bis die mittlere Zahl im System 95 % ihres Gleichgewichtswerts erreicht; die Faustformel 1/(1 − √ρ)² trifft die "
    f"Größenordnung (Verhältnis {t95 * (1 - (relax_pct / 100) ** 0.5) ** 2:.2f}). Die Zeit wächst bei Annäherung an volle Auslastung wie (1 − ρ)⁻²."
)

st.markdown("---")
st.subheader("🔬 Wann kommt der erste Verlust?")
hit_a = st.select_slider("Angebot des Verlustsystems (Erlang)", options=C.HIT_A_OPTIONS, value=C.DEFAULT_HIT_A, key="hit_a_select", format_func=lambda v: f"{v:g}",
                         help="Erlang-Verlustsystem ohne Warteraum, Start leer.")
with st.spinner("Löse das Erste-Passage-System …"):
    exact_hit, sim_cs, sim_hit = _hitting(hit_a, seed)
st.markdown(
    "Die **mittlere Zeit bis zum ersten Erreichen** eines Zustands ergibt sich aus den Gleichungen der ersten Schritte (ein lineares System über die übrigen Zustände, das sich Stufe für Stufe lösen lässt): aus dem leeren Gate bis alle c Spuren belegt sind, also bis zum **ersten Verlust**. "
    "Bis zehn Spuren lässt sich das auch simulieren (Punkte, 1000 Läufe); darüber hinaus nicht mehr: das ist das Problem der seltenen Ereignisse aus Stück 9."
)
st.plotly_chart(build_hitting_chart(hit_a, list(C.HIT_C_RANGE), exact_hit, sim_cs, sim_hit), width="stretch", key=f"hit_{hit_a}_{seed}")
rows = []
for cc in (5, 10, 15, 20):
    e = exact_hit[C.HIT_C_RANGE.index(cc)]
    rows.append(f"| {cc} | {C.fmt_sci(e)} | {_time_text(e)} |")
st.markdown("| Spuren | Zeit bis zum ersten Verlust (Abfertigungsdauern) | in Zeit (3 min je Dauer) |\n|---|---|---|\n" + "\n".join(rows))
e5, e20 = exact_hit[C.HIT_C_RANGE.index(5)], exact_hit[C.HIT_C_RANGE.index(20)]
st.info(
    f"Bei Angebot {hit_a:g} dauert es mit 5 Spuren im Mittel {e5:.1f} Abfertigungsdauern bis zum ersten Verlust, mit 20 Spuren {C.fmt_sci(e20)} ({_time_text(e20)}). Die Kette liefert diese Zahl exakt; "
    "eine Simulation würde sie nie erleben."
)

st.markdown("---")
st.subheader("🔬 Nicht exponentielle Dauer: Phasen")
phase_pct = st.select_slider("Auslastung der M/E_k/1-Kette", options=C.PHASE_RHO_PCT, value=C.DEFAULT_PHASE_RHO_PCT, key="phase_rho_select", format_func=lambda v: f"{v} %")
with st.spinner("Löse die Phasen-Ketten …"):
    phases = _phases(phase_pct / 100)
st.markdown(
    "Eine **Erlang-k-Dauer** ist die Summe von k exponentiellen Phasen. Merkt man sich zusätzlich die **Phase** des laufenden Auftrags, wird auch M/E_k/1 zur Markov-Kette, und sie liefert **genau** die Wartezeit der "
    "Pollaczek-Khinchine-Formel für cs² = 1/k (Stück 10). Der Preis: der Zustandsraum wächst mit k."
)
st.plotly_chart(build_phases_chart(phase_pct / 100, [p[0] for p in phases], [p[1] for p in phases], [p[3] for p in phases]), width="stretch", key=f"phases_{phase_pct}")
st.markdown("| Phasen k | cs² = 1/k | Wartezeit (Kette) | Pollaczek-Khinchine | Zustände |\n|---|---|---|---|---|\n" + "\n".join(
    f"| {kk} | {1 / kk:g} | {w:.4f} | {pk:.4f} | {C.fmt_int(n)} |" for kk, w, n, pk in phases))
st.info(
    f"Bei Auslastung {phase_pct} % stimmt die Kette mit der Formel überein (k = 4: {phases[2][1]:.4f} gegen {phases[2][3]:.4f}); der Zustandsraum wächst von {C.fmt_int(phases[0][2])} (k = 1) auf {C.fmt_int(phases[3][2])} (k = 10) Zustände. "
    "Für beliebige Verteilungen reicht das nicht mehr; dafür gibt es die Formeln aus Stück 10."
)

st.markdown("---")
st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Gedächtnislose (exponentielle) Zeiten** | Mit Phasen lassen sich Erlang- und Hyperexponential-Dauern abbilden, der Zustandsraum wächst aber mit jeder Phase; für beliebige Dauern braucht man andere Werkzeuge. | **[M/G/1, Kingman-Näherung](https://sebastianhanisch-mg1-kingman-demo.streamlit.app/)** |
| **Ein Gate** | In Netzen ist der Zustand der Vektor aller Schlangenlängen; der Zustandsraum wächst mit jeder Station, und es gibt Formeln mit Produktform. | **[Jackson-Netze](https://sebastianhanisch-jackson-network-demo.streamlit.app/)** |
| **Eine Klasse von Lkw** | Mit Prioritätsklassen wird der Zustand zweidimensional (Zahl je Klasse), die Kette wird zum Gitter. | **[Prioritätsklassen](https://sebastianhanisch-priority-queue-demo.streamlit.app/)** |
| **Konstante Raten** | Bei zeitabhängigen Raten gibt es kein Gleichgewicht; p(t) folgt dann einer Differentialgleichung mit zeitabhängigem Generator. | **[Zeitvariable Ankünfte](https://sebastianhanisch-time-varying-arrivals-demo.streamlit.app/)** |
| **Der Zustandsraum ist klein genug zum Lösen** | Große oder sehr seltene Zustände lassen sich nicht mehr exakt rechnen (hier ab etwa 20 Spuren für die erste Passage); dann bleibt nur Simulation, mit Tricks. | **[Seltene Ereignisse (Splitting)](https://sebastianhanisch-splitting-demo.streamlit.app/)** |
| **Abgeschnittene Kette** | Die M/M/1-Kette hat unendlich viele Zustände; hier wird sie bei 400 abgeschnitten, der Fehler fällt wie ρ⁴⁰⁰. | kein Folgestück |
"""
)
st.caption(
    "Verwandt im Portfolio: [mm1-queue-demo](https://sebastianhanisch-mm1-queue-demo.streamlit.app/) (Stück 1: die Kette einer Spur), [output-analysis-demo](https://sebastianhanisch-output-analysis-demo.streamlit.app/) "
    "(Stück 2: Warm-up), [mmc-queue-demo](https://sebastianhanisch-mmc-queue-demo.streamlit.app/) (Stück 3: mehrere Spuren), [erlang-b-demo](https://sebastianhanisch-erlang-b-demo.streamlit.app/) (Stück 8: K = c), "
    "[power-of-d-demo](https://sebastianhanisch-power-of-d-demo.streamlit.app/) (Stück 7: eine mehrdimensionale Kette als Referenz) und die Rettungsdienst-Demo "
    "[ems-demo](https://sebastianhanisch-ems-demo.streamlit.app/) (Hypercube-Modell: eine Markov-Kette über alle Belegungszustände)."
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Kette.** Zustand $n \in \{0, \dots, K\}$, Ankunftsrate $a$ für $n < K$, Abgangsrate $\mu_n = \min(n, c)$. Generator $Q$: $q_{n,n+1} = a$, $q_{n,n-1} = \mu_n$, $q_{n,n} = -\sum_{j \ne n} q_{n,j}$.

**Gleichgewicht.** $\pi Q = 0$, $\sum_n \pi_n = 1$. Geburts-Sterbe-Kette: $\pi_n = \pi_0 \prod_{j=1}^{n} a/\mu_j$ (Gleichgewicht von Zufluss und Abfluss zwischen $n$ und $n+1$: $\pi_n a = \pi_{n+1}\mu_{n+1}$).
Kennzahlen: $L = \sum n\pi_n$, $L_q = \sum (n - c)^+\pi_n$, Verlust $\pi_K$, Durchsatz $a(1 - \pi_K)$, $W = L/(a(1 - \pi_K))$ (Little).

**Potenzmethode.** Uniformisierung: $P = I + Q/\Lambda$ mit $\Lambda \ge \max_n |q_{n,n}|$ ist eine stochastische Matrix mit demselben Gleichgewicht; $x_{t+1} = x_t P$ konvergiert gegen $\pi$, die Schrittzahl wächst wie $(1 - \rho)^{-2}$.

**Einschwingen.** $p(t) = p(0)\,e^{Qt}$ (zeitstetig, exakt); Einschwingzeit = erstes $t$ mit $E[N(t)] \ge 0.95\,E[N(\infty)]$. Faustformel: $1/(1 - \sqrt\rho)^2$.

**Erste Passage.** Erwartete Zeit $h_i$ bis zum Erreichen von $c$ aus Zustand $i$: $(a + i)\,h_i = 1 + a\,h_{i+1} + i\,h_{i-1}$ für $i < c$, $h_c = 0$ (lineares System).

**Phasen.** $M/E_k/1$: Zustand $(n, \text{Phase})$, Phasenrate $k$, Ankunft $\rho$; Wartezeit $L_q/\rho$ (Little) $= \rho\,(1 + 1/k)/(2(1 - \rho))$ (Pollaczek-Khinchine mit $c_s^2 = 1/k$).

Implementiert in `mkv_chain.py` (Generator, Lösungswege, Einschwingen, erste Passage, Phasen), `mkv_simulation.py` (Gillespie).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html))."
)
