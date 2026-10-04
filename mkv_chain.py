"""Warteschlangen als zeitstetige Markov-Ketten (CTMC): Generator, drei Wege zum Gleichgewicht, Kennzahlen, Einschwingzeit, erste Passage, Phasen.

**Die Kette.** Zustand n = Zahl der Lkw im System (0 … K). Ankunftsrate a (Angebot in Erlang, mittlere Abfertigungsdauer 1), Abgangsrate μ_n = min(n, c). Der Generator Q hat
q[n, n+1] = a (n < K), q[n, n−1] = μ_n, die Diagonale macht jede Zeile zur Summe null. Das Gleichgewicht π erfüllt πQ = 0, Σπ = 1; es gibt es als **Geburts-Sterbe-Kette** auch
in geschlossener Form (π_n ∝ a^n/n! für n ≤ c, danach mit (a/c)^{n−c}).

**Drei Wege zum Gleichgewicht.** (1) Direkt: lineares System (eine Gleichung durch die Normierung ersetzt). (2) Potenzmethode: die uniformisierte Kette P = I + Q/Λ (Λ ≥ größte Austrittsrate)
wird so lange auf eine Startverteilung angewandt, bis sich nichts mehr ändert; die Schrittzahl wächst wie (1 − ρ)⁻². (3) Simulation: Zeitanteile eines langen Laufs (Gillespie, `mkv_simulation.py`).

**Weitere Größen aus derselben Kette.** Einschwingzeit: p(t) = p(0)·e^{Qt} (exakt über `expm_multiply`), Zeit bis E[N(t)] 95 % des Gleichgewichtswerts erreicht (Start leer). Erste Passage: erwartete Zeit bis zum
Erreichen eines Zustands = Lösung eines linearen Systems über die übrigen Zustände (Stufe für Stufe stabil lösbar; dicht gelöst verliert es für große c Stellen). Phasen: Eine Erlang-k-Dauer ist die Summe von k exponentiellen Phasen; der Zustand (n, Phase) macht M/E_k/1 zur Markov-Kette."""

import math

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags, identity
from scipy.sparse.linalg import expm_multiply, spsolve

from mkv_constants import POWER_TOL, RELAX_LEVEL, RELAX_TRUNCATION


def service_rate(n, c):
    """Abgangsrate im Zustand n: min(n, c) (jede belegte Spur mit Rate 1)."""
    return float(min(n, c))


def generator(c, k, a):
    """Generator Q der M/M/c/K-Kette als dichte Matrix (K + 1 Zustände)."""
    if c < 1 or k < c or a <= 0:
        raise ValueError("c ≥ 1, K ≥ c, a > 0 erwartet")
    q = np.zeros((k + 1, k + 1))
    for n in range(k + 1):
        if n < k:
            q[n, n + 1] = a
        if n > 0:
            q[n, n - 1] = service_rate(n, c)
        q[n, n] = -q[n].sum()
    return q


def stationary_direct(q):
    """Gleichgewicht aus πQ = 0, Σπ = 1: Qᵀπ = 0 mit der ersten Gleichung durch die Normierung ersetzt (dichte Matrix oder dünn besetzt)."""
    n = q.shape[0]
    if isinstance(q, np.ndarray):
        a = q.T.copy()
        a[0, :] = 1.0
        b = np.zeros(n)
        b[0] = 1.0
        return np.linalg.solve(a, b)
    a = csr_matrix(q).T.tolil()
    a[0, :] = 1.0
    b = np.zeros(n)
    b[0] = 1.0
    return spsolve(a.tocsr(), b)


def stationary_closed(c, k, a):
    """Gleichgewicht der Geburts-Sterbe-Kette in geschlossener Form: π_n ∝ Π_{j≤n} a/μ_j."""
    w = [1.0]
    for n in range(1, k + 1):
        w.append(w[-1] * a / service_rate(n, c))
    total = sum(w)
    return np.array([x / total for x in w])


def uniformized(q, margin=1.0):
    """Uniformisierte Kette P = I + Q/Λ mit Λ = margin · größte Austrittsrate (Zeilensumme 1, nichtnegativ)."""
    lam = margin * float((-np.diag(q)).max())
    return np.eye(q.shape[0]) + q / lam, lam


def power_method(q, steps, start=0):
    """Verteilung nach `steps` Schritten der Potenzmethode auf der uniformisierten Kette, Start im Zustand `start`; gibt (Verteilung, Liste der L1-Abstände zur exakten Lösung je Schritt) zurück."""
    p, _ = uniformized(q)
    pi = stationary_direct(q)
    x = np.zeros(q.shape[0])
    x[start] = 1.0
    dists = [float(np.abs(x - pi).sum())]
    for _ in range(steps):
        x = x @ p
        dists.append(float(np.abs(x - pi).sum()))
    return x, dists


def steps_to_converge(q, tol=POWER_TOL, start=0, max_steps=5_000_000):
    """Schritte der Potenzmethode, bis der L1-Abstand zur exakten Lösung unter `tol` liegt (dünn besetzt für große Ketten)."""
    qs = csr_matrix(q)
    lam = float(-qs.diagonal().min())
    pt = (identity(qs.shape[0], format="csr") + qs / lam).T.tocsr()
    pi = stationary_direct(qs)
    x = np.zeros(qs.shape[0])
    x[start] = 1.0
    it = 0
    while np.abs(x - pi).sum() > tol and it < max_steps:
        x = pt @ x
        it += 1
    return it


def metrics(c, k, a, pi=None):
    """Kennzahlen aus dem Gleichgewicht: mittlere Zahl im System L, in der Schlange Lq, Verlust (π_K), Durchsatz a(1 − π_K), Verweilzeit W = L/Durchsatz (Little), Wartezeit Wq = Lq/Durchsatz, Auslastung."""
    pi = stationary_closed(c, k, a) if pi is None else pi
    n = np.arange(k + 1)
    lq = float((np.maximum(n - c, 0) * pi).sum())
    block = float(pi[-1])
    thr = a * (1.0 - block)
    busy = float((np.minimum(n, c) * pi).sum())
    return {"L": float((n * pi).sum()), "Lq": lq, "blocking": block, "throughput": thr, "W": float((n * pi).sum()) / thr, "Wq": lq / thr, "utilisation": busy / c}


def mm1_truncated(rho, size=RELAX_TRUNCATION):
    """M/M/1-Kette mit `size` Zuständen als dünn besetzter Generator (Abschneiden bei size − 1)."""
    n = size
    up = np.full(n - 1, float(rho))
    down = np.ones(n - 1)
    q = diags([up, down], [1, -1], shape=(n, n), format="lil")
    q.setdiag(0)
    q = q.tocsr()
    out = np.asarray(q.sum(axis=1)).ravel()
    return (q - diags(out)).tocsr()


def transient_mean(rho, times, size=RELAX_TRUNCATION):
    """Mittlere Zahl im System E[N(t)] der M/M/1-Kette (Start leer) zu den Zeiten `times`, exakt über die Matrixexponentialfunktion (expm_multiply)."""
    q = mm1_truncated(rho, size)
    x0 = np.zeros(size)
    x0[0] = 1.0
    n = np.arange(size)
    return [float((expm_multiply(q.T * t, x0) * n).sum()) for t in times]


def relaxation_time(rho, level=RELAX_LEVEL, size=RELAX_TRUNCATION):
    """Zeit (Abfertigungsdauern), bis die mittlere Zahl im System der M/M/1-Kette von leer aus `level` des Gleichgewichtswerts ρ/(1 − ρ) erreicht (Bisektion über expm_multiply)."""
    steady = rho / (1.0 - rho)
    lo, hi = 0.0, 1.0
    while transient_mean(rho, [hi], size)[0] < level * steady:
        hi *= 2
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if transient_mean(rho, [mid], size)[0] < level * steady:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def relaxation_heuristic(rho):
    """Faustformel für die Einschwingzeit: 1/(1 − √ρ)² (Größenordnung der Relaxationszeit der M/M/1-Kette)."""
    return 1.0 / (1.0 - math.sqrt(rho)) ** 2


def hitting_time_dense(c, a):
    """Erste Passage als dichtes lineares System über die Zustände 0 … c − 1 (Gegenprobe für kleine c): (a + i)·h_i = 1 + a·h_{i+1} + i·h_{i−1}, h_c = 0.
    Die Matrix ist für große c schlecht konditioniert (Einträge von sehr verschiedener Größe); ab etwa c = 12 verliert die Lösung Stellen, ab c = 20 kann sie sogar negativ werden."""
    if c < 1 or a <= 0:
        raise ValueError("c ≥ 1 und a > 0 erwartet")
    m = np.zeros((c, c))
    b = -np.ones(c)
    for i in range(c):
        m[i, i] = -(a + i)
        if i + 1 < c:
            m[i, i + 1] = a
        if i > 0:
            m[i, i - 1] = float(i)
    return float(np.linalg.solve(m, b)[0])


def hitting_time_loss(c, a):
    """Erwartete Zeit bis zum ersten Erreichen von c Belegten (also dem ersten Verlust) aus dem leeren Erlang-Verlustsystem. Dieselben Gleichungen der ersten Schritte wie im dichten System,
    aber Stufe für Stufe gelöst: d_i = Zeit von i nach i + 1 erfüllt d_0 = 1/a und d_i = (1 + i·d_{i−1})/a (nur positive Summanden, daher auch für große c stabil); h_0 = Σ d_i."""
    if c < 1 or a <= 0:
        raise ValueError("c ≥ 1 und a > 0 erwartet")
    d, total = 0.0, 0.0
    for i in range(c):
        d = (1.0 + i * d) / a
        total += d
    return total


def mek1_chain(rho, k, size=300):
    """M/E_k/1 als Markov-Kette: Zustand (n, Phase des laufenden Auftrags), n ≤ `size`; Phasenrate k (Mittel der Dauer 1), Ankunftsrate ρ. Gibt (dünn besetzter Generator, Zahl der Zustände) zurück."""
    index = {(0, 0): 0}
    for n in range(1, size + 1):
        for ph in range(k):
            index[(n, ph)] = len(index)
    rows, cols, vals = [], [], []
    for (n, ph), i in index.items():
        if n < size:
            rows.append(i)
            cols.append(index[(n + 1, ph if n > 0 else 0)])
            vals.append(float(rho))
        if n > 0:
            target = index[(n, ph + 1)] if ph + 1 < k else index[(n - 1, 0)]
            rows.append(i)
            cols.append(target)
            vals.append(float(k))
    q = coo_matrix((vals, (rows, cols)), shape=(len(index), len(index))).tocsr()
    out = np.asarray(q.sum(axis=1)).ravel()
    return (q - diags(out)).tocsr(), index


def mek1_wait(rho, k, size=300):
    """Wartezeit der M/E_k/1-Kette über Little: Lq/ρ; gibt (Wartezeit, Zahl der Zustände) zurück."""
    q, index = mek1_chain(rho, k, size)
    pi = stationary_direct(q)
    lq = sum(pi[i] * max(n - 1, 0) for (n, _), i in index.items())
    return float(lq / rho), len(index)


def pk_wait(rho, scv):
    """Pollaczek-Khinchine: ρ(1 + cs²)/(2(1 − ρ)) bei mittlerer Dauer 1."""
    if not 0 < rho < 1:
        raise ValueError("ρ muss in (0, 1) liegen")
    return rho * (1.0 + scv) / (2.0 * (1.0 - rho))


def erlang_b(c, a):
    """Erlang-B-Verlustwahrscheinlichkeit über die stabile Rekursion (Kopie aus erlang-b-demo)."""
    b = 1.0
    for j in range(1, c + 1):
        b = a * b / (j + a * b)
    return b


def to_minutes(x, mean_service_min=3.0):
    """Zeit in Abfertigungsdauern → Minuten."""
    return x * mean_service_min
