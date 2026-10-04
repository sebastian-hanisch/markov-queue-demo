"""Simulation der Ketten (Gillespie): aus dem Zustand n springt die Kette nach einer exponentiell verteilten Zeit mit der Gesamtrate der Übergänge; welcher Übergang es ist, entscheidet der Anteil seiner Rate.
Zufall nur über einen übergebenen `SplitMix64`-Strom (Verweilzeit, Entscheidung). Zwei Auswertungen: Zeitanteile je Zustand (Gleichgewicht, Zeitmittel) und die Zeit bis zum ersten Erreichen eines Zustands."""

import math

import numpy as np

_MASK = (1 << 64) - 1


class SplitMix64:
    """Kleiner, gut gemischter 64-Bit-Zufallsgenerator (Vigna); reine Ganzzahl-Arithmetik."""

    def __init__(self, seed):
        self.state = seed & _MASK

    def next(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & _MASK
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK
        return z ^ (z >> 31)

    def uniform(self):
        """Gleichverteilt auf [0, 1) mit 53 Bit."""
        return (self.next() >> 11) * (1.0 / (1 << 53))

    def expovariate(self, rate):
        """Exponentiell mit Mittel 1/rate (Inversion; 1 − u liegt in (0, 1], der Logarithmus ist endlich)."""
        return -math.log(1.0 - self.uniform()) / rate


def rates(n, c, k, a):
    """(Ankunftsrate, Abgangsrate) im Zustand n der M/M/c/K-Kette."""
    return (a if n < k else 0.0), float(min(n, c))


def jump(n, c, k, a, rng):
    """Ein Sprung aus Zustand n: Verweilzeit (exponentiell mit der Gesamtrate) und Folgezustand (Ankunft mit Wahrscheinlichkeit Ankunftsrate/Gesamtrate). Gibt (Zeit, neuer Zustand) zurück."""
    up, down = rates(n, c, k, a)
    total = up + down
    dt = rng.expovariate(total)
    return dt, (n + 1 if rng.uniform() < up / total else n - 1)


def time_fractions(c, k, a, horizon, rng, start=0):
    """Zeitanteile je Zustand 0 … K eines Laufs über die simulierte Zeit `horizon` (Start im Zustand `start`; der letzte Sprung wird abgeschnitten)."""
    spent = np.zeros(k + 1)
    t, n = 0.0, start
    while t < horizon:
        dt, nxt = jump(n, c, k, a, rng)
        spent[n] += min(dt, horizon - t)
        t += dt
        n = nxt
    return spent / spent.sum()


def first_loss_time(c, a, rng):
    """Zeit bis zum ersten Erreichen von c Belegten im Erlang-Verlustsystem (Start leer): Zustand n < c mit Ankunftsrate a und Abgangsrate n."""
    t, n = 0.0, 0
    while n < c:
        dt, n = jump(n, c, c + 1, a, rng)         # Kapazität c + 1 genügt: bei n < c gibt es immer eine Ankunft; der Zustand c wird nicht verlassen
        t += dt
    return t


def mean_first_loss_time(c, a, runs, rng):
    """Mittelwert von `runs` unabhängigen Zeiten bis zum ersten Verlust."""
    return sum(first_loss_time(c, a, rng) for _ in range(runs)) / runs
