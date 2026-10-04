import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class ScriptedRng:
    """Liefert vorgegebene Werte statt Zufall (Mini-Instanzen von Hand gerechnet). `expovariate(rate)` ignoriert die Rate und gibt der
    Reihe nach die Werte zurück, `uniform()` ebenso aus einer eigenen Liste."""

    def __init__(self, exp_values=(), uniform_values=()):
        self.exp_values, self.uniform_values = list(exp_values), list(uniform_values)
        self.n_exp = self.n_uniform = 0

    def expovariate(self, rate):
        v = self.exp_values[self.n_exp]
        self.n_exp += 1
        return v

    def uniform(self):
        v = self.uniform_values[self.n_uniform]
        self.n_uniform += 1
        return v


@pytest.fixture
def gillespie_mini():
    """c = 1, K = 2, a = 1: Start leer (n = 0). Verweilzeiten (Skript) 0.5 / 0.25 / 0.25 / 1.0, Entscheidungs-Zufall 0.0 / 0.9 / 0.1 / 0.9.
    Von Hand: n = 0, Gesamtrate 1 (nur Ankunft): 0.5 Zeit, Ankunft → n = 1; Gesamtrate 2 (Ankunft 1, Abgang 1), Zufall 0.9 > 1/2 → Abgang: 0.25 Zeit bei n = 1, dann n = 0;
    Gesamtrate 1: 0.25 Zeit, Zufall 0.1 < 1 → Ankunft, n = 1; Gesamtrate 2: 1.0 Zeit bei n = 1, Zufall 0.9 → Abgang, n = 0. Zeiten: n = 0: 0.5 + 0.25 = 0.75, n = 1: 0.25 + 1.0 = 1.25,
    n = 2: 0; Summe 2.0, Anteile 0.375 / 0.625 / 0."""
    return ScriptedRng(exp_values=[0.5, 0.25, 0.25, 1.0], uniform_values=[0.0, 0.9, 0.1, 0.9])
