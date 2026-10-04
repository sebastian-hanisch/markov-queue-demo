"""Konstanten der Markov-Ketten-Demo: Regler, Voreinstellungen, Studien-Achsen. Zeiteinheit der Rechnung ist die mittlere Abfertigungsdauer (= 1, Abfertigungsrate 1 je Spur);
angezeigt werden Abfertigungsdauern und Minuten bei 3 min Mittel."""


def fmt_int(n):
    """Ganzzahl mit Punkt als Tausendertrenner (10000 -> 10.000)."""
    return f"{n:,}".replace(",", ".")


def fmt_pct(x, digits=0):
    """Anteil als Prozent mit Leerzeichen (0.086 -> "9 %", mit digits=1 "8.6 %")."""
    return f"{x:.{digits}%}".replace("%", " %")


def fmt_sci(x, digits=1):
    """Zahl in Zehnerpotenz-Schreibweise mit Dezimalpunkt (5.46e-07 -> "5.5·10⁻⁷", 193000 -> "1.9·10⁵")."""
    if x == 0:
        return "0"
    mantissa, exponent = f"{x:.{digits}e}".split("e")
    e = int(exponent)
    sup = str(abs(e)).translate(str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return f"{mantissa}·10{'⁻' if e < 0 else ''}{sup}"


MEAN_SERVICE_MIN = 3.0                                  # mittlere Abfertigungsdauer je Spur (Minuten)

C_MIN, C_MAX, DEFAULT_C = 1, 4, 2                       # Spuren der kleinen Kette (Abschnitt 1)
K_MIN, K_MAX, DEFAULT_K = 4, 10, 6                      # Kapazität (Plätze insgesamt, Spuren eingeschlossen); K ≥ 4 ≥ c, damit K nie unter c liegt
RHO_PCT_MIN, RHO_PCT_MAX, RHO_PCT_STEP, DEFAULT_RHO_PCT = 50, 150, 10, 80   # Last je Spur in Prozent (Überlast erlaubt: die Kette ist endlich)
SEED_MAX = 999999
DEFAULT_SEED = 35

STEPS_MIN, STEPS_MAX, DEFAULT_STEPS = 1, 400, 25        # Schritte der Potenzmethode (Abschnitt 1)
SIM_HORIZON = 100_000.0                                  # simulierte Zeit (Abfertigungsdauern) für die Zeitanteile
POWER_TOL = 1e-8                                         # Abbruchschwelle der Potenzmethode (L1-Abstand zur exakten Lösung)

RELAX_RHO_PCT = (50, 80, 90, 95)                        # Abschnitt 2: Auslastung der M/M/1-Kette
DEFAULT_RELAX_RHO_PCT = 90
RELAX_TRUNCATION = 400                                  # Zustände der abgeschnittenen M/M/1-Kette
RELAX_LEVEL = 0.95                                      # Anteil des Gleichgewichtswerts der mittleren Zahl im System

HIT_A_OPTIONS = (1.0, 2.0, 3.0, 5.0)                    # Abschnitt 3: Angebot (Erlang) des Verlustsystems
DEFAULT_HIT_A = 3.0
HIT_C_RANGE = tuple(range(2, 21))                       # Spuren der Kurve
HIT_SIM_MAX_C = 10                                      # bis hierhin wird die erste Passage auch simuliert
HIT_SIM_RUNS = 2000

PHASE_RHO_PCT = (50, 80, 95)                            # Abschnitt 4: Auslastung der M/E_k/1-Kette
DEFAULT_PHASE_RHO_PCT = 80
PHASE_K_OPTIONS = (1, 2, 4, 10)
DEFAULT_PHASE_K = 4
PHASE_TRUNCATION = 300                                  # Aufträge im System, bei denen die Kette abgeschnitten wird

PRESET_ORDER = ("Eine Spur, 6 Plätze", "Zwei Spuren, 6 Plätze", "Überlast (150 %)", "Nur Spuren (K = c)")


def _preset(c=DEFAULT_C, k=DEFAULT_K, rho_pct=DEFAULT_RHO_PCT):
    return {"c": c, "k": k, "rho_pct": rho_pct, "seed": DEFAULT_SEED}


PRESETS = {
    "Eine Spur, 6 Plätze": _preset(c=1),
    "Zwei Spuren, 6 Plätze": _preset(),
    "Überlast (150 %)": _preset(rho_pct=150),
    "Nur Spuren (K = c)": _preset(c=4, k=4),
}
# Zahlen aus der Kette (gelöst über πQ = 0), 3 min mittlere Abfertigung; tests/test_claims.py rechnet jede nach
PRESET_HELP = {
    "Eine Spur, 6 Plätze": "Eine Spur, 6 Plätze, Last 80 %: Verlust 6.6 %, im Mittel 2.14 Lkw im System, die Angenommenen warten 5.6 min.",
    "Zwei Spuren, 6 Plätze": "Zwei Spuren, 6 Plätze, Last 80 % (Angebot 1.6 Erlang): Verlust 7.6 %, im Mittel 2.45 Lkw im System, die Angenommenen warten 2.0 min.",
    "Überlast (150 %)": "Zwei Spuren, 6 Plätze, Last 150 % (Angebot 3 Erlang): Verlust 36.0 %, die Spuren sind zu 96 % ausgelastet; die endliche Kette bleibt lösbar, die Angenommenen warten 4.0 min.",
    "Nur Spuren (K = c)": "Vier Spuren ohne Warteplatz, Last 80 %: Verlust 22.8 %, genau Erlang B (Stück 8); niemand wartet.",
}
