# Warteschlangen als Markov-Ketten (Streamlit-Demo)

**[→ Demo live ausprobieren](https://sebastianhanisch-markov-queue-demo.streamlit.app/)**

---

Interaktive Einführung in das Rechenwerkzeug hinter den Formeln der Linie: **Zusatzstück der Konzepte-Linie „Warteschlangentheorie und Simulation“** im Portfolio von
[Sebastian Hanisch](https://sebastianhanisch.net) (Operations Research und Machine Learning). Das Stück ist nicht nummeriert; es liefert die Grundlage zu
[mm1-queue-demo](https://github.com/sebastian-hanisch/mm1-queue-demo) (Stück 1) und [mmc-queue-demo](https://github.com/sebastian-hanisch/mmc-queue-demo) (Stück 3) und ist der Bezug für alle späteren Ketten der Linie.

Fast jede Formel der Linie ist das Ergebnis einer **Markov-Kette**: Zustand = Zahl der Lkw im Gate, Pfeile mit Raten zu den Nachbarn, daraus die Generatormatrix Q und das Gleichgewicht πQ = 0. Die Demo macht diesen Weg sichtbar
und beantwortet drei Fragen, die die Formeln offen lassen: **Wie lange dauert das Einschwingen?** (das ist die Warm-up-Frage aus Stück 2), **Wann kommt der erste Verlust?** (das ist die Frage der seltenen Ereignisse aus Stück 9)
und **was tun, wenn die Dauer nicht exponentiell ist?**

## Kernfrage

Wie wird aus einem Gate eine lösbare Kette, auf welchen Wegen kommt man zum Gleichgewicht, und wo stößt die Methode an ihre Grenze?

## Modell und Methodik

- **Kette** (`mkv_chain.py`): Zustand n = Zahl der Lkw im System (0 bis K, K = Spuren plus Warteplätze), Ankunftsrate a (Angebot in Erlang, mittlere Abfertigungsdauer = 1, im Text 3 min), Abgangsrate min(n, c). Der Generator Q
  hat q[n, n+1] = a (n < K) und q[n, n−1] = min(n, c); die Diagonale macht jede Zeile zur Summe null. Kennzahlen aus dem Gleichgewicht: Verlust π_K, mittlere Zahl im System, Wartezeit der Angenommenen nach Little (Lq geteilt durch
  a(1 − π_K)), Auslastung.
- **Drei Wege zum Gleichgewicht:** (1) **direkt** als lineares System (eine Gleichung durch die Normierung ersetzt); (2) **Potenzmethode** auf der uniformisierten Kette P = I + Q/Λ (Start leer); (3) **Simulation**
  (`mkv_simulation.py`, Gillespie: exponentielle Verweilzeit mit der Gesamtrate, der Übergang wird nach seinem Anteil an der Rate gewählt) über 100 000 Abfertigungsdauern. Dazu die geschlossene Form der Geburts-Sterbe-Kette als Gegenprobe.
- **Einschwingzeit:** M/M/1 von leer aus, abgeschnitten bei 400 Zuständen; p(t) = p(0)·e^{Qt} exakt über `scipy.sparse.linalg.expm_multiply`; Einschwingzeit = Zeit, bis die mittlere Zahl im System 95 % ihres Gleichgewichtswerts erreicht (Bisektion).
- **Erste Passage:** mittlere Zeit vom leeren Erlang-Verlustsystem bis zum ersten Verlust (alle c Spuren belegt) aus den Gleichungen der ersten Schritte. Gelöst wird Stufe für Stufe: d_i = Zeit von i nach i + 1,
  d_0 = 1/a, d_i = (1 + i·d_{i−1})/a, Summe über alle Stufen; das dichte lineare System dient als Gegenprobe für kleine c (Befunde unten).
- **Phasen:** M/E_k/1 als Kette mit Zustand (n, Phase des laufenden Auftrags), Phasenrate k; die Wartezeit nach Little ist die der Pollaczek-Khinchine-Formel (Stück 10) für cs² = 1/k.
- **Gegenproben:** (1) Generator, Gleichgewicht und Erste-Passage-Zeiten von Hand (c = 1, 2, 3); (2) direkte Lösung gegen geschlossene Form und πQ = 0 auf 10⁻¹² genau; (3) Potenzmethode konvergiert monoton gegen die direkte Lösung;
  (4) Einschwingen gegen eine **unabhängige Runge-Kutta-Integration** von dp/dt = pQ; (5) Erste Passage gegen **exakte Bruchrechnung** (Python `Fraction`) und gegen die Simulation; (6) die Gillespie-Einheit von Hand mit
  skriptgesteuertem Zufall; (7) M/E_k/1-Kette gegen Pollaczek-Khinchine auf 10⁻⁶; (8) K = c gegen Erlang B (Stück 8).

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py`. Zeiten in Abfertigungsdauern, wo nicht Minuten dasteht (3 min Mittel).

| Frage | Befund |
|---|---|
| Voreinstellung der App (zwei Spuren, 6 Plätze, Last 80 %)? | Verlust **7.60 %**, im Mittel 2.45 Lkw im System, die Angenommenen warten **1.98 min**, Auslastung der Spuren 73.9 %. |
| Die vier Voreinstellungen? | Eine Spur, 6 Plätze, 80 %: Verlust 6.6 %, L = 2.14, Wartezeit 5.6 min. Überlast (150 %, zwei Spuren): Verlust 36.0 %, Spuren zu 96 % ausgelastet, Wartezeit 4.0 min. Vier Spuren ohne Warteplatz: Verlust **22.8 %**, genau Erlang B, niemand wartet. |
| Wie viele Schritte braucht die Potenzmethode? | Eine Spur, 300 Plätze, bis der L1-Abstand unter 10⁻⁸ liegt: **219 / 2 076 / 9 296 / 38 587 Schritte** bei Last 50 / 80 / 90 / 95 %; die Zahl wächst wie (1 − ρ)⁻². Die Voreinstellung (zwei Spuren, 6 Plätze) braucht 149 Schritte. |
| Wie gut trifft die Simulation? | Voreinstellung, Seed 35, 100 000 Abfertigungsdauern: L1-Abstand zur direkten Lösung 0.0087; die direkte Lösung weicht von der geschlossenen Form um weniger als 10⁻¹⁵ ab. |
| Wie lange dauert das Einschwingen? | M/M/1 von leer aus, bis 95 % des Gleichgewichts: **14.9 / 102.1 / 418.9 / 1 696.3 Abfertigungsdauern** bei Last 50 / 80 / 90 / 95 %, also rund 45 min / 5 h / 21 h / 3.5 Tage. Die Faustformel 1/(1 − √ρ)² trifft die Größenordnung (11.7 / 89.7 / 379.7 / 1 559.7; Verhältnis 1.28 / 1.14 / 1.10 / 1.09). |
| Wann kommt der erste Verlust? | Angebot 3 Erlang: mit 5 Spuren nach **4.0**, 8 Spuren **28.7**, 10 Spuren **192.9**, 15 Spuren **156 329** (rund 11 Monate bei 3 min) und 20 Spuren **8.3·10⁸** Abfertigungsdauern (rund 4 760 Jahre). Die Simulation bestätigt die Werte für 2 Spuren (Angebot 1) und 5 Spuren (Angebot 3) mit je 4 000 Läufen auf unter 6 % genau. |
| Was die direkte Lösung des linearen Systems nicht kann | Dichte Lösung der ersten Passage bei Angebot 1 und 20 Spuren: Betrag um 10¹⁷, aber mehr als 100 % neben dem richtigen Wert **3.5·10¹⁷** (auf dem Entwicklungsrechner sogar negativ; die Matrix ist schlecht konditioniert); für 15 Spuren Fehler um 10⁻⁶; bei Angebot 3 und 20 Spuren noch 10⁻⁸ bis 10⁻⁷. Die genauen Stellen hängen von der LAPACK-Bibliothek ab. Die Stufenrekursion mit nur positiven Summanden stimmt mit der exakten Bruchrechnung auf 10⁻¹⁵ überein. |
| Phasen: stimmt die Kette mit der Formel? | M/E_k/1, Last 80 %, k = 1 / 2 / 4 / 10: Wartezeit **4.0 / 3.0 / 2.5 / 2.2** Abfertigungsdauern, genau die Pollaczek-Khinchine-Werte; der Zustandsraum wächst mit k auf **301 / 601 / 1 201 / 3 001** Zustände (Abschneiden bei 300 Aufträgen). |

## Befunde und Korrekturen gegenüber der Vorab-Messreihe

- **Die Erste-Passage-Rechnung war zunächst falsch, ohne dass es auffiel.** Die erste Fassung löste das lineare System dicht. Bei der Voreinstellung (Angebot 3) war das richtig, bei Angebot 1 und 2 nicht: Bei Angebot 1 und
  20 Spuren kam sogar eine negative Zeit heraus. Aufgefallen ist es beim Nachrechnen der README-Zahlen für alle Angebote. Die App löst jetzt Stufe für Stufe; ein Test vergleicht alle Kombinationen aus 2 bis 20 Spuren und Angebot 1 bis 5 mit exakter Bruchrechnung.
- **Die Einschwingzeit ist keine Zufallsgröße.** Stück 2 schätzt das Warm-up aus Läufen; hier ergibt es sich exakt aus der Kette, und der exakte Wert liegt bei Last 50 bis 95 % um 9 bis 28 % über der Faustformel 1/(1 − √ρ)².

## Ehrliche Grenzen

- **Nur exponentielle Zeiten.** Phasen bilden Erlang-Dauern ab (Gegenprobe oben), der Zustandsraum wächst aber mit jeder Phase; beliebige Dauern behandelt Stück 10 mit Formeln.
- **Ein Gate.** Netze haben als Zustand den Vektor aller Schlangenlängen ([Jackson-Netze](https://github.com/sebastian-hanisch/jackson-network-demo)).
- **Eine Klasse.** Mit Prioritäten wird der Zustand ein Gitter (Stück 11).
- **Konstante Raten.** Bei zeitabhängiger Ankunftsrate gibt es kein Gleichgewicht (Stück 6).
- **Zustandsraum klein genug zum Lösen.** Die erste Passage lässt sich bis 20 Spuren exakt rechnen, aber nicht simulieren; wie man seltene Ereignisse trotzdem simuliert, zeigt Stück 9.
- **Abgeschnittene Kette.** Die M/M/1-Kette hat unendlich viele Zustände und wird bei 400 abgeschnitten (Fehler ρ⁴⁰⁰); die Einschwingzeiten gelten für diese abgeschnittene Kette.
- Die Simulation der ersten Passage läuft nur für gerade Spurzahlen bis 10 (und nur, wo die Zeit unter 250 Abfertigungsdauern liegt), damit die App schnell bleibt.

## Verwandte Demos im Portfolio

- [`mm1-queue-demo`](https://github.com/sebastian-hanisch/mm1-queue-demo) (Stück 1) und [`mmc-queue-demo`](https://github.com/sebastian-hanisch/mmc-queue-demo) (Stück 3): die Ketten einer und mehrerer Spuren.
- [`output-analysis-demo`](https://github.com/sebastian-hanisch/output-analysis-demo) (Stück 2): Warm-up aus Läufen, hier exakt.
- [`erlang-b-demo`](https://github.com/sebastian-hanisch/erlang-b-demo) (Stück 8): der Fall K = c.
- [`splitting-demo`](https://github.com/sebastian-hanisch/splitting-demo) (Stück 9): seltene Ereignisse.
- [`mg1-kingman-demo`](https://github.com/sebastian-hanisch/mg1-kingman-demo) (Stück 10): Pollaczek-Khinchine, hier als Phasen-Kette bestätigt.
- [`priority-queue-demo`](https://github.com/sebastian-hanisch/priority-queue-demo) (Stück 11): zweidimensionale Kette.
- [`time-varying-arrivals-demo`](https://github.com/sebastian-hanisch/time-varying-arrivals-demo) (Stück 6) und [`power-of-d-demo`](https://github.com/sebastian-hanisch/power-of-d-demo) (Stück 7).

## Bewusst nicht umgesetzt

Jede dieser Annahmen hebt ein Folgestück der Linie auf:

| Annahme | Folgestück |
|---|---|
| Gedächtnislose Zeiten | [M/G/1, Kingman-Näherung](https://github.com/sebastian-hanisch/mg1-kingman-demo) |
| Ein Gate | [Jackson-Netze](https://github.com/sebastian-hanisch/jackson-network-demo) |
| Eine Klasse | [Prioritätsklassen](https://github.com/sebastian-hanisch/priority-queue-demo) |
| Konstante Raten | [Zeitvariable Ankünfte](https://github.com/sebastian-hanisch/time-varying-arrivals-demo) |
| Der Zustandsraum ist klein genug | [Seltene Ereignisse (Splitting)](https://github.com/sebastian-hanisch/splitting-demo) |

## Tests

123 Tests, rund 90 Sekunden: Generator, Gleichgewicht, Kennzahlen und Erste-Passage-Zeiten von Hand, direkte Lösung gegen geschlossene Form, Gleichgewichtsgleichungen und Little, Erlang B als Sonderfall, Potenzmethode, Einschwingen
gegen Runge-Kutta, Erste Passage gegen Bruchrechnung und das dichte System, Phasen-Kette gegen Pollaczek-Khinchine, Gillespie-Einheiten mit skriptgesteuertem Zufall, Simulation gegen die Kette, Presets und Permalink, Diagramme
(gesperrte Achsen), AppTest-Rauchtests mit festem Würfel-Seed, der Smoke-Test der Portfolio-Vorlage, ein Quelltext-Test gegen Satz-Komma-Fehler und `test_claims.py` für jede Zahl dieser README.

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche |
| `mkv_chain.py` | Generator, Gleichgewicht (direkt, geschlossen, Potenzmethode), Kennzahlen, Einschwingen, erste Passage, Phasen |
| `mkv_simulation.py` | SplitMix64, Gillespie-Sprung, Zeitanteile, erste Passage |
| `mkv_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `mkv_presets.py`, `mkv_constants.py` | Presets, Permalink, Grenzen, Formatierer |
| `tests/` | siehe oben |

## Literatur

- Gillespie, D. T. (1977): Exact stochastic simulation of coupled chemical reactions. *The Journal of Physical Chemistry* 81(25), 2340–2361 (die Simulationsmethode für Markov-Sprungprozesse, hier auf Warteschlangen angewandt).
- Jensen, A. (1953): Markoff chains as an aid in the study of Markoff processes. *Skandinavisk Aktuarietidskrift* 36, 87–91 (Ursprung der Uniformisierung; als Rechenmethode für Übergangswahrscheinlichkeiten oft Grassmann 1977 zugeschrieben).

## Lokal ausführen

```
pip install -r requirements.txt
streamlit run app.py
```

Tests: `pip install -r requirements-dev.txt` und `python -m pytest tests/ -v`.

Gebaut mit Streamlit und Plotly (die Rechnung nutzt scipy).
