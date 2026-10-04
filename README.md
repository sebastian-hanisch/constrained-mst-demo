# Beschränkte Spannbäume – Bottleneck, Knotengrad, Hops – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-constrained-mst-demo.streamlit.app/)**

Sechstes Stück der **Spannbaum-Reihe** der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations Research und Machine Learning". Der minimale Spannbaum ist das Optimum ohne Nebenbedingung. Reale Netze haben Grenzen: ein **Verteiler hat nur Δ Anschlüsse** (Knotengrad), ein **Signal darf nur H Sprünge vom Depot** laufen (Hops), das **schwächste Glied** - die längste Leitung - soll kurz sein (Bottleneck). Der MST ist für den Bottleneck schon optimal, aber weder grad- noch hop-optimal; die beiden letzten Nebenbedingungen machen das Problem **NP-schwer** (Grad Δ = 2 ist ein Hamiltonpfad). Die Demo misst, **was jede Grenze kostet**, ob die einfachen Greedy-Verfahren überhaupt einen gültigen Baum finden, wie nah eine **Lagrange-Untergrenze** (Knotenstrafen, Volgenant 1989) und ein exaktes **Branch-and-Bound** (nur kleine Instanzen) an den besten Baum herankommen und was die Hop-Grenze am **Umweg** ändert. Der Kruskal-Baum aus [kruskal-demo](../kruskal-demo) ist der Ausgangspunkt.

**Einordnung in die Reihe:** geplant sind elf Stücke, dies ist das sechste:

```
Kruskal (Wurzel)                                                                           [gebaut: kruskal-demo]
 ├─ Prim (Kontrast: wächst von einem Punkt)                                                [gebaut: prim-demo]
 ├─ Borůvka (Kontrast: alle Komponenten parallel)                                          [gebaut: boruvka-demo]
 ├─ Euklidischer MST (keine n²-Kantenliste, Delaunay)                                      [gebaut: euclidean-mst-demo]
 ├─ Gerichteter Spannbaum (Chu-Liu/Edmonds)                                                [gebaut: arborescence-demo]
 ├─ Bottleneck-/Grad-/Hop-beschränkter Spannbaum                                           [DIESES STÜCK]
 │    └─ Kapazitierter MST                                                                 [gebaut: cmst-demo]
 ├─ Steiner-Baum → Prize-Collecting Steiner-Baum                                           [gebaut: steiner-tree-demo, pcst-demo]
 ├─ MST-Sensitivität & dynamischer MST                                                     [gebaut: mst-sensitivity-demo]
 └─ Zufällige Spannbäume & Kirchhoff                                                       [gebaut: random-spanning-tree-demo]
```

Ergebnis in Kürze: **Der MST ist bottleneck-optimal, aber "bottleneck-optimal" sagt fast nichts (es gibt auf 20 Filialen im Median 10^10,6 solche Bäume, der teuerste kostet 74 % mehr). Die Gradgrenze kostet auf gleichverteilten Instanzen erst bei Δ = 2 etwas (+7,29 % bei 20 Filialen), in Ortschaften schon bei Δ = 3 (+1,06 %). Die Hop-Grenze ist teurer und stark nichtlinear (12 Filialen: H = 6 +1,4 %, H = 3 +15,1 %, H = 1 +134 %). Kruskal/Prim mit Grenze scheitern oft, obwohl ein gültiger Baum existiert (Δ = 2: 20 % der Instanzen, Prim mit Tiefengrenze bei H = 4 auf 20 Filialen 80 %); Lagrange und der Schichtenbaum fanden in allen gemessenen Konfigurationen einen Baum, wo ein anderes Verfahren einen fand.** Die Lagrange-Untergrenze schließt die Lücke bei kleinen Instanzen ganz.

| Frage | Ergebnis (Karte gleichverteilt, k = 6 nächste Nachbarn, ohne Geländezuschlag, sofern nicht anders angegeben; **Median** über 5 feste Instanzen, Seeds 100000–100004; vollständig deterministisch) |
|---|---|
| **Ist der MST bottleneck-optimal?** | ✅ ja, direkt geprüft: gleich dem Minimum über **alle** Spannbäume (Brute-Force, 200 Kleingraphen mit Gleichständen) und gleich der kleinsten Schwelle, ab der der Graph zusammenhängt; der MST-Pfad zwischen zwei Knoten ist ein Minimax-Pfad |
| **Wie viele bottleneck-optimale Bäume gibt es?** | Genau die Spannbäume des Graphen aller Kanten ≤ b* (Brute-Force-Mengengleichheit; Zahl per Kirchhoff, Matrix-Baum-Satz). n = 10/20/30/60: **10^3,2 / 10^10,6 / 10^14,2 / 10^37,3** Bäume; der teuerste kostet **+45,6/+74,2/+64,4/+100,8 %** mehr als der MST, ein per Zufallsreihenfolge gezogener +21,1/+41,3/+30,2/+52,2 %. Mit gerundeten Kosten oder Geländezuschlag bleibt das Bild (n = 20: 10^10,6 und +74,5 % bzw. 10^8,0 und +62,0 %) |
| **Was kostet die Gradgrenze?** (20 Filialen) | Δ = 2/3/4/5/6: **+7,29/0/0/0/0 %** (bester Fund; Untergrenze +6,82 %). Der MST hat hier höchstens Grad 3. In **Ortschaften** (Verteiler mit 5 Anschlüssen; MST-Grad bis 5): Δ = 2/3/4/5/6: **+4,58/+1,06/+0,23/0/0 %** |
| **Finden Kruskal/Prim mit Gradgrenze einen Baum?** | ⚠️ oft nicht: bei Δ = 2 scheitern beide in **20 %** der Instanzen (12 und 20 Filialen), in Ortschaften in **60 %**; im 50-Instanzen-Experiment 6 % (12 Filialen), 34 % (20 Filialen), 50 % (Ortschaften). **Lagrange scheitert in keiner dieser Konfigurationen**; bei Δ = 3 in Ortschaften scheitert niemand |
| **Wie nah ist Greedy am Optimum?** (Δ = 2, 20 Filialen) | Greedy mit Kantentausch **+13,7 %** über dem MST, der exakte Wert **+7,29 %** (Aufschlag des Greedy auf den besten Fund: +4,67 %). Die Lagrange-Untergrenze ist bei n = 8/12/16 gleich dem Optimum, bei n = 20 +6,82 % gegen +7,29 % |
| **Wie ändert sich der Preis mit n?** (Δ = 2) | ⚠️ nicht monoton: n = 8/12/16/20/30: **+1,65/+0,66/+6,36/+7,29/+9,09 %** (n = 30 ohne Exaktverfahren, Untergrenze +7,85 %) |
| **Was kostet die Hop-Grenze?** (12 Filialen, k = 6) | H = 2/3/4/5/6/8/10: **+40,6/+15,1/+6,49/+4,45/+1,40/0/0 %** (bester Fund; H = 1 ist bei k = 6 nicht möglich: das Depot erreicht nicht alle direkt). Greedy mit Kantentausch: +42,3/+18,1/+9,4/+5,09/+1,40/0/0 %; bei H = 4 liegt Greedy +9,4 %, exakt +6,49 % |
| **Und mit vollständigem Kandidatengraphen?** (12 Filialen) | H = 1/2/3/4/5/6/8/10: **+133,6/+32,8/+15,1/+6,49/+4,45/+1,40/0/0 %**; der MST ist 8 Hops tief |
| **Was ändert die Hop-Grenze am Umweg?** | Größter Umweg (Weglänge im Baum / Luftlinie): MST **2,25**; beschränkter Baum bei H = 1/2/3/4–6/8: **1,00/2,20/1,59/1,90/2,25** - der Umweg sinkt nur bei sehr engen Grenzen deutlich und nicht monoton |
| **Die Grenze verschiebt auch den Bottleneck** | Längste Kante des beschränkten Baums gegen b*: H = 1/2/3: **+119,7/+77,9/+21,9 %**, ab H = 4 unverändert; Δ = 2 (20 Filialen) +7,55 % |
| **Scheitert Prim mit Tiefengrenze?** | ⚠️ ja: bei H = 4 auf 12 Filialen in 4 % (50 Instanzen), auf **20 Filialen in 60 %** (5 Instanzen: 80 %); der **Schichtenbaum** (Breitensuche) scheitert nie, wo ein Baum existiert (bei H = 3 auf 20 Filialen gibt es in 8 % der Instanzen keinen) |
| **Wächst der Hop-Preis mit n?** (H = 4) | ✅ ja: n = 8/12/16/20/30: **+3,65/+6,49/+11,0/+13,5/+22,9 %** (ab n = 16 nur Heuristik; bei n = 30 gibt es in 40 % der Instanzen keinen Baum mit H = 4) |
| **Wie groß wird der MST-Grad?** | Euklidisch ohne Geländezuschlag: 300 Instanzen mit n = 30: **Grad 3 in 250, Grad 4 in 50** (Höchstgrad ≤ 5 gilt in Allgemeinlage als Satz); mit Geländezuschlag 1,0: Grad 3/4/5/6 in 111/169/19/1 Instanzen; in Ortschaften mit 5 Anschlüssen Grad 4/5 in 3/297 |

## Was die Demo zeigt

1. **Der beschränkte Baum in Aktion** (Schritt-Slider): **Instanz und MST** (Punktgröße wächst mit dem Grad, rote Punkte verletzen die Grenze) → **Die Nebenbedingung** (Bottleneck: Schwellwert-Slider über die j billigsten Kanten, Punktfarbe = Komponente, Zählung der bottleneck-optimalen Bäume; Grad: Grad am Knoten und **Lagrange-Verlauf** mit Untergrenze und bester zulässiger Lösung; Hops: Hop-Tiefe am Knoten, größter Umweg) → **Der beschränkte Baum** (Umschalter Bester Fund / Greedy / Lagrange / Exakt / MST bzw. MST / teuerster / zufälliger bottleneck-optimaler Baum: grün = wie der MST, orange = weicht ab, grau gestrichelt = MST-Kante fehlt; findet ein Verfahren keinen Baum, steht das da; darunter die Mehrkosten als Balken).
2. **Was kostet die Nebenbedingung?** Preis (billigster gültiger Baum gegen den MST), Untergrenze, Greedy, exakt (bewiesen / nicht bewiesen / zu groß); für Hops Tiefe des MST, Umweg, kleinste mögliche Tiefe; für Bottleneck b*, Zahl der optimalen Bäume, Mehrkosten des teuersten und eines zufälligen.
3. **🎲 Machbarkeits-Experiment** (auf Abruf, Grad und Hops): 50 Instanzen - wie oft gibt es einen gültigen Baum, wie oft scheitern die Greedy-Verfahren, wie oft rettet Lagrange.
4. **♾️ Zahl der bottleneck-optimalen Bäume** über n (auf Abruf).
5. **📐 Sweeps** über die Grad-/Hop-Grenze, n, k und (Ortschaften) die Zahl der Anschlüsse: Preis gegen den MST, Ausfall der Greedy-Verfahren, größter Grad bzw. Umweg (5 feste Instanzen, Median, 10.–90. Perzentil-Band).
6. **📏 Grad-Verteilung des MST** über 300 Instanzen (auf Abruf).
7. **🚧 Grenzen:** Tabelle "Annahme – was passiert – wer setzt an".

Regler: Instanz (Karte / **Ortschaften** / **Lehrbuchbeispiel** im Kreuz), **Nebenbedingung** (Bottleneck / Grad / Hops), **Δ** (2–6) bzw. **H** (1–12), Filialen n (5–60), k (3 bis vollständig), Geländezuschlag, Kosten runden, Anschlüsse je Verteiler (nur Ortschaften), Seed (+ 🎲), Baumauswahl. Alle Regler wirken auf Kosten, Kanten oder Grenze; nur die zur Nebenbedingung passende Grenze ist sichtbar. Kein Zufall im Kern (die "zufällige" Bottleneck-Reihenfolge ist eine ganzzahlige Mischfunktion von Seed und Kantenindex, unabhängig von der numpy-Version).

## Messwerte der Presets

| Preset | Instanz | Ergebnis |
|---|---|---|
| Standardfall (Voreinstellung) | 12 Filialen, k = 6, Seed 35, Δ = 2 | MST 215,65 (Grad 3, 9 Hops tief); Pfad-Baum 224,37 (+4,04 %) - Greedy, Lagrange und exakt finden dasselbe; die Lagrange-Untergrenze schließt die Lücke nach 3 Iterationen |
| Greedy scheitert (Grad ≤ 2) | Seed 100001 | Kruskal und Prim mit Gradgrenze finden **keinen** Baum; Lagrange und exakt: 267,66 (+9,06 % gegen den MST 245,43) |
| Ortschaften: Grad ≤ 3 | 20 Filialen, 5 Anschlüsse | MST 189,43 (Grad bis 5); Δ = 3: exakt +1,07 % (= Lagrange-Untergrenze), Greedy +1,89 % |
| Kostenlose Grenze (Grad ≤ 3) | 20 Filialen | MST-Grad 3: die Grenze ändert nichts (+0,00 %) |
| Hop-Grenze H = 3 | 12 Filialen, k = 6 | exakt +18,42 % (bewiesen, 7809 Suchknoten), Greedy +33,91 %; größter Umweg 2,74 → 1,90, aber längste Kante 36,93 → 42,43 |
| Alle direkt ans Depot (H = 1) | vollständiger Graph | Stern 555,17 gegen 215,65 (+157,44 %), Umweg 1,00, Depotgrad 12 |
| Bottleneck: Milliarden Bäume | 20 Filialen | 4,6 · 10^9 Bäume mit b* = 33,30; MST 316,00 der billigste, teuerster +74,57 %, per Zufallsreihenfolge gezogener +49,08 % |
| Lehrbuchbeispiel (Grad ≤ 2) | Kreuz aus 5 Punkten | MST = Stern um C (Grad 4, Kosten 80); Δ = 3: 88,28 (+10,36 %); Δ = 2: 96,57 (+20,71 %); H = 1: 116,57 (+45,71 %) |

Die Einzelinstanz weicht von den Medianen ab - die Mediane sind die belastbaren Zahlen; die Presets prüfen sich zusätzlich über die 5 festen Instanzen gegen eine gemessene Spannweite des Medians der Kennzahl (`tests/test_presets.py`). Das **Lehrbuchbeispiel** ist von Hand nachzurechnen: die Diagonalen haben Länge 20·√2 ≈ 28,28, die gegenüberliegenden Punkte Abstand 40. Ohne Gradgrenze ist der Stern um C (4 · 20) optimal; mit Δ = 3 hängt S statt an C an W (20 + 20 + 20 + 28,28); mit Δ = 2 entsteht ein Pfad.

## Modell und Verfahren

- **Instanz** (`cons_scenario.py`): Depot (Knoten 0, fest links) und n Filialen; Kosten = euklidische Länge · Geländefaktor u ∈ [1, 1 + Zuschlag] je Kante (unabhängig vom Kandidatengraphen); Kandidaten sind die k nächsten Nachbarn je Knoten, bei Bedarf um die kürzeste Zwischenkante ergänzt, bis der Graph zusammenhängt. **Ortschaften:** Verteiler mit `sats` Anschlussnehmern im Kreis (Radius etwa 6). Schlüssel jeder Kante ist (Kosten, Kantenindex): eine strikte Gesamtordnung.
- **Bottleneck** (`cons_algorithm.py`): b* = Länge der letzten Kruskal-Kante; die bottleneck-optimalen Bäume sind die Spannbäume von G≤b*; ihre Zahl per Determinante der reduzierten Laplace-Matrix; billigster (Kruskal), teuerster (Kruskal mit absteigender Ordnung) und ein per Mischfunktion gezogener Baum. Linearzeit-Algorithmen (Camerini 1978, Gabow & Tarjan 1988) sind nicht gebaut.
- **Grad:** `kruskal_capped` / `prim_capped` (können scheitern), `exchange_degree` (Kantentausch mit Gradprüfung, beste Verbesserung, streng fallend), `degree_greedy` (beste der beiden, nach dem Tausch). **Lagrange** (`lagrange_degree`): L(π) = MST(c + π_u + π_v) − Δ·Σπ, Subgradient deg − Δ, Polyak-Schrittweite mit Halbierung bei Stillstand; jeder gefundene Baum mit Grad ≤ Δ (auch Kruskal mit Gradgrenze auf den geänderten Kosten) ist eine zulässige Lösung. **Exakt** (`exact_tree`): Branch-and-Bound über Kanten (nehmen/lassen in Schlüsselreihenfolge), Schranke = bisherige Kanten + Kruskal-Ergänzung auf den Lagrange-Kosten, obere Schranke aus den Heuristiken; bei erschöpftem Suchknoten-Budget (20 000) wird `proved = False` gemeldet.
- **Hops:** `prim_hop` (Prim mit Tiefengrenze, kann scheitern), `layer_tree` (Breitensuche, jeder Knoten an der billigsten Kante zur Schicht darüber; kleinstmögliche Tiefe, scheitert nie, wenn ein Baum existiert), `exchange_hop` (Kantentausch, dabei wird der abgetrennte Teilbaum auch umgedreht; Tiefenprüfung über den größten Abstand im Teilbaum), `hop_greedy` (beide, nach dem Tausch). **Exakt:** dasselbe Branch-and-Bound, zulässig geprüft über die Tiefe der Wurzelkomponente und den Radius der Fragmente ohne Wurzel; als Untergrenze dient nur der MST (Hops haben hier keine Lagrange-Schranke).
- **Exakt wird angeboten** bis n = 24 (Grad) bzw. n = 12 (Hops) und höchstens 300 Kandidatenkanten.
- **Umweg** = Weglänge im Baum vom Depot / Luftlinie je Filiale (Median und Maximum).

## Was nicht funktioniert hat / Grenzen

- **Erwartung "Δ = 3 kostet spürbar" - auf gleichverteilten Instanzen widerlegt:** der MST hat dort höchstens Grad 4 (Grad 3 in 250 von 300 Instanzen), Δ = 3 kostet nichts. Erst **Ortschaften** (Verteiler mit 5 Anschlüssen) machen Δ = 3 (+1,06 %) und Δ = 4 (+0,23 %) zum Thema. Die Ortschaften sind eine bewusste Konstruktion - auf gleichverteilten Punkten ist die Gradgrenze fast immer gratis, ab Δ = 3.
- **Erwartung "der Preis wächst mit n" - nicht monoton:** Δ = 2 kostet bei 8/12/16/20/30 Filialen +1,65/+0,66/+6,36/+7,29/+9,09 % (5 Instanzen je n, der Median springt).
- **Erwartung "die Hop-Grenze senkt den Umweg deutlich" - nur teilweise:** der größte Umweg sinkt von 2,25 (MST) nur bei H = 1 auf 1,00 und bei H = 3 auf 1,59; bei H = 2 bleibt er bei 2,20, bei H = 4 bis 6 bei 1,90. Wer den Umweg kaufen will, zahlt viel (+15,1 % für H = 3), und die längste Kante steigt mit (+21,9 % bei H = 3).
- **Greedy-Verfahren sind nicht nur schlechter, sondern scheitern:** Kruskal/Prim mit Gradgrenze enden im Wald oder stecken fest, obwohl ein Baum existiert (per Skriptsuche gefundene Kleinstinstanzen sind als Test festgeschrieben); Prim mit Tiefengrenze ebenso. Die Aufschläge der Verfahren sind nur über die Instanzen mit gültigem Baum gemessen - die Ausfälle sind meist die schweren Instanzen, deshalb sind die Preise verschiedener Verfahren nicht direkt vergleichbar (die App weist zusätzlich den Aufschlag gegen den besten Fund aus).
- **Auch Lagrange kann leer ausgehen:** Ortschaften mit 6 Anschlüssen, 30 Filialen, Δ = 2, Seed 100002: kein Verfahren findet einen gültigen Baum; ob einer existiert, ist offen (das Exaktverfahren wird bei n = 30 nicht angeboten) - die App sagt dann "Kein Verfahren fand einen gültigen Baum", nicht "es gibt keinen".
- **Exakt ist klein:** das Branch-and-Bound wird bis n = 24 (Grad) bzw. n = 12 (Hops) angeboten und ist bei enger Hop-Grenze (H = 2, 12 Filialen: in keiner der 5 Instanzen fertig) auch dort nicht immer fertig; ab n = 13 (Hops) misst die Demo keine Lücke der Heuristik mehr. **Für Hops gibt es keine Lagrange-Untergrenze** (nur den MST). Branch-and-Cut, Layered-Graph-Formulierungen (Gouveia u. a.) sind nicht gebaut.
- **Aufwand ist nicht das Thema dieses Stücks:** die Demo vergleicht Kosten und Gültigkeit, keine Laufzeit und keine Elementarschritte; das Branch-and-Bound zählt nur Suchknoten (die Zahl wird bei den Presets genannt).
- **Synthetisches Modell:** Punkte im Quadrat, k nächste Nachbarn, ein Depot; keine Kapazitäten (Folgestück **Kapazitierter MST**), keine echten Netze. Approximationsalgorithmen für Grad- und Längenbeschränkung (Fürer-Raghavachari, Haeupler u. a., "Simple Length-Constrained Minimum Spanning Trees", arXiv 2410.08170) sind nicht gebaut.
- **NP-Schwere:** Gradbeschränkter Spannbaum ist für jedes Δ ≥ 2 NP-vollständig (Garey & Johnson 1979); der hop-beschränkte ist für allgemeine Kosten NP-schwer (in der Literatur schon für H = 2 gezeigt). Für euklidische Instanzen wird das hier nicht behauptet.
- **Nachfolger (inzwischen gebaut):** Kapazitierter MST (cmst-demo), Steiner-Bäume (steiner-tree-demo, pcst-demo), Sensitivität (mst-sensitivity-demo), zufällige Spannbäume (random-spanning-tree-demo).

## Verifikation

- **Bottleneck-Satz direkt:** Bottleneck des Kruskal-Baums = Minimum über alle Spannbäume (Brute-Force, 200 Instanzen mit Gleichständen); die bottleneck-optimalen Bäume = die Spannbäume des Schwellwertgraphen (Mengengleichheit), Zahl gleich Kirchhoff, Cayley-Formel n^(n−2) für vollständige Graphen bis n = 8; billigster/teuerster Baum gleich Brute-Force; MST-Pfad = Minimax-Pfad (Brute-Force über alle einfachen Pfade).
- **Exaktheit:** das Branch-and-Bound gleich Brute-Force (alle Spannbäume) für Δ = 1, 2, 3 und H = 1 bis 4 auf je 150 Kleingraphen mit Gleichständen, auch **unlösbar erkannt**; mit Lagrange-Kosten und oberer Schranke ebenso; der billigste Hamiltonpfad (Δ = 2, vollständiger Graph) gleich Brute-Force über alle Permutationen.
- **Gültigkeit:** jede Ausgabe ist ein Spannbaum (n − 1 Kanten, zusammenhängend), erfüllt die Grenze; "kein Baum gefunden" wird nie als Baum ausgegeben.
- **Schrankenkette:** Lagrange-Untergrenze ≤ exakt ≤ Heuristik, Untergrenze ≥ MST-Kosten; die Untergrenze gilt für **beliebige** Strafen; Lokalsuche endet in einem lokalen Optimum der Tauschnachbarschaft (nachgeprüft über alle Tausche); der Schichtenbaum hat die kleinstmögliche Tiefe (Brute-Force).
- **Sonderfälle:** Δ ≥ MST-Grad bzw. H ≥ MST-Tiefe → MST; H = 1 → Stern; Δ = 1 nur für zwei Knoten; n = 1, n = 2, unzusammenhängend, Kette, Stern, gleiche Kosten überall; Kruskal-Kopie reproduziert die Zahlen der Geschwister (466,63 / 929 Schritte).
- **Schwierige Fixtures:** vier per Skriptsuche gefundene Kleinstinstanzen (Greedy Grad scheitert, Greedy Grad gültig aber teurer, Prim mit Tiefengrenze scheitert, Greedy Hop teurer), gegen Brute-Force bestätigt.
- **Zahlen:** jede Zahl in App-Text und README ist in `tests/test_claims.py` über die echten Auswertungsfunktionen (`ev.analyse`, `ev.run_config`, `ev.sweep`, `ev.feasibility`, `ev.bottleneck_counts`, `ev.degree_profile`) belegt.

## Lokal starten

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements-dev.txt
streamlit run app.py
python -m pytest tests -v
```

## Literatur

- Camerini, P. M. (1978). *The min-max spanning tree problem and some extensions.* Information Processing Letters 7(1), 10–14.
- Gabow, H. N., & Tarjan, R. E. (1988). *Algorithms for two bottleneck optimization problems.* Journal of Algorithms 9(3), 411–417.
- Volgenant, A. (1989). *A Lagrangean approach to the degree-constrained minimum spanning tree problem.* European Journal of Operational Research 39(3), 325–331.
- Garey, M. R., & Johnson, D. S. (1979). *Computers and Intractability: A Guide to the Theory of NP-Completeness.* W. H. Freeman.
- Gouveia, L. und Mitarbeiter: Modellierung des hop-beschränkten Spannbaums als Steiner-Baum-Problem über geschichtete Graphen (Mathematical Programming).
- *Simple Length-Constrained Minimum Spanning Trees* (arXiv 2410.08170, Oktober 2024) - zur längenbeschränkten Variante; Autoren und Stand vor einer Zitierung gegen die Quelle prüfen.

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Spannbäume: vom Kruskal bis zum Zufallsbaum](https://sebastianhanisch.net/konzepte-spannbaum.html).
