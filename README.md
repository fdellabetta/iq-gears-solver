# IQ Gears – Solver (SmartGames)

Questo repository contiene un **solver** per il gioco da tavolo **IQ Gears** di SmartGames. Il programma cerca automaticamente una disposizione valida dei 7 pezzi sulla griglia 5×5 che:

- riempia **tutte** le 25 celle;
- copra gli **indizi** delle rotelline (`*`) eventualmente presenti nella sfida;
- realizzi una **connessione continua di ingranaggi** tra due celle obiettivo (START→TARGET).

Per una descrizione ufficiale del gioco (regole e panoramica delle 120 sfide), vedi:

- **Sito SmartGames – IQ Gears**: https://www.smartgames.eu/uk/one-player-games/iq-gears
- **Challenge Booklet / Regole (PDF)**: [SmartGames_IQ_Gears\_\_Challenge_Booklet.pdf](SmartGames_IQ_Gears__Challenge_Booklet.pdf)

> **Ambiente**: sviluppato e testato con **Python 3.11.9**. Per l’esportazione PNG è necessario il pacchetto **Pillow**.

---

## Requisiti

Installare **Pillow**:

```bash
pip install Pillow
```

---

## File principali

- `ig-gears-solver.py` — solver deterministico con:
  - output **ANSI** allineato (celle a larghezza fissa; rotelline `*` in **bianco**);
  - esportazione **PNG** “a pallini” opzionale (singolo file).
- File di esempio delle sfide (`challenge_*.json`).

---

## Formato delle sfide (JSON)

Ogni sfida è un file `.json` con due griglie 5×5 (righe dall’alto verso il basso):

- `piece_grid`: lettere **A–G** per pezzi pre‑posizionati (blocchi fissati). `.` = cella vuota.
- `gear_grid`: `*` dove deve esserci un **dentino**; `.` altrove.

Esempio:

```json
{
  "gear_grid": [".....", ".....", "....*", ".....", "....."],
  "piece_grid": [".....", "...E.", "...EE", "...E.", "....."]
}
```

---

## Utilizzo

### Output ANSI (terminal)

```bash
python ig-gears-solver.py --challenge challenge_100.json
```

- Stampa una griglia 5×5 **allineata**: ogni cella è un riquadro a 2 caratteri con **sfondo colorato** in base al pezzo;
- il dentino è indicato da un `*` **bianco**.

### Esportare un PNG “a pallini”

```bash
python ig-gears-solver.py --challenge challenge_100.json --png 100 --png-scale 120
```

- Genera `100.png` con **cerchi colorati** al centro delle celle e un **asterisco** vettoriale per le rotelline.
- `--png-scale` controlla la **dimensione della cella** in pixel (default 96).

### Opzioni CLI

- `--challenge PATH` (obbligatorio): file JSON della sfida.
- `--debug` (facoltativo): stampa informazioni di debug.
- `--png PREFIX` (facoltativo): esporta un **solo PNG** `PREFIX.png`.
- `--png-scale N` (facoltativo): lato cella in pixel per il PNG.

---

## Note implementative

- **Deterministico**: l’ordine di esplorazione è fisso; a parità di sfida, il solver segue sempre lo stesso percorso.
- **Validazione**: una soluzione è accettata solo se la griglia è completa, gli indizi sono coperti e il path START→TARGET esiste con adiacenza 4‑connessa.

---

### Avvertenze

IQ Gears è un marchio di **SmartGames**. Le sfide ufficiali e il booklet sono soggetti ai relativi diritti.
