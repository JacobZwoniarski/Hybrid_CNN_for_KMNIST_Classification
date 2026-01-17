# Hybrid CNN + Quantum Layer for KMNIST (PyTorch + PennyLane)

Projekt na przedmiot **Uczenie Maszynowe**: porównanie dwóch modeli na zbiorze **KMNIST**:
- **baseline (matched classical)** – klasyczna sieć CNN z “dopasowaną” architekturą (fair baseline)
- **hybrid (quantum)** – ta sama architektura, ale z dodatkową **warstwą kwantową** (PennyLane) w miejscu klasycznego bloku pośredniego

Projekt zawiera:
- trening, ewaluację, wykresy i confusion matrix,
- “fair comparison” (baseline vs hybrid przy tych samych hiperparametrach),
- inference na nowych danych (PNG -> predykcja + top-k),
- przykładowe wejścia/wyjścia do infer (bez dołączania całego datasetu).

## Struktura repo (skrót)
- `src/qcnn_kmnist/` – kod źródłowy (train/eval/infer/plots)
- `scripts/` – skrypty bash do uruchamiania pipeline’u
- `outputs/` – logi, checkpointy, wykresy, porównania (generowane)
- `examples/inputs/` – przykładowe wejścia do infer (PNG)
- `examples/outputs/` – przykładowe wyjścia infer (JSON)

## Wymagania
- Python 3.9+ (testowane na 3.11)
- pip
- System z bash (Mac/Linux; na Windows najlepiej Git Bash/WSL)
- Dla hybrydy rekomendowane uruchamianie na CPU (PennyLane + torch + MPS bywa problematyczne)

---

# Wariant 1: Odtwarzanie środowiska i uruchamianie treningu + porównania

## 1) Klon repo
```bash
git clone https://github.com/JacobZwoniarski/Hybrid_CNN_for_KMNIST_Classification.git
cd Hybrid_CNN_for_KMNIST_Classification
```

## 2) Utworzenie i aktywacja środowiska (venv)
Jeśli masz skrypt:
```bash
bash scripts/setup_venv.sh
source .venv/bin/activate
```
Jeśli bez skryptu (ręcznie):
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
```

## 3) Instalacja zależności
```bash
pip install -r requirements.txt
```

## 4) Instalacja projektu (żeby działało python -m qcnn_kmnist...)
```bash
pip install -e .
```

## Trening modeli
Trening baseline (matched classical)
```bash
bash scripts/train_baseline.sh
```
Trening hybrydy (quantum)
```bash
bash scripts/train_hybrid.sh
```

## Ewaluacja (test) + confusion matrix
Ewaluacja baseline (latest)
```bash
bash scripts/eval_latest.sh
```
Ewaluacja hybrydy (latest)
```bash
bash scripts/eval_latest_hybrid.sh
```
Wyniki zapisują się jako:
- `outputs/predictions/<run_name>/eval_best/eval_test.json`
- `outputs/figures/<run_name>/eval_best/confusion_matrix_test.png`
- `outputs/figures/<run_name>/eval_best/confusion_matrix_test_norm.png`

## Fair comparison (baseline vs hybrid tymi samymi parametrami)
To jest docelowy sposób porównania (identyczne hiperparametry, seed, bottleneck).
Docelowa konfiguracja hybrydy:
```bash
N_QUBITS=6 N_LAYERS=3 EPOCHS=15 LR=5e-4 bash scripts/run_fair_comparison.sh
```
Wynik porównania:
- `outputs/comparisons/<timestamp>/summary.json`
- `outputs/comparisons/<timestamp>/summary.csv`
- `outputs/comparisons/<timestamp>/figures/` (kopie confusion-matrix)

Uwaga: `N_LAYERS` nie wpływa na baseline (klasyczny), ale baseline używa `N_QUBITS` jako wymiaru bottlenecku, więc porównanie pozostaje “fair”.

## Wizualizacje jakościowe (co widzi model + top błędy)
Generuje:
- siatkę predykcji z confidence,
- histogram pewności,
- “najbardziej pewne błędy”.

```bash
bash scripts/qual_latest.sh
```
Pliki trafiają do:
`outputs/figures/<run_name>/qualitative_best/`

---

# Wariant 2: Testowanie wytrenowanych modeli na “nowych danych” (infer)
Ten wariant nie wymaga trenowania. Uruchamia infer na kilku obrazkach PNG i dostaje wyniki w JSON.

## 1) Utworzenie środowiska + zależności
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

## 2) Checkpointy modeli
Repo powinno zawierać checkpointy w:
- `outputs/checkpoints/baseline_*/best.pt`
- `outputs/checkpoints/hybrid_*/best.pt`


## 3) Przykładowe wejścia (kilka PNG)
W repo są już przykładowe wejścia w `examples/inputs/`.
Jeśli ich nie ma, można je wygenerować (pobierze KMNIST, ale zapisze tylko kilka PNG):
```bash
bash scripts/generate_examples.sh
```

## 4) Infer (predykcje na przykładach)
Infer dla baseline (najświeższy checkpoint)
```bash
MODEL=baseline bash scripts/infer_example.sh
```
Infer dla hybrydy (najświeższy checkpoint)
```bash
MODEL=hybrid bash scripts/infer_example.sh
```
Wyniki zapisują się jako JSON-y w:
`examples/outputs/<run_name>/*.json`

Każdy JSON zawiera m.in.:
- `pred_id`, `pred_name`
- `confidence`
- `topk` (lista top-k klas z prawdopodobieństwami)

## Porównawcze PNG: baseline vs hybrid (Top-3 na jednym obrazku)

Skrypt generuje dla każdego PNG z `examples/inputs/` osobny obrazek porównawczy:
- u góry: wejściowy obrazek + true label (jeśli dostępny)
- na dole: Top-3 prawdopodobieństwa dla baseline oraz hybrid

Uruchom:
```bash
source .venv/bin/activate
pip install -e .
bash scripts/infer_compare_plots.sh
```
Wyniki zapisują się w:
`examples/compare_outputs/<baselineRun>__vs__<hybridRun>/`

### Infer na własnym obrazku PNG (opcjonalnie)
Jeśli chce się sprawdzić pojedynczy plik `my.png` (najlepiej 28x28, grayscale):
```bash
python -m qcnn_kmnist.infer --checkpoint outputs/checkpoints/baseline_*/best.pt --image my.png --device cpu
```

## Najczęstsze problemy

**pip install -e . nie działa**
Upewnij się, że w root repo jest `pyproject.toml`, a potem:
```bash
pip install -e .
```

**Hybryda działa wolno**
To normalne: symulacja obwodu kwantowego na CPU jest kosztowna.
Dla odtwarzalności rekomendowane jest uruchamianie hybrydy na CPU.

## Quickstart (podsumowanie komend)

**Trening + porównanie:**
```bash
bash scripts/train_baseline.sh
bash scripts/train_hybrid.sh
bash scripts/eval_latest.sh
bash scripts/eval_latest_hybrid.sh
N_QUBITS=6 N_LAYERS=3 bash scripts/run_fair_comparison.sh
```

**Infer na przykładach:**
```bash
bash scripts/generate_examples.sh
MODEL=baseline bash scripts/infer_example.sh
MODEL=hybrid bash scripts/infer_example.sh
```
