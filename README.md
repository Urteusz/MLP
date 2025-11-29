# MLP & Autoencoder

Implementacja prostego MLP od podstaw (bez frameworków) z obsługą:
- Klasyfikacji Iris
- Autoenkodera 4-2-4
- Batch uruchomień (predefiniowane konfiguracje)
- Trybu ALL (wszystko po kolei)

## Struktura
```
MLP.py / Layer.py / Trainer.py / Data_utils.py / runner.py / Main.py
models/ (zapisy modeli per tryb)
logs/   (pliki CSV z błędem per tryb)
figures/ (automatycznie generowane wykresy porównawcze)
config/requirements.txt (zależności + pytest)
tests/ (testy jednostkowe)
LICENSE (MIT)
README.md
```

## Instalacja
```bash
python -m pip install -r config/requirements.txt
```

## Szybki start (interaktywnie)
```bash
python Main.py
```

## CLI (przykłady)
```bash
python Main.py --mode iris --architecture 4,8,3 --learning-rate 0.2 --momentum 0.3 --max-epochs 1000 --log-interval 10
python Main.py --mode autoencoder --architecture 4,2,4 --learning-rate 0.9 --momentum 0.0 --max-epochs 3000 --error-threshold 0.000001
python Main.py --mode batch --batch-prefix serie1
python Main.py --mode all --batch-prefix wszystko
```

### Argumenty
| Argument | Opis |
|----------|------|
| --mode | Dostępne: `iris`, `autoencoder`, `batch`, `all` |
| --architecture | np. `4,8,3` (domyślna zależnie od trybu) |
| --learning-rate | > 0 (domyślnie 0.1) |
| --momentum | W zakresie [0,1] (domyślnie 0.2) |
| --no-bias | Wyłącza bias (domyślnie włączony) |
| --max-epochs | Iris domyślnie 1000, Autoencoder 3000 |
| --error-threshold | Próg MSE (Iris: None, Autoencoder: 1e-6 domyślnie) |
| --log-interval | Co ile epok zapis do CSV (domyślnie 1) |
| --model-file | Własna nazwa pliku modelu (opcjonalnie) |
| --batch-prefix | Prefiks dla modeli w batch |

## Batch nadpisania
W trybie `--mode batch` można globalnie nadpisać: `architecture`, `learning-rate`, `momentum`, `bias`, `max-epochs`, `error-threshold`, `log-interval`.

## Architektura
Model MLP tworzony jest na liście rozmiarów warstw, np. `[4, 8, 3]`:
- Warstwa wejściowa: rozmiar 4 (cechy Iris)
- 1 warstwa ukryta: 8 neuronów (sigmoid)
- Warstwa wyjściowa: 3 neurony (one-hot klas)
Autoencoder: `[4, 2, 4]` kompresuje wejście 4D do 2D i rekonstruuje 4D.
Aktualizacja wag: klasyczne backprop + momentum, aktywacja sigmoid, funkcja błędu MSE.

## Dane
Plik `iris.data` (pełny zbiór 150 rekordów) musi znajdować się w katalogu głównym repo. Ładowanie odbywa się poprzez ścieżkę bezwzględną w `runner.py`.
Autoencoder używa syntetycznych wektorów one-hot.
Źródło Iris: https://archive.ics.uci.edu/ml/datasets/iris

## Logi i modele
Przykład logu: `logs/iris/Iris_lr0.2_mom0.3_biasTrue.csv`
Przykład modelu: `models/iris/Iris_model.pkl` lub `models/autoencoder/autoencoder_4_2_4_bias_on.pkl`

## Metryki
Klasyfikacja Iris: Accuracy, macierz pomyłek, Precision/Recall/F1 per klasa + macro-averages.
Autoencoder: MSE per wzorzec + średni MSE + porównanie z wczytanym modelem.

## Generowanie wykresów
Po zebraniu logów możesz wygenerować zestawienia:
```bash
python plots.py
```
Zapisane pliki znajdziesz w `figures/` (np. `iris_all.png`, `autoencoder_all.png`).

## Testy
Uruchomienie testów jednostkowych:
```bash
python -m pytest -q
```
Obecnie testy obejmują:
- Poprawność kształtu wyjścia sieci
- Spadek błędu po iteracjach uczenia na prostym zbiorze XOR

## Przykładowe wyniki (referencyjne)
| Zadanie | Architektura | LR | Momentum | Epoki | Accuracy / MSE |
|---------|--------------|----|----------|-------|----------------|
| Iris    | 4,8,3        |0.2 |0.3       | 30    | Accuracy ≈ 97.8% |
| Autoencoder | 4,2,4    |0.2 |0.1       | 3000  | Średni MSE ≈ 0.00025 (przy wyższych LR) |

(Uwaga: Wyniki zależą od inicjalizacji wag i losowego shuffle.)

## Rozszerzenia (pomysły)
- Eksport metryk do JSON w `reports/`
- Parametr `--seed` dla pełnej reprodukowalności (obecnie tylko split dataset)
- Wizualizacja przebiegu błędu z oknem GUI / dashboard
- Dodatkowe zbiory (Wine, Digits) w osobnym module loaderów
- Testy dla metryk i zapisu/odczytu modelu

## Licencja
Projekt udostępniany na licencji MIT (patrz plik `LICENSE`).
