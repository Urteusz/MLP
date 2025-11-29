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
config/requirements.txt
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
| --mode | iris | autoencoder | batch | all |
| --architecture | np. 4,8,3 (domyślna zależnie od trybu) |
| --learning-rate | > 0 (domyślnie 0.1) |
| --momentum | [0,1] (domyślnie 0.2) |
| --no-bias | wyłącza bias |
| --max-epochs | domyślnie Iris 1000, Autoencoder 3000 |
| --error-threshold | próg MSE (Iris None, Autoencoder 1e-6) |
| --log-interval | co ile epok zapis do CSV |
| --model-file | własna nazwa pliku modelu |
| --batch-prefix | prefiks dla modeli w batch |

## Batch nadpisania
W trybie `--mode batch` można globalnie nadpisać: architekturę, lr, momentum, bias, max_epochs, error_threshold, log_interval.

## Dane
Plik `iris.data` musi znajdować się w katalogu głównym repo (jest używany przez runner przez ścieżkę bezwzględną). Autoencoder używa danych syntetycznych.

## Logi i modele
Przykład nazwy logu: `logs/iris/Iris_lr0.2_mom0.3_biasTrue.csv`
Przykład modelu: `models/iris/iris_mlp_model.pkl` lub `models/autoencoder/autoencoder_4_2_4_bias_on.pkl`

## Metryki
Iris: accuracy, confusion matrix, precision/recall/F1 per klasa.
Autoencoder: MSE per wzorzec + średni MSE + porównanie z wczytanym modelem.

## Rozszerzenia (pomysły)
- Eksport metryk do JSON w `reports/`
- Parametr `--seed`
- Wizualizacja logów (skrypt w `tools/`)
- Testy jednostkowe pytest dla forward/backward
- Inne zbiory danych
