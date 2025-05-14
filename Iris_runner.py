# Iris_runner.py
import numpy as np
from MLP import MLP
from Trainer import Trainer
from Data_utils import load_iris_data, normalize_features, split_dataset

# --- Konfiguracja ---
IRIS_DATA_FILE = "iris.data"  # Upewnij się, że ten plik jest w katalogu
TRAIN_RATIO = 0.7
RANDOM_SEED = 42  # Dla powtarzalności podziału danych

# Parametry sieci i treningu
# Możesz eksperymentować z architekturą: [wejścia, ...warstwy_ukryte..., wyjścia]
# Dla Irysów: 4 cechy wejściowe, 3 klasy wyjściowe (one-hot)
NETWORK_ARCHITECTURE = [4, 8, 3]  # Np. 4 wejścia, 1 warstwa ukryta z 8 neuronami, 3 wyjścia
USE_BIAS = True
LEARNING_RATE = 0.1  # Współczynniki nauki często są mniejsze dla problemów klasyfikacyjnych
MOMENTUM = 0.2
MAX_EPOCHS = 500
ERROR_THRESHOLD = 0.01  # Cel MSE
LOG_INTERVAL_EPOCHS = 20  # Co ile epok logować błąd
SHUFFLE_TRAINING_DATA = True

# Nazwy plików wynikowych
TRAINED_MODEL_FILE = "iris_mlp_model.pkl"
TRAINING_LOG_FILE = "iris_training_log.csv"
TESTING_LOG_FILE = "iris_testing_log.csv"

# Nazwy klas dla raportów
IRIS_CLASS_NAMES = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]


def run_iris_classification():
    print("--- Klasyfikacja zbioru Irysów ---")

    # 1. Wczytanie i przygotowanie danych
    print(f"\n1. Wczytywanie danych z pliku: {IRIS_DATA_FILE}")
    dataset = load_iris_data(IRIS_DATA_FILE)
    if not dataset:
        print("Nie udało się wczytać danych. Przerwanie.")
        return

    print(f"Wczytano {len(dataset)} wzorców.")

    print("Normalizacja cech (Min-Max do [0,1])...")
    dataset = normalize_features(dataset)

    print(f"Podział danych na zbiór treningowy ({TRAIN_RATIO * 100}%) i testowy (seed={RANDOM_SEED})...")
    train_data, test_data = split_dataset(dataset, train_ratio=TRAIN_RATIO, seed=RANDOM_SEED)
    print(f"Rozmiar zbioru treningowego: {len(train_data)}")
    print(f"Rozmiar zbioru testowego: {len(test_data)}")

    # 2. Budowa sieci neuronowej
    print(f"\n2. Budowa sieci MLP o architekturze: {NETWORK_ARCHITECTURE}, Użycie biasu: {USE_BIAS}")
    mlp = MLP(layer_sizes=NETWORK_ARCHITECTURE, use_bias=USE_BIAS)

    # 3. Trening sieci
    print("\n3. Trening sieci...")
    trainer = Trainer(mlp, train_data,
                      learning_rate=LEARNING_RATE,
                      momentum=MOMENTUM,
                      shuffle=SHUFFLE_TRAINING_DATA)

    trainer.train(max_epochs=MAX_EPOCHS,
                  error_threshold=ERROR_THRESHOLD,
                  log_interval=LOG_INTERVAL_EPOCHS,
                  log_file=TRAINING_LOG_FILE)

    # 4. Zapis wytrenowanej sieci
    print("\n4. Zapis wytrenowanej sieci...")
    mlp.save_network(TRAINED_MODEL_FILE)

    # 5. Ewaluacja na zbiorze treningowym
    print("\n5. Ewaluacja na zbiorze treningowym:")
    trainer.calculate_classification_metrics(train_data, class_names=IRIS_CLASS_NAMES)

    # 6. Ewaluacja na zbiorze testowym
    print("\n6. Ewaluacja na zbiorze testowym:")
    metrics_test = trainer.calculate_classification_metrics(test_data, class_names=IRIS_CLASS_NAMES)

    # Dodatkowe informacje o poprawnie sklasyfikowanych obiektach
    y_true_test = [np.argmax(y) for _, y in test_data]
    y_pred_test = []
    for x, _ in test_data:
        output = mlp.forward(x)
        y_pred_test.append(np.argmax(output))

    print("\nLiczba poprawnie sklasyfikowanych obiektów (zbiór testowy):")
    for i, class_name in enumerate(IRIS_CLASS_NAMES):
        correct_class = np.sum((np.array(y_true_test) == i) & (np.array(y_pred_test) == i))
        total_class = np.sum(np.array(y_true_test) == i)
        if total_class > 0:
            print(f"  {class_name}: {correct_class} / {total_class} ({(correct_class / total_class) * 100:.2f}%)")
        else:
            print(f"  {class_name}: {correct_class} / {total_class} (brak wzorców w tej klasie w zbiorze testowym)")
    print(
        f"  Łącznie: {metrics_test['accuracy'] * len(test_data):.0f} / {len(test_data)} ({metrics_test['accuracy'] * 100:.2f}%)")

    # 7. Szczegółowe testowanie z zapisem do pliku
    print(f"\n7. Zapis szczegółowych wyników testowania do pliku: {TESTING_LOG_FILE}")
    # Można wybrać, co logować, np. pominąć wagi dla czytelności, jeśli jest ich dużo
    log_elements_for_test = ['input', 'expected', 'output', 'pattern_error', 'output_errors']
    trainer.test(test_data, log_file=TESTING_LOG_FILE, log_elements=log_elements_for_test)

    # 8. Przykładowe predykcje (opcjonalnie)
    print("\n8. Przykładowe predykcje dla kilku wzorców testowych:")
    num_samples_to_show = min(5, len(test_data))
    for i in range(num_samples_to_show):
        x_sample, y_expected_sample = test_data[i]
        y_actual_probs_sample = mlp.forward(x_sample)

        predicted_class_idx = np.argmax(y_actual_probs_sample)
        actual_class_idx = np.argmax(y_expected_sample)

        print(f"  Wzorzec #{i + 1}:")
        print(f"    Wejście (znormalizowane): {np.round(x_sample, 3).tolist()}")
        print(f"    Oczekiwana klasa: {IRIS_CLASS_NAMES[actual_class_idx]} (one-hot: {y_expected_sample.tolist()})")
        print(
            f"    Przewidziana klasa: {IRIS_CLASS_NAMES[predicted_class_idx]} (prawdopodobieństwa: {np.round(y_actual_probs_sample, 3).tolist()})")
        print(f"    Poprawnie: {'TAK' if predicted_class_idx == actual_class_idx else 'NIE'}")

    # 9. Test wczytania zapisanej sieci
    print("\n9. Test wczytania zapisanego modelu...")
    try:
        loaded_mlp = MLP.load_network(TRAINED_MODEL_FILE)
        # Stworzenie nowego trainera dla wczytanego modelu lub przypisanie do istniejącego
        # Ważne: Użyj danych testowych, które nie były "widziane" przez nowy model
        # po jego inicjalizacji (choć tu to ten sam zbiór testowy)
        trainer_for_loaded_model = Trainer(loaded_mlp, test_data)  # Użyj test_data, a nie train_data
        print("  Model wczytany pomyślnie. Sprawdzanie dokładności na zbiorze testowym...")
        metrics_loaded = trainer_for_loaded_model.calculate_classification_metrics(test_data,
                                                                                   class_names=IRIS_CLASS_NAMES)
        print(f"  Dokładność wczytanego modelu na zbiorze testowym: {metrics_loaded['accuracy']:.4f}")
        if np.isclose(metrics_loaded['accuracy'], metrics_test['accuracy']):
            print("  Dokładność wczytanego modelu zgodna z oryginalnym. Test zapisu/odczytu pomyślny.")
        else:
            print("  UWAGA: Dokładność wczytanego modelu różni się od oryginalnego. Sprawdź proces zapisu/odczytu.")

    except Exception as e:
        print(f"  Błąd podczas wczytywania lub testowania zapisanego modelu: {e}")

    print("\n--- Zakończono klasyfikację Irysów ---")


if __name__ == "__main__":
    run_iris_classification()