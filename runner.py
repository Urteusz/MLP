import numpy as np
from MLP import MLP
from Trainer import Trainer
from Data_utils import load_iris_data, normalize_features, split_dataset

# Główna funkcja do uruchamiania klasyfikatora lub autoenkodera
def run_classification(
        name="Iris",                      # Nazwa zadania (Iris lub Autoencoder)
        architecture=None,               # Architektura sieci, np. [4, 2, 4] dla autoenkodera
        use_bias=True,                   # Czy używać biasów w neuronach
        learning_rate=0.1,               # Współczynnik uczenia
        momentum=0.2,                    # Momentum dla uczenia
        max_epochs=None,                 # Maksymalna liczba epok
        error_threshold=None,            # Próg błędu kończący uczenie
        log_interval=20,                 # Co ile epok logować postęp
        trained_model_file="iris_mlp_model.pkl"  # Nazwa pliku dla zapisu modelu klasyfikatora
):

    # Tworzymy nazwę pliku logowania błędu na podstawie parametrów
    training_log_file = (
        f"{name}_lr{learning_rate}_mom{momentum}_bias{use_bias}.csv"
    )

    print(f"--- Uruchamianie dla: {name} (parametryzowane) ---")

    # --- Wczytywanie i przygotowanie danych ---
    if name == "Iris":
        # Wczytanie zbioru danych Iris
        IRIS_DATA_FILE = "iris.data"
        IRIS_CLASS_NAMES = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
        dataset = load_iris_data(IRIS_DATA_FILE)
        if not dataset:
            print("Nie udało się wczytać danych. Przerwanie.")
            return
        # Normalizacja cech i podział na zbiory treningowy/testowy
        dataset = normalize_features(dataset)
        train_data, test_data = split_dataset(dataset)

    elif name == "Autoencoder":
        # Przykładowe dane wejściowe i oczekiwane wyjścia (rekonstrukcja tożsama)
        train_data = [
            (np.array([1, 0, 0, 0], dtype=float), np.array([1, 0, 0, 0], dtype=float)),
            (np.array([0, 1, 0, 0], dtype=float), np.array([0, 1, 0, 0], dtype=float)),
            (np.array([0, 0, 1, 0], dtype=float), np.array([0, 0, 1, 0], dtype=float)),
            (np.array([0, 0, 0, 1], dtype=float), np.array([0, 0, 0, 1], dtype=float))
        ]
        test_data = train_data  # W przypadku autoenkodera zbiór testowy = treningowy
        IRIS_CLASS_NAMES = None
        if architecture is None:
            architecture = [4, 2, 4]  # Domyślna architektura autoenkodera
            print(f"Autoencoder: Używam domyślnej architektury {architecture}")

    else:
        print(f"Nieznany typ zadania: {name}")
        return

    # --- Budowa i inicjalizacja sieci neuronowej ---
    mlp = MLP(layer_sizes=architecture, use_bias=use_bias)

    # --- Trening modelu ---
    trainer = Trainer(mlp, train_data,
                      learning_rate=learning_rate,
                      momentum=momentum)

    avg_final_error_train = None  # Do późniejszego raportowania błędu MSE

    trainer.train(error_threshold=error_threshold,
                  max_epochs=max_epochs,
                  log_interval=log_interval,
                  log_file=training_log_file)

    # Obliczanie końcowego błędu dla autoenkodera
    if name == "Autoencoder":
        temp_error_sum = 0
        for x, y_exp in train_data:
            y_act = mlp.forward(x)
            temp_error_sum += mlp.calculate_pattern_error(y_act, y_exp)
        if train_data:
            avg_final_error_train = temp_error_sum / len(train_data)

    # --- Zapis modelu do pliku ---
    autoencoder_model_file = "autoencoder_model.pkl"
    if name == "Iris":
        mlp.save_network(trained_model_file)
    elif name == "Autoencoder":
        suffix = "_bias_on" if use_bias else "_bias_off"
        arch_str = "_".join(map(str, architecture)) if architecture else "unknown_arch"
        autoencoder_model_file = f"autoencoder_{arch_str}{suffix}.pkl"
        mlp.save_network(autoencoder_model_file)

    # --- Ewaluacja modelu ---
    if name == "Iris":
        # Ocena skuteczności na zbiorze testowym
        print("\n--- Ewaluacja na zbiorze testowym ---")
        metrics_test = trainer.calculate_classification_metrics(test_data, class_names=IRIS_CLASS_NAMES)

        # Wyświetlenie kilku przykładowych predykcji
        print("\n--- Przykładowe predykcje (zbiór testowy) ---")
        for i in range(min(5, len(test_data))):
            x_sample, y_expected = test_data[i]
            y_actual = mlp.forward(x_sample)
            pred_idx = np.argmax(y_actual)
            actual_idx = np.argmax(y_expected)
            print(f"Wzorzec {i + 1}:")
            print(f"  Wejście: {np.round(x_sample, 2)}")
            print(f"  Oczekiwane: {IRIS_CLASS_NAMES[actual_idx]}")
            print(f"  Przewidziane: {IRIS_CLASS_NAMES[pred_idx]} | Prawdopodobieństwa: {np.round(y_actual, 3)}")

        # Test poprawności zapisu i odczytu modelu
        print("\n--- Test zapisanego modelu Iris (na zbiorze testowym) ---")
        try:
            loaded_mlp_iris = MLP.load_network(trained_model_file)
            trainer_loaded_iris = Trainer(loaded_mlp_iris, test_data)
            metrics_loaded_iris = trainer_loaded_iris.calculate_classification_metrics(test_data,
                                                                                       class_names=IRIS_CLASS_NAMES)
            print(f"Dokładność wczytanego modelu na zbiorze testowym: {metrics_loaded_iris['accuracy']:.4f}")
            if metrics_test and 'accuracy' in metrics_test:
                if abs(metrics_loaded_iris['accuracy'] - metrics_test['accuracy']) < 1e-6:
                    print("  Potwierdzenie: Wydajność wczytanego modelu jest zgodna z oryginałem.")
                else:
                    print("  Uwaga: Wydajność wczytanego modelu różni się od oryginału.")
        except FileNotFoundError:
            print(f"Błąd: Nie znaleziono pliku modelu: {trained_model_file}")
        except Exception as e:
            print(f"Błąd podczas testu wczytanego modelu Iris: {e}")

        print("\n--- Zakończono klasyfikację Irysów ---")

    elif name == "Autoencoder":
        # Pokazanie finalnego błędu MSE
        if avg_final_error_train is not None:
            print(f"Finalny średni błąd MSE (trening): {avg_final_error_train:.8f}")

        # Szczegółowe wyniki dla każdego wzorca
        print("\n--- Szczegółowe wyniki dla każdego wzorca treningowego Autoenkodera ---")
        col_widths = {'id': 8, 'input': 25, 'hidden': 20, 'output': 25, 'mse': 12}

        header = (f"{'Wzorzec':<{col_widths['id']}} | "
                  f"{'Wejście':<{col_widths['input']}} | "
                  f"{'Warstwa Ukryta':<{col_widths['hidden']}} | "
                  f"{'Rekonstrukcja':<{col_widths['output']}} | "
                  f"{'MSE':<{col_widths['mse']}}")
        print(header)
        print("-" * len(header))

        hidden_layer_index = 0  # Pierwsza warstwa (po wejściu) to enkoder

        if not train_data:
            print("Brak danych treningowych.")
        else:
            for i, (pattern_in, expected_out) in enumerate(train_data):
                reconstructed_out = mlp.forward(pattern_in)

                hidden_representation_str = "N/A"
                if mlp.layers and len(mlp.layers) > hidden_layer_index and hasattr(mlp.layers[hidden_layer_index], 'last_output'):
                    hidden_activations = mlp.layers[hidden_layer_index].last_output
                    if hidden_activations is not None:
                        hidden_representation_str = str(np.round(hidden_activations, 4))

                pattern_in_str = str(np.round(pattern_in, 4))
                reconstructed_out_str = str(np.round(reconstructed_out, 4))
                pattern_mse = mlp.calculate_pattern_error(reconstructed_out, expected_out)

                print(f"{i + 1:<{col_widths['id']}} | "
                      f"{pattern_in_str:<{col_widths['input']}} | "
                      f"{hidden_representation_str:<{col_widths['hidden']}} | "
                      f"{reconstructed_out_str:<{col_widths['output']}} | "
                      f"{pattern_mse:<{col_widths['mse']}.8f}")

        # Przykładowe rekonstrukcje
        print("\n--- Przykładowe Rekonstrukcje ---")
        num_samples_to_show = min(5, len(test_data))

        header_samples = (f"{'Wzorzec':<{col_widths['id']}} | "
                          f"{'Oryginał':<{col_widths['input']}} | "
                          f"{'Warstwa Ukryta':<{col_widths['hidden']}} | "
                          f"{'Rekonstrukcja':<{col_widths['output']}} | "
                          f"{'MSE':<{col_widths['mse']}} | "
                          f"{'Jakość':<10}")
        print(header_samples)
        print("-" * len(header_samples))

        for i in range(num_samples_to_show):
            x_sample, y_expected = test_data[i]
            y_actual = mlp.forward(x_sample)

            hidden_representation_sample_str = "N/A"
            if mlp.layers and len(mlp.layers) > hidden_layer_index and hasattr(mlp.layers[hidden_layer_index], 'last_output'):
                hidden_activations_sample = mlp.layers[hidden_layer_index].last_output
                if hidden_activations_sample is not None:
                    hidden_representation_sample_str = str(np.round(hidden_activations_sample, 4))

            x_sample_str = str(np.round(x_sample, 4))
            y_actual_str = str(np.round(y_actual, 4))
            pattern_mse_sample = mlp.calculate_pattern_error(y_actual, y_expected)

            print(f"{i + 1:<{col_widths['id']}} | "
                  f"{x_sample_str:<{col_widths['input']}} | "
                  f"{hidden_representation_sample_str:<{col_widths['hidden']}} | "
                  f"{y_actual_str:<{col_widths['output']}} | "
                  f"{pattern_mse_sample:<{col_widths['mse']}.8f} | ")

        # Ewaluacja modelu z pliku
        print("\n--- Test zapisanego modelu Autoenkodera ---")
        try:
            loaded_mlp_autoencoder = MLP.load_network(autoencoder_model_file)

            total_mse_loaded_model = 0
            for x_test, y_test_expected in test_data:
                y_test_actual = loaded_mlp_autoencoder.forward(x_test)
                total_mse_loaded_model += loaded_mlp_autoencoder.calculate_pattern_error(y_test_actual, y_test_expected)

            avg_mse_loaded_model = total_mse_loaded_model / len(test_data)
            print(f"Średni błąd MSE wczytanego modelu autoenkodera: {avg_mse_loaded_model:.8f}")

            if avg_final_error_train is not None:
                if abs(avg_mse_loaded_model - avg_final_error_train) < 1e-6:
                    print("  Potwierdzenie: Wydajność wczytanego modelu jest zgodna z oryginałem.")
                else:
                    print("  Uwaga: Różnica w wydajności wczytanego modelu.")

        except FileNotFoundError:
            print(f"Błąd: Nie znaleziono pliku modelu: {autoencoder_model_file}")
        except Exception as e:
            print(f"Błąd podczas testu wczytanego modelu autoenkodera: {e}")

        print("\n--- Zakończono przetwarzanie Autoenkodera ---")
