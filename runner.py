import numpy as np
from MLP import MLP
from Trainer import Trainer
from Data_utils import load_iris_data, normalize_features, split_dataset


def run_classification(
        name="Iris",
        architecture=None,
        use_bias=True,
        learning_rate=0.1,
        momentum=0.2,
        max_epochs=None,
        error_threshold=None,
        log_interval=20,
        trained_model_file="iris_mlp_model.pkl"
):
    training_log_file = f"{name}_lr{learning_rate}_mom{momentum}_bias{use_bias}.csv"

    if name == "Iris":
        IRIS_DATA_FILE = "iris.data"
        IRIS_CLASS_NAMES = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
        dataset = load_iris_data(IRIS_DATA_FILE)
        if not dataset:
            print("Nie udało się wczytać danych.")
            return
        dataset = normalize_features(dataset)
        train_data, test_data = split_dataset(dataset)
    elif name == "Autoencoder":
        train_data = [
            (np.array([1, 0, 0, 0], dtype=float), np.array([1, 0, 0, 0], dtype=float)),
            (np.array([0, 1, 0, 0], dtype=float), np.array([0, 1, 0, 0], dtype=float)),
            (np.array([0, 0, 1, 0], dtype=float), np.array([0, 0, 1, 0], dtype=float)),
            (np.array([0, 0, 0, 1], dtype=float), np.array([0, 0, 0, 1], dtype=float))
        ]
        test_data = train_data
        if architecture is None:
            architecture = [4, 2, 4]

    else:
        print(f"Nieznany typ zadania: {name}")
        return

    mlp = MLP(layer_sizes=architecture, use_bias=use_bias)
    trainer = Trainer(mlp, train_data, learning_rate=learning_rate, momentum=momentum)

    trainer.train(error_threshold=error_threshold, max_epochs=max_epochs, log_interval=log_interval,
                  log_file=training_log_file)

    autoencoder_model_file = "autoencoder_model.pkl"
    if name == "Iris":
        mlp.save_network(trained_model_file)
    elif name == "Autoencoder":
        suffix = "_bias_on" if use_bias else "_bias_off"
        arch_str = "_".join(map(str, architecture)) if architecture else "unknown_arch"
        autoencoder_model_file = f"autoencoder_{arch_str}{suffix}.pkl"
        mlp.save_network(autoencoder_model_file)

        total_mse = 0
        for x, y_exp in train_data:
            y_act = mlp.forward(x)
            total_mse += mlp.calculate_pattern_error(y_act, y_exp)

        avg_mse = total_mse / len(train_data) if train_data else 0
        print(f"Finalny średni błąd MSE (trening): {avg_mse:.8f}")

        try:
            loaded_mlp = MLP.load_network(autoencoder_model_file)
            total_mse_loaded = 0
            for x_test, y_test in test_data:
                y_test_act = loaded_mlp.forward(x_test)
                total_mse_loaded += loaded_mlp.calculate_pattern_error(y_test_act, y_test)

            avg_mse_loaded = total_mse_loaded / len(test_data) if test_data else 0
            print(f"Średni błąd MSE wczytanego modelu autoenkodera: {avg_mse_loaded:.8f}")

        except FileNotFoundError:
            print(f"Błąd: Nie znaleziono pliku modelu: {autoencoder_model_file}")
        except Exception as e:
            print(f"Błąd podczas testu wczytanego modelu autoenkodera: {e}")