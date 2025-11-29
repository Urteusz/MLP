import numpy as np
import os
from MLP import MLP
from Trainer import Trainer
from Data_utils import load_iris_data, normalize_features, split_dataset

# Ustal katalog bazowy (bieżący plik), aby poprawnie znaleźć iris.data niezależnie od cwd
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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
    log_dir = os.path.join("logs", name.lower())
    model_dir = os.path.join("models", name.lower())
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    training_log_file = os.path.join(log_dir, f"{name}_lr{learning_rate}_mom{momentum}_bias{use_bias}.csv")
    print(f"--- Uruchamianie dla: {name} ---")

    if name == "Iris":
        IRIS_DATA_FILE = os.path.join(BASE_DIR, "iris.data")
        IRIS_CLASS_NAMES = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
        dataset = load_iris_data(IRIS_DATA_FILE)
        if not dataset:
            print("Brak danych (iris.data).")
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
        IRIS_CLASS_NAMES = None
        if architecture is None:
            architecture = [4, 2, 4]
            print(f"Domyślna architektura {architecture}")
    else:
        print("Nieznane zadanie.")
        return

    mlp = MLP(layer_sizes=architecture, use_bias=use_bias)
    trainer = Trainer(mlp, train_data, learning_rate=learning_rate, momentum=momentum)
    avg_final_error_train = None
    trainer.train(error_threshold=error_threshold, max_epochs=max_epochs, log_interval=log_interval, log_file=training_log_file)

    if name == "Autoencoder":
        temp_error_sum = 0
        for x, y_exp in train_data:
            y_act = mlp.forward(x)
            temp_error_sum += mlp.calculate_pattern_error(y_act, y_exp)
        if train_data:
            avg_final_error_train = temp_error_sum / len(train_data)

    if name == "Iris":
        base_name = trained_model_file if trained_model_file else "iris_mlp_model.pkl"
        saved_model_path = base_name if os.path.dirname(base_name) else os.path.join(model_dir, base_name)
        mlp.save_network(saved_model_path)
    else:
        suffix = "_bias_on" if use_bias else "_bias_off"
        arch_str = "_".join(map(str, architecture)) if architecture else "unknown_arch"
        auto_name_default = f"autoencoder_{arch_str}{suffix}.pkl"
        base_name = trained_model_file if (trained_model_file and trained_model_file != "iris_mlp_model.pkl") else auto_name_default
        autoencoder_model_file = base_name if os.path.dirname(base_name) else os.path.join(model_dir, base_name)
        mlp.save_network(autoencoder_model_file)

    if name == "Iris":
        print("\n--- Ewaluacja ---")
        metrics_test = trainer.calculate_classification_metrics(test_data, class_names=IRIS_CLASS_NAMES)
        print("\n--- Przykładowe predykcje ---")
        for i in range(min(5, len(test_data))):
            x_sample, y_expected = test_data[i]
            y_actual = mlp.forward(x_sample)
            pred_idx = np.argmax(y_actual)
            actual_idx = np.argmax(y_expected)
            print(f"{i + 1}: in={np.round(x_sample, 2)} exp={IRIS_CLASS_NAMES[actual_idx]} pred={IRIS_CLASS_NAMES[pred_idx]} prob={np.round(y_actual,3)}")
        print("\n--- Test zapisu ---")
        try:
            loaded_mlp_iris = MLP.load_network(saved_model_path)
            trainer_loaded_iris = Trainer(loaded_mlp_iris, test_data)
            _ = trainer_loaded_iris.calculate_classification_metrics(test_data, class_names=IRIS_CLASS_NAMES)
        except Exception as e:
            print(f"Błąd wczytania modelu: {e}")
    else:
        if avg_final_error_train is not None:
            print(f"Finalny MSE: {avg_final_error_train:.8f}")
        print("\n--- Wzorce ---")
        hidden_layer_index = 0
        for i, (pattern_in, expected_out) in enumerate(train_data):
            reconstructed_out = mlp.forward(pattern_in)
            hidden_representation_str = "N/A"
            if mlp.layers and len(mlp.layers) > hidden_layer_index and hasattr(mlp.layers[hidden_layer_index], 'last_output'):
                hidden_activations = mlp.layers[hidden_layer_index].last_output
                if hidden_activations is not None:
                    hidden_representation_str = str(np.round(hidden_activations, 4))
            pattern_mse = mlp.calculate_pattern_error(reconstructed_out, expected_out)
            print(f"{i + 1}: in={np.round(pattern_in,4)} hid={hidden_representation_str} out={np.round(reconstructed_out,4)} mse={pattern_mse:.8f}")
        print("\n--- Test zapisu ---")
        try:
            loaded_mlp_autoencoder = MLP.load_network(autoencoder_model_file)
            total_mse_loaded_model = 0
            for x_test, y_test_expected in test_data:
                y_test_actual = loaded_mlp_autoencoder.forward(x_test)
                total_mse_loaded_model += loaded_mlp_autoencoder.calculate_pattern_error(y_test_actual, y_test_expected)
            avg_mse_loaded_model = total_mse_loaded_model / len(test_data)
            print(f"MSE wczytanego: {avg_mse_loaded_model:.8f}")
        except Exception as e:
            print(f"Błąd wczytania modelu: {e}")
