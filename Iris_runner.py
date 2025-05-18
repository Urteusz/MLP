# Iris_runner.py
import numpy as np
from MLP import MLP
from Trainer import Trainer
from Data_utils import load_iris_data, normalize_features, split_dataset


def run_iris_classification(
        architecture=None,
        use_bias=True,
        learning_rate=0.1,
        momentum=0.2,
        max_epochs=500,
        error_threshold=0.01,
        log_interval=20,
        train_ratio=0.7,
        random_seed=42,
        trained_model_file="iris_mlp_model.pkl",
        training_log_file="iris_training_log.csv"
):
    training_log_file = (
        f"iris_lr{learning_rate}_mom{momentum}_bias{use_bias}.csv"
    )

    if architecture is None:
        architecture = [4, 8, 3]
    print("--- Klasyfikacja zbioru Irysów (parametryzowana) ---")

    IRIS_DATA_FILE = "iris.data"
    IRIS_CLASS_NAMES = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

    # 1. Wczytanie i przygotowanie danych
    dataset = load_iris_data(IRIS_DATA_FILE)
    if not dataset:
        print("Nie udało się wczytać danych. Przerwanie.")
        return

    dataset = normalize_features(dataset)
    train_data, test_data = split_dataset(dataset, train_ratio=train_ratio, seed=random_seed)

    # 2. Budowa sieci
    mlp = MLP(layer_sizes=architecture, use_bias=use_bias)

    # 3. Trening
    trainer = Trainer(mlp, train_data,
                      learning_rate=learning_rate,
                      momentum=momentum,
                      shuffle=True)

    if max_epochs is None:
        trainer.train(error_threshold=error_threshold,
                      max_epochs=max_epochs,
                      log_interval=log_interval,
                      log_file=training_log_file)
    else:
        trainer.train(error_threshold=error_threshold,
                      log_interval=log_interval,
                      max_epochs=max_epochs,
                      log_file=training_log_file)


    mlp.save_network(trained_model_file)

    # 4. Ewaluacja
    print("\n--- Ewaluacja ---")
    trainer.calculate_classification_metrics(train_data, class_names=IRIS_CLASS_NAMES)
    metrics_test = trainer.calculate_classification_metrics(test_data, class_names=IRIS_CLASS_NAMES)

    # 5. Przykładowe predykcje
    print("\n--- Przykładowe predykcje ---")
    for i in range(min(5, len(test_data))):
        x_sample, y_expected = test_data[i]
        y_actual = mlp.forward(x_sample)
        pred_idx = np.argmax(y_actual)
        actual_idx = np.argmax(y_expected)
        print(f"Wzorzec {i + 1}:")
        print(f"  Wejście: {np.round(x_sample, 2)}")
        print(f"  Oczekiwane: {IRIS_CLASS_NAMES[actual_idx]}")
        print(f"  Przewidziane: {IRIS_CLASS_NAMES[pred_idx]} | Prawdopodobieństwa: {np.round(y_actual, 3)}")
        print(f"  {'✔️' if pred_idx == actual_idx else '❌'}")

    # 6. Test modelu z pliku
    print("\n--- Test zapisanego modelu ---")
    try:
        loaded_mlp = MLP.load_network(trained_model_file)
        trainer_loaded = Trainer(loaded_mlp, test_data)
        metrics_loaded = trainer_loaded.calculate_classification_metrics(test_data,
                                                                         class_names=IRIS_CLASS_NAMES)
        print(f"Dokładność wczytanego modelu: {metrics_loaded['accuracy']:.4f}")
    except Exception as e:
        print(f"Błąd podczas testu wczytanego modelu: {e}")

    print("\n--- Zakończono klasyfikację Irysów ---")

if __name__ == "__main__":
    run_iris_classification()