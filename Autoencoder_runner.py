# Autoencoder_runner.py
import numpy as np
from MLP import MLP
from Trainer import Trainer

# --- Konfiguracja Autoenkodera ---
AUTOENCODER_ARCHITECTURE = [4, 2, 4]  # Wejście: 4, Ukryta: 2, Wyjście: 4

# Wzorce do nauki autoasocjacji
# ((wejścia),(wyjścia))
AUTOENCODER_PATTERNS = [
    (np.array([1, 0, 0, 0], dtype=float), np.array([1, 0, 0, 0], dtype=float)),
    (np.array([0, 1, 0, 0], dtype=float), np.array([0, 1, 0, 0], dtype=float)),
    (np.array([0, 0, 1, 0], dtype=float), np.array([0, 0, 1, 0], dtype=float)),
    (np.array([0, 0, 0, 1], dtype=float), np.array([0, 0, 0, 1], dtype=float))
]

# Parametry treningu dla badania wpływu biasu
LR_BIAS_STUDY = 0.6
MOMENTUM_BIAS_STUDY = 0.0
EPOCHS_BIAS_STUDY = 2000  # Może wymagać więcej epok
ERROR_THRESH_BIAS_STUDY = 0.001  # Dość niski próg błędu
LOG_INTERVAL_AUTOENCODER = 100

# Parametry dla badania wpływu współczynnika nauki i momentum
# Użyjemy konfiguracji (z biasem lub bez), która okazała się zbieżna
LR_MOMENTUM_CONFIGS = [
    {'lr': 0.9, 'momentum': 0.0},
    {'lr': 0.6, 'momentum': 0.0},
    {'lr': 0.2, 'momentum': 0.0},
    {'lr': 0.9, 'momentum': 0.6},
    {'lr': 0.2, 'momentum': 0.9},
]
EPOCHS_LR_MOMENTUM_STUDY = 1000  # Mniej epok, żeby zobaczyć różnice w szybkości
ERROR_THRESH_LR_MOMENTUM_STUDY = 0.01


def run_autoencoder_experiments():
    print("--- Eksperymenty z Autoenkoderem ---")

    # --- Część 1: Badanie wpływu obciążenia (bias) ---
    print("\nCZĘŚĆ 1: Badanie wpływu obciążenia (bias) w neuronach nieliniowych")
    print(f"Architektura: {AUTOENCODER_ARCHITECTURE}, LR={LR_BIAS_STUDY}, Momentum={MOMENTUM_BIAS_STUDY}")
    print(f"Wzorce uczące ({len(AUTOENCODER_PATTERNS)}):")
    for i, (p_in, p_out) in enumerate(AUTOENCODER_PATTERNS):
        print(f"  {i + 1}: {p_in.tolist()} -> {p_out.tolist()}")

    results_bias_study = {}

    for use_bias_setting in [True, False]:
        print(
            f"\n--- Trening autoenkodera {'Z OBciążeniem (bias)' if use_bias_setting else 'BEZ OBCIążENIA (bias)'} ---")
        mlp_auto = MLP(layer_sizes=AUTOENCODER_ARCHITECTURE, use_bias=use_bias_setting)
        trainer_auto = Trainer(mlp_auto, AUTOENCODER_PATTERNS,
                               learning_rate=LR_BIAS_STUDY,
                               momentum=MOMENTUM_BIAS_STUDY,
                               shuffle=True)  # Losowa kolejność wzorców w epoce

        log_filename = f"autoencoder_bias_{'on' if use_bias_setting else 'off'}_train_log.csv"
        trainer_auto.train(max_epochs=EPOCHS_BIAS_STUDY,
                           error_threshold=ERROR_THRESH_BIAS_STUDY,
                           log_interval=LOG_INTERVAL_AUTOENCODER,
                           log_file=log_filename)

        # Sprawdzenie finalnego błędu
        final_epoch_error = 0
        for x, y_exp in AUTOENCODER_PATTERNS:
            y_act = mlp_auto.forward(x)
            final_epoch_error += mlp_auto.calculate_pattern_error(y_act, y_exp)
        avg_final_error = final_epoch_error / len(AUTOENCODER_PATTERNS)
        print(f"Finalny średni błąd MSE po treningu: {avg_final_error:.8f}")
        results_bias_study[use_bias_setting] = {'mlp': mlp_auto, 'final_error': avg_final_error}

        # Badanie wartości na wyjściach neuronów ukrytych
        print("\nWartości na wyjściach neuronów warstwy ukrytej po zakończeniu nauki:")
        if mlp_auto.layers:  # Sprawdzenie czy są warstwy
            hidden_layer_index = 0  # Pierwsza warstwa przetwarzająca to warstwa ukryta [4,2,4]
            for i, (pattern_in, _) in enumerate(AUTOENCODER_PATTERNS):
                mlp_auto.forward(pattern_in)  # Ustawia last_output w warstwach
                # Zakładamy, że AUTOENCODER_ARCHITECTURE = [wej, ukryta, wyj]
                # więc warstwa ukryta jest pierwszą warstwą przetwarzającą (indeks 0)
                if hidden_layer_index < len(mlp_auto.layers):
                    hidden_outputs = mlp_auto.layers[hidden_layer_index].last_output
                    print(
                        f"  Wzorzec wejściowy {i + 1} ({pattern_in.tolist()}): Aktywacje ukryte = {np.round(hidden_outputs, 4).tolist()}")
                else:
                    print("  Błąd: Nie znaleziono warstwy ukrytej do analizy.")

        # Zapiszmy sieć
        mlp_auto.save_network(f"autoencoder_model_bias_{'on' if use_bias_setting else 'off'}.pkl")

    # Wyjaśnienie wyniku - porównanie zbieżności
    print("\n--- Podsumowanie wpływu biasu ---")
    converged_with_bias = results_bias_study[True]['final_error'] <= ERROR_THRESH_BIAS_STUDY
    converged_without_bias = results_bias_study[False]['final_error'] <= ERROR_THRESH_BIAS_STUDY

    print(
        f"Zbieżność z biasem (błąd <= {ERROR_THRESH_BIAS_STUDY}): {'TAK' if converged_with_bias else 'NIE'} (finalny błąd: {results_bias_study[True]['final_error']:.8f})")
    print(
        f"Zbieżność bez biasu (błąd <= {ERROR_THRESH_BIAS_STUDY}): {'TAK' if converged_without_bias else 'NIE'} (finalny błąd: {results_bias_study[False]['final_error']:.8f})")


    best_config_use_bias = None
    if converged_with_bias and (
            not converged_without_bias or results_bias_study[True]['final_error'] < results_bias_study[False][
        'final_error']):
        best_config_use_bias = True
        print("Wybrano konfigurację Z OBciążeniem do dalszych testów.")
    elif converged_without_bias:
        best_config_use_bias = False
        print("Wybrano konfigurację BEZ OBciążenia do dalszych testów.")
    else:
        print(
            "Żadna konfiguracja (z/bez biasu) nie osiągnęła zadowalającej zbieżności. Dalsze testy mogą być niemiarodajne.")
        # Mimo wszystko, wybierzmy jedną, np. z biasem, do dalszych testów
        best_config_use_bias = True  # Domyślnie, jeśli obie nie zbiegły lub obie zbiegły tak samo.
        print("Domyślnie wybrano konfigurację Z OBciążeniem do dalszych testów.")

    # --- Część 2: Badanie szybkości uczenia (LR, Momentum) ---
    if best_config_use_bias is not None:
        print(
            f"\nCZĘŚĆ 2: Badanie szybkości uczenia dla konfiguracji {'Z OBciążeniem' if best_config_use_bias else 'BEZ OBCIążENIA'}")

        for config_idx, params in enumerate(LR_MOMENTUM_CONFIGS):
            lr = params['lr']
            momentum_val = params['momentum']

            print(f"\n--- Trening autoenkodera: LR={lr}, Momentum={momentum_val} ---")
            mlp_speed_test = MLP(layer_sizes=AUTOENCODER_ARCHITECTURE, use_bias=best_config_use_bias)
            trainer_speed_test = Trainer(mlp_speed_test, AUTOENCODER_PATTERNS,
                                         learning_rate=lr,
                                         momentum=momentum_val,
                                         shuffle=True)

            log_filename_speed = f"autoencoder_lr{lr}_mom{momentum_val}_bias{best_config_use_bias}_train_log.csv"
            trainer_speed_test.train(max_epochs=EPOCHS_LR_MOMENTUM_STUDY,
                                     error_threshold=ERROR_THRESH_LR_MOMENTUM_STUDY,
                                     # Wyższy próg, by zobaczyć różnice w czasie
                                     log_interval=50,  # Rzadsze logowanie
                                     log_file=log_filename_speed)

            # Ocena po treningu
            final_epoch_error_speed = 0
            for x, y_exp in AUTOENCODER_PATTERNS:
                y_act = mlp_speed_test.forward(x)
                final_epoch_error_speed += mlp_speed_test.calculate_pattern_error(y_act, y_exp)
            avg_final_error_speed = final_epoch_error_speed / len(AUTOENCODER_PATTERNS)
            print(f"Finalny średni błąd MSE po treningu (LR={lr}, Mom={momentum_val}): {avg_final_error_speed:.8f}")
            # Można tu zbierać dane o liczbie epok do zbieżności i porównywać je.
            # Analiza plików *_train_log.csv będzie tu kluczowa.
    else:
        print("\nCZĘŚĆ 2 pominięta, ponieważ nie udało się ustalić zbieżnej konfiguracji z CZĘŚCI 1.")

    print("\n--- Zakończono eksperymenty z Autoenkoderem ---")


if __name__ == "__main__":
    # Możesz wybrać, który runner uruchomić:
    run_autoencoder_experiments()