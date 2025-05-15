# main.py

from Iris_runner import run_iris_classification

def ask_float(prompt, default):
    try:
        value = input(f"{prompt} [{default}]: ").strip()
        return float(value) if value else default
    except ValueError:
        print("Niepoprawna liczba, użyto wartości domyślnej.")
        return default

def ask_int(prompt, default):
    try:
        value = input(f"{prompt} [{default}]: ").strip()
        return int(value) if value else default
    except ValueError:
        print("Niepoprawna liczba całkowita, użyto domyślnej.")
        return default

def ask_bool(prompt, default):
    value = input(f"{prompt} [{default}]: ").strip().lower()
    if value in ("true", "t", "1", "yes", "y"):
        return True
    elif value in ("false", "f", "0", "no", "n"):
        return False
    return default

def ask_architecture(prompt, default):
    value = input(f"{prompt} [{','.join(map(str, default))}]: ").strip()
    if not value:
        return default
    try:
        return [int(x) for x in value.split(",")]
    except ValueError:
        print("Błąd formatu, użyto architektury domyślnej.")
        return default

def main_menu():
    print("=== MLP dla zbioru Iris ===")
    print("Podaj parametry sieci i treningu (ENTER = wartość domyślna):\n")

    architecture = ask_architecture("Architektura sieci (np. 4,8,3)", [4, 8, 3])
    use_bias = ask_bool("Używać biasu? (True/False)", True)
    learning_rate = ask_float("Learning rate", 0.1)
    momentum = ask_float("Momentum", 0.2)
    max_epochs = ask_int("Maksymalna liczba epok", 500)
    error_threshold = ask_float("Próg błędu MSE", 0.01)
    log_interval = ask_int("Logowanie co N epok", 20)

    print("\n--- Uruchamianie klasyfikacji Irysów ---\n")

    run_iris_classification(
        architecture=architecture,
        use_bias=use_bias,
        learning_rate=learning_rate,
        momentum=momentum,
        max_epochs=max_epochs,
        error_threshold=error_threshold,
        log_interval=log_interval,
    )

if __name__ == "__main__":
    main_menu()
