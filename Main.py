# main.py

from runner import run_classification

def ask_float(prompt, default):
    try:
        value = input(f"{prompt} [{default}]: ").strip()
        return float(value) if value else default
    except ValueError:
        print("Niepoprawna liczba, użyto wartości domyślnej.")
        return default

def ask_float_none(prompt, default=None):
    try:
        value = input(f"{prompt} [{default}]: ").strip()
        if prompt == "":
            return None
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

def run_mlp_menu(choose):
    print("=== MLP dla zbioru Iris ===")
    print("Podaj parametry sieci i treningu (ENTER = wartość domyślna):\n")

    architecture = ask_architecture("Architektura sieci (np. 4,8,3)", [4, 8, 3])
    use_bias = ask_bool("Używać biasu? (True/False)", True)
    learning_rate = ask_float("Learning rate", 0.1)
    momentum = ask_float("Momentum", 0.2)
    max_epochs = ask_int("Maksymalna liczba epok", 1000)
    error_threshold = ask_float_none("Próg błędu MSE", None)
    log_interval = ask_int("Logowanie co N epok", 1)

    print("\n--- Uruchamianie klasyfikacji Irysów ---\n")

    run_classification(name=choose,
        architecture=architecture,
        use_bias=use_bias,
        learning_rate=learning_rate,
        momentum=momentum,
        max_epochs=max_epochs,
        error_threshold=error_threshold,
        log_interval=log_interval,
        trained_model_file=f"{choose}_model.pkl"
    )

def run_mlp(choose,
        architecture,
        learning_rate,
        momentum,
        use_bias=True,
        max_epochs=1000,
        error_threshold=0.005,
        log_interval=1):
    run_classification(name=choose,
                       architecture=architecture,
                       use_bias=use_bias,
                       learning_rate=learning_rate,
                       momentum=momentum,
                       max_epochs=max_epochs,
                       error_threshold=error_threshold,
                       log_interval=log_interval,
                       trained_model_file=f"{choose}_model.pkl"
                       )

def main():
    choose = input("Wybierz opcję:\n1. Klasyfikacja Irysów\n2. Autoenkoder\n3. Uruchomienia do sprawozdania \n4. Wyjście\n")
    while choose not in ['1', '2', '3','4']:
        print("Niepoprawny wybór, spróbuj ponownie.")
        choose = input("Wybierz opcję:\n1. Klasyfikacja Irysów\n2. Autoenkoder\n3. Uruchomienia do sprawozdania \n4. Wyjście\n")
    if choose == '1':
        run_mlp_menu("Iris")
    elif choose == '2':
        run_mlp_menu("Autoencoder")
    elif choose == '3':
        run_mlp("Iris",
                [4, 8, 3],
                0.9,
                0.0)
        run_mlp("Iris",
                [4, 8, 3],
                0.6,
                0.0)
        run_mlp("Iris",
                [4, 8, 3],
                0.2,
                0.0)
        run_mlp("Iris",
                [4, 8, 3],
                0.9,
                0.6)
        run_mlp("Iris",
                [4, 8, 3],
                0.2,
                0.9)
        run_mlp("Autoencoder",
                [4, 2, 4],
                0.9,
                0.0,
                True,
                3000,
                0.000001,
                1)
        run_mlp("Autoencoder",
                [4, 2, 4],
                0.6,
                0.0,
                True,
                3000,
                0.000001,
                1)
        run_mlp("Autoencoder",
                [4, 2, 4],
                0.2,
                0.0,
                True,
                3000,
                0.000001,
                1)
        run_mlp("Autoencoder",
                [4, 2, 4],
                0.9,
                0.6,
                True,
                3000,
                0.000001,
                1)
        run_mlp("Autoencoder",
                [4, 2, 4],
                0.2,
                0.9,
                True,
                3000,
                0.000001,
                1)
    elif choose == '4':
        print("Zamykam program...")
        return





if __name__ == "__main__":
    main()
