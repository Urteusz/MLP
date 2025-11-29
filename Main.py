from runner import run_classification
import argparse
import sys

def ask_float(prompt, default):
    try:
        value = input(f"{prompt} [{default}]: ").strip()
        return float(value) if value else default
    except ValueError:
        return default

def ask_float_none(prompt, default=None):
    try:
        value = input(f"{prompt} [{default}]: ").strip()
        if prompt == "":
            return None
        return float(value) if value else default
    except ValueError:
        return default

def ask_int(prompt, default):
    try:
        value = input(f"{prompt} [{default}]: ").strip()
        return int(value) if value else default
    except ValueError:
        return default

def ask_bool(prompt, default):
    value = input(f"{prompt} [{default}]: ").strip().lower()
    if value in ("true", "t", "1", "yes", "y"):
        return True
    if value in ("false", "f", "0", "no", "n"):
        return False
    return default

def ask_architecture(prompt, default):
    value = input(f"{prompt} [{','.join(map(str, default))}]: ").strip()
    if not value:
        return default
    try:
        return [int(x) for x in value.split(",")]
    except ValueError:
        return default

MODE_DEFAULT_ARCH = {
    'iris': [4, 8, 3],
    'autoencoder': [4, 2, 4]
}

def parse_architecture_string(s: str):
    parts = [p.strip() for p in s.split(',') if p.strip()]
    if not parts:
        raise ValueError("Architektura nie może być pusta.")
    arch = []
    for p in parts:
        if not p.isdigit():
            raise ValueError(f"Element architektury '{p}' nie jest dodatnią liczbą całkowitą.")
        v = int(p)
        if v <= 0:
            raise ValueError("Rozmiary warstw muszą być dodatnie.")
        arch.append(v)
    return arch

def validate_architecture(mode: str, arch: list):
    if mode == 'iris':
        if arch[0] != 4:
            raise ValueError("Pierwsza warstwa dla Iris musi mieć rozmiar 4.")
        if arch[-1] != 3:
            raise ValueError("Ostatnia warstwa dla Iris musi mieć rozmiar 3.")
    elif mode == 'autoencoder':
        if arch[0] != arch[-1]:
            raise ValueError("Wejście i wyjście autoenkodera muszą być równe.")
        if arch[0] != 4:
            raise ValueError("Autoencoder zakłada wejście/wyjście 4.")

def run_mlp_menu(choose):
    architecture = ask_architecture("Architektura sieci (np. 4,8,3)", [4, 8, 3] if choose == 'Iris' else [4, 2, 4])
    use_bias = ask_bool("Używać biasu? (True/False)", True)
    learning_rate = ask_float("Learning rate", 0.1)
    momentum = ask_float("Momentum", 0.2)
    max_epochs = ask_int("Maksymalna liczba epok", 1000 if choose == 'Iris' else 3000)
    error_threshold = ask_float_none("Próg błędu MSE", None if choose == 'Iris' else 0.000001)
    log_interval = ask_int("Logowanie co N epok", 1)
    run_classification(name=choose,
        architecture=architecture,
        use_bias=use_bias,
        learning_rate=learning_rate,
        momentum=momentum,
        max_epochs=max_epochs,
        error_threshold=error_threshold,
        log_interval=log_interval,
        trained_model_file=f"{choose}_model.pkl")

def run_mlp(choose, architecture, learning_rate, momentum, use_bias=True, max_epochs=1000, error_threshold=0.005, log_interval=1, model_file_override=None):
    trained_file = model_file_override if model_file_override else f"{choose}_model.pkl"
    run_classification(name=choose,
                       architecture=architecture,
                       use_bias=use_bias,
                       learning_rate=learning_rate,
                       momentum=momentum,
                       max_epochs=max_epochs,
                       error_threshold=error_threshold,
                       log_interval=log_interval,
                       trained_model_file=trained_file)

BATCH_RUNS = [
    {"choose": "Iris", "architecture": [4, 8, 3], "learning_rate": 0.9, "momentum": 0.0, "use_bias": True, "max_epochs": 1000, "error_threshold": None, "log_interval": 1},
    {"choose": "Iris", "architecture": [4, 8, 3], "learning_rate": 0.6, "momentum": 0.0, "use_bias": True, "max_epochs": 1000, "error_threshold": None, "log_interval": 1},
    {"choose": "Iris", "architecture": [4, 8, 3], "learning_rate": 0.2, "momentum": 0.0, "use_bias": True, "max_epochs": 1000, "error_threshold": None, "log_interval": 1},
    {"choose": "Iris", "architecture": [4, 8, 3], "learning_rate": 0.9, "momentum": 0.6, "use_bias": True, "max_epochs": 1000, "error_threshold": None, "log_interval": 1},
    {"choose": "Iris", "architecture": [4, 8, 3], "learning_rate": 0.2, "momentum": 0.9, "use_bias": True, "max_epochs": 1000, "error_threshold": None, "log_interval": 1},
    {"choose": "Autoencoder", "architecture": [4, 2, 4], "learning_rate": 0.9, "momentum": 0.0, "use_bias": True, "max_epochs": 3000, "error_threshold": 0.000001, "log_interval": 1},
    {"choose": "Autoencoder", "architecture": [4, 2, 4], "learning_rate": 0.6, "momentum": 0.0, "use_bias": True, "max_epochs": 3000, "error_threshold": 0.000001, "log_interval": 1},
    {"choose": "Autoencoder", "architecture": [4, 2, 4], "learning_rate": 0.2, "momentum": 0.0, "use_bias": True, "max_epochs": 3000, "error_threshold": 0.000001, "log_interval": 1},
    {"choose": "Autoencoder", "architecture": [4, 2, 4], "learning_rate": 0.9, "momentum": 0.6, "use_bias": True, "max_epochs": 3000, "error_threshold": 0.000001, "log_interval": 1},
    {"choose": "Autoencoder", "architecture": [4, 2, 4], "learning_rate": 0.2, "momentum": 0.9, "use_bias": True, "max_epochs": 3000, "error_threshold": 0.000001, "log_interval": 1},
]

def run_batch_report(model_file_prefix=None, overrides=None):
    print("=== Seria uruchomień (batch) ===")
    for i, base_cfg in enumerate(BATCH_RUNS, start=1):
        cfg = base_cfg.copy()
        if overrides:
            for key in ["architecture", "learning_rate", "momentum", "use_bias", "max_epochs", "error_threshold", "log_interval"]:
                if key in overrides and overrides[key] is not None:
                    cfg[key] = overrides[key]
            if "architecture" in overrides and overrides.get("architecture") is not None:
                mode_lower = 'iris' if cfg['choose'] == 'Iris' else 'autoencoder'
                validate_architecture(mode_lower, cfg['architecture'])
        print(f"\n[Batch {i}/{len(BATCH_RUNS)}] {cfg['choose']} lr={cfg['learning_rate']} mom={cfg['momentum']}")
        model_file = None
        if model_file_prefix:
            model_file = f"{model_file_prefix}_{cfg['choose'].lower()}_{cfg['learning_rate']}_{cfg['momentum']}.pkl"
        run_mlp(cfg['choose'], cfg['architecture'], cfg['learning_rate'], cfg['momentum'], use_bias=cfg['use_bias'], max_epochs=cfg['max_epochs'], error_threshold=cfg['error_threshold'], log_interval=cfg['log_interval'], model_file_override=model_file)

def build_arg_parser():
    p = argparse.ArgumentParser(description="MLP / Autoencoder")
    p.add_argument('--mode', choices=['iris', 'autoencoder', 'batch', 'all'])
    p.add_argument('--architecture')
    p.add_argument('--learning-rate', type=float)
    p.add_argument('--momentum', type=float)
    p.add_argument('--no-bias', action='store_true')
    p.add_argument('--max-epochs', type=int)
    p.add_argument('--error-threshold', type=float)
    p.add_argument('--log-interval', type=int)
    p.add_argument('--model-file')
    p.add_argument('--batch-prefix')
    return p

def handle_cli(args):
    mode = args.mode
    if not mode:
        return False
    if mode in ('iris', 'autoencoder'):
        arch = parse_architecture_string(args.architecture) if args.architecture else MODE_DEFAULT_ARCH[mode]
        validate_architecture(mode, arch)
        lr = args.learning_rate if args.learning_rate is not None else 0.1
        mom = args.momentum if args.momentum is not None else 0.2
        if lr <= 0:
            raise ValueError("Learning rate musi być > 0.")
        if mom < 0 or mom > 1:
            raise ValueError("Momentum w [0,1].")
        max_epochs = args.max_epochs if args.max_epochs is not None else (1000 if mode == 'iris' else 3000)
        if max_epochs <= 0:
            raise ValueError("max_epochs > 0")
        err_thr = args.error_threshold if args.error_threshold is not None else (None if mode == 'iris' else 0.000001)
        log_int = args.log_interval if args.log_interval is not None else 1
        use_bias = not args.no_bias
        model_file = args.model_file
        name = 'Iris' if mode == 'iris' else 'Autoencoder'
        run_mlp(name, arch, lr, mom, use_bias=use_bias, max_epochs=max_epochs, error_threshold=err_thr, log_interval=log_int, model_file_override=model_file)
        return True
    if mode == 'batch':
        overrides = {}
        if args.architecture:
            overrides['architecture'] = parse_architecture_string(args.architecture)
        overrides['learning_rate'] = args.learning_rate
        overrides['momentum'] = args.momentum
        overrides['use_bias'] = (False if args.no_bias else None)
        overrides['max_epochs'] = args.max_epochs
        overrides['error_threshold'] = args.error_threshold
        overrides['log_interval'] = args.log_interval
        run_batch_report(model_file_prefix=args.batch_prefix, overrides=overrides)
        return True
    if mode == 'all':
        overrides = {}
        if args.architecture:
            overrides['architecture'] = parse_architecture_string(args.architecture)
        overrides['learning_rate'] = args.learning_rate
        overrides['momentum'] = args.momentum
        overrides['use_bias'] = (False if args.no_bias else None)
        overrides['max_epochs'] = args.max_epochs
        overrides['error_threshold'] = args.error_threshold
        overrides['log_interval'] = args.log_interval
        handle_cli(argparse.Namespace(mode='iris', architecture=args.architecture, learning_rate=args.learning_rate, momentum=args.momentum, no_bias=args.no_bias, max_epochs=args.max_epochs, error_threshold=args.error_threshold, log_interval=args.log_interval, model_file=args.model_file, batch_prefix=args.batch_prefix))
        handle_cli(argparse.Namespace(mode='autoencoder', architecture=args.architecture, learning_rate=args.learning_rate, momentum=args.momentum, no_bias=args.no_bias, max_epochs=args.max_epochs, error_threshold=args.error_threshold, log_interval=args.log_interval, model_file=args.model_file, batch_prefix=args.batch_prefix))
        run_batch_report(model_file_prefix=args.batch_prefix, overrides=overrides)
        return True
    return False

def main():
    parser = build_arg_parser()
    if len(sys.argv) > 1:
        args = parser.parse_args()
        try:
            if handle_cli(args):
                return
        except Exception as e:
            print(f"Błąd CLI: {e}")
            return
    choose = input("Wybierz opcję:\n1. Klasyfikacja Irysów\n2. Autoenkoder\n3. Batch \n4. Wyjście\n")
    while choose not in ['1', '2', '3', '4']:
        choose = input("Wybierz opcję:\n1. Klasyfikacja Irysów\n2. Autoenkoder\n3. Batch \n4. Wyjście\n")
    if choose == '1':
        run_mlp_menu("Iris")
    elif choose == '2':
        run_mlp_menu("Autoencoder")
    elif choose == '3':
        run_batch_report()
    else:
        print("Koniec.")

if __name__ == "__main__":
    main()
