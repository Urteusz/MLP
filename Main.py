choose = input("Wybierz 1. Klasyfikacja zbioru Irysów, 2. Autoasocjacja: ").strip()
if choose == '1':
    from Iris_runner import run_iris_classification
    run_iris_classification()
elif choose == '2':
    from Autoencoder_runner import run_autoencoder_experiments
    run_autoencoder_experiments()