import numpy as np

from MLP import MLP

def main():
    layer_sizes_input = input("Podaj rozmiary warstw (odzielone przecinkiem): ")
    layer_sizes = list(map(int, layer_sizes_input.split(',')))
    print (f"Rozmiary warstw: {layer_sizes}")

    use_bias_input = input("Używać bias? (tak/nie): ")
    use_bias = use_bias_input.lower() == 'tak'

    mlp = MLP(layer_sizes, use_bias=use_bias)


    input_data_input = input("Podaj dane wejściowe (np. '0.5,-0.2'): ")
    input_data = np.array(list(map(float, input_data_input.split(','))))

    while input_data.shape[0] != layer_sizes[0]:
        print(f"Niepoprawny rozmiar danych wejściowych. Oczekiwano {layer_sizes[0]}, otrzymano {input_data.shape[0]}.")
        input_data_input = input("Podaj dane wejściowe (np. '0.5,-0.2'): ")
        input_data = np.array(list(map(float, input_data_input.split(','))))

    output = mlp.forward(input_data)

    savefile = input("Podaj nazwę pliku do zapisu: ")
    with open(savefile, 'w') as f:
        f.write(f"{layer_sizes}\n")
        f.write(f"{use_bias}\n")
        f.write(f"{input_data}\n")
        f.write(f"{output}\n")

    

if __name__ == "__main__":
    main()