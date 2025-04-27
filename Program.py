import numpy as np

from MLP import MLP

def read_from_text():
    layer_sizes_input = input("Podaj rozmiary warstw (odzielone przecinkiem): ")
    layer_sizes = list(map(int, layer_sizes_input.split(',')))
    print(f"Rozmiary warstw: {layer_sizes}")

    use_bias_input = input("Używać bias? (tak/nie): ")
    use_bias = use_bias_input.lower() == 'tak'

    input_data_input = input("Podaj dane wejściowe (np. '0.5,-0.2'): ")
    input_data = np.array(list(map(float, input_data_input.split(','))))

    while input_data.shape[0] != layer_sizes[0]:
        print(f"Niepoprawny rozmiar danych wejściowych. Oczekiwano {layer_sizes[0]}, otrzymano {input_data.shape[0]}.")
        input_data_input = input("Podaj dane wejściowe (np. '0.5,-0.2'): ")
        input_data = np.array(list(map(float, input_data_input.split(','))))

    return layer_sizes,use_bias,input_data

def read_from_file(file):
    with open(file, 'r') as f:
        layer_sizes = eval(f.readline().strip())  # Wczytanie warstw jako listy
        use_bias = eval(f.readline().strip())  # Wczytanie wartości boolean (tak/nie)
        input_data = np.array(eval(f.readline().strip()))

    return layer_sizes, use_bias, input_data


def save_to_file(layer_sizes, use_bias, input_d, output):
    savefile = input("Podaj nazwę pliku do zapisu: ")
    with open(savefile, 'w') as f:
        f.write(f"{layer_sizes}\n")
        f.write(f"{use_bias}\n")
        f.write(f"{input_d}\n")
        f.write(f"{output}\n")

def main():

    layer_sizes,use_bias,input_d = read_from_text()

    mlp = MLP(layer_sizes, use_bias=use_bias)

    output = mlp.forward(input_d)

    save_to_file(layer_sizes, use_bias, input_d, output)

if __name__ == "__main__":
    main()