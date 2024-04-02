import matplotlib.pyplot as plt
import json

def load_numbers(file_path):
    with open(file_path, 'r') as f:
        numbers = json.load(f)
    return numbers

def plot_numbers(numbers, label):
    plt.plot(numbers, label=label)

numbers1 = load_numbers('./out/1e-4.out')
numbers2 = load_numbers('./out/1e-5.out')
numbers3 = load_numbers('./out/1e-6.out')
plot_numbers(numbers1, '1e-4')
plot_numbers(numbers2, '1e-5')
plot_numbers(numbers3, '1e-6')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.savefig('plot.png')