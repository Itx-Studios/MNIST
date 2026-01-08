import json
import pickle
import random
from dataclasses import dataclass

@dataclass
class NetworkStore:
    weights: list[list[list[float]]]
    biases: list[list[float]]

def init(*layer_sizes: int) -> NetworkStore:
    n = NetworkStore([], [])
    set_random(n, layer_sizes)
    return n

def set_random(n: NetworkStore, layer_sizes: tuple[int, ...]):
    for i in range(len(layer_sizes) - 1):
        rows = layer_sizes[i + 1]
        cols = layer_sizes[i]
        
        weight_matrix = []
        for _ in range(rows):
            row = [random.uniform(-1, 1) for _ in range(cols)]
            weight_matrix.append(row)
        n.weights.append(weight_matrix)
        
        bias_vector = [random.uniform(-1, 1) for _ in range(rows)]
        n.biases.append(bias_vector)

def get_weights(n: NetworkStore, layer: int) -> list[list[float]]:
    return n.weights[layer]

def get_biases(n: NetworkStore, layer: int) -> list[float]:
    return n.biases[layer]

def set_weights(n: NetworkStore, layer: int, weights: list[list[float]]):
    n.weights[layer] = weights

def set_biases(n: NetworkStore, layer: int, biases: list[float]):
    n.biases[layer] = biases

def save_pickle(n: NetworkStore, filename: str):
    data = {
        'weights': n.weights,
        'biases': n.biases
    }
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(n: NetworkStore, filename: str):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    
    n.weights = data['weights']
    n.biases = data['biases']

def save_to_json(n: NetworkStore, filename: str):
    data = {
        'weights': n.weights,
        'biases': n.biases
    }
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    
def load_from_json(n: NetworkStore, filename: str):
    with open(filename, 'r') as f:
        data = json.load(f)
    
    n.weights = data['weights']
    n.biases = data['biases']


store = init(784, 128, 128, 10)
