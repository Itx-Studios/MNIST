import json
import pickle
import random
import math

class NeuralNetworkStorage:
    def __init__(self, *layer_sizes):
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        
        self.weights = []
        self.biases = []
        
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        for i in range(self.num_layers - 1):
            input_size = self.layer_sizes[i]
            output_size = self.layer_sizes[i + 1]
            
            limit = math.sqrt(6.0 / (input_size + output_size))
            
            weight_matrix = []
            for _ in range(output_size):
                row = []
                for _ in range(input_size):
                    weight = random.uniform(-limit, limit)
                    row.append(weight)
                weight_matrix.append(row)
            
            self.weights.append(weight_matrix)
            
            bias_vector = [0.0 for _ in range(output_size)]
            self.biases.append(bias_vector)
    
    def get_weights(self, layer_index):
        return self.weights[layer_index]
    
    def get_biases(self, layer_index):
        return self.biases[layer_index]
    
    def set_weights(self, layer_index, weight_matrix):
        self.weights[layer_index] = weight_matrix
    
    def set_biases(self, layer_index, bias_vector):
        self.biases[layer_index] = bias_vector

    def save_to_json(self, filename):
        data = {
            'layer_sizes': self.layer_sizes,
            'weights': self.weights,
            'biases': self.biases
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_from_json(self, filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        
        self.layer_sizes = data['layer_sizes']
        self.weights = data['weights']
        self.biases = data['biases']
        self.num_layers = len(self.layer_sizes)
    
    def save_to_pickle(self, filename):
        data = {
            'layer_sizes': self.layer_sizes,
            'weights': self.weights,
            'biases': self.biases
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
    
    def load_from_pickle(self, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        self.layer_sizes = data['layer_sizes']
        self.weights = data['weights']
        self.biases = data['biases']
        self.num_layers = len(self.layer_sizes)
    
nn = NeuralNetworkStorage(784, 128, 128, 10)