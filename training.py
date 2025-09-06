from train import back_propagation
from load import load_trainings_data
from network import nn
import random

def training():
    print("Load")
    data_matrix = load_trainings_data()
    print("Loaded")

    data_sets = [(row[0], row[1:]) for row in data_matrix]
    print("Shuffle")
    random.shuffle(data_sets)
    print("Shuffled")

    print("Train")
    for data_set in data_sets:
        y, X = data_set
        print(f"Epach E, Label: {y}")
        back_propagation(X, y)
    print("Trained")

    print("Save")
    nn.save_to_pickle("after.pickle")
    print("Saved")

training()
