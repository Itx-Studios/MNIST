
# This file runs back propergation for all samples in Data

import os
import random
import sys

fixed_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
script_path = os.path.join(fixed_path, "Scripts")
if script_path not in sys.path:
    sys.path.insert(0, script_path)
if fixed_path not in sys.path:
    sys.path.insert(0, fixed_path
)

from train import back_propagation
from load import load_trainings_data
from network import nn

model_dir = os.path.join(fixed_path, "Models")
model_path = os.path.join(model_dir, "after.pickle")

def training():
    print("load pickle? Y/N")
    if input() == "Y":
        nn.load_from_pickle(model_path)
    else:
        pass
    print("Load")
    data_matrix = load_trainings_data()
    print("Loaded")

    data_sets = [(row[0], row[1:]) for row in data_matrix]
    print("Shuffle")
    random.shuffle(data_sets)
    print("Shuffled")

    print("Train")
    for i, data_set in enumerate(data_sets):
        y, X = data_set
        print(f"Epach E: {i}, Label: {y}")
        back_propagation(X, y)
    print("Trained")

    print("Save")
    nn.save_to_pickle(model_path)
    print("Saved")

training()
