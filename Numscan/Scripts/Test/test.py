
# This file runs feed forward predictions for all samples and give evalution

import os
import random
import sys

fixed_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
script_path = os.path.join(fixed_path, "Scripts")

if script_path not in sys.path:
    sys.path.insert(0, script_path)

if fixed_path not in sys.path:
    sys.path.insert(0, fixed_path)

from load import load_trainings_data
from predict import feed_forward
from network import nn

model_dir = os.path.join(fixed_path, "Models")
model_path = os.path.join(model_dir, "after.pickle")

def test():
    nn.load_from_pickle(model_path)

    print("Load Data-Matrix")
    data_matrix = load_trainings_data()

    print("Shuffle and Split Data")
    data_sets = [(row[0], row[1:]) for row in data_matrix]
    random.shuffle(data_sets)
    data_sets = data_sets[:1000]
    
    right_tries = 0

    for data_set in data_sets:
        y, X = data_set

        y_p = feed_forward(X)
        if y == y_p:
            right_tries += 1
        print(f"Label: {y}, Predicted: {y_p}")

    print(f"Accuracy: {right_tries / len(data_sets)}")

test()
