import os
import random
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRIPTS_DIR = os.path.join(ROOT_DIR, "Scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from load import load_trainings_data
from predict import feed_forward
from network import nn

MODELS_DIR = os.path.join(ROOT_DIR, "Models")
MODEL_PATH = os.path.join(MODELS_DIR, "after.pickle")

def test():
    nn.load_from_pickle(MODEL_PATH)

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
