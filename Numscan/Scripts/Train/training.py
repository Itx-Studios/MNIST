import os
import random
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRIPTS_DIR = os.path.join(ROOT_DIR, "Scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from train import back_propagation
from load import load_trainings_data
from network import nn

MODELS_DIR = os.path.join(ROOT_DIR, "Models")
MODEL_PATH = os.path.join(MODELS_DIR, "after.pickle")

def training():
    print("load pickle? Y/N")
    if input() == "Y":
        nn.load_from_pickle(MODEL_PATH)
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
    nn.save_to_pickle(MODEL_PATH)
    print("Saved")

training()
