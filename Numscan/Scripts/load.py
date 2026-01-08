from PIL import Image
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT_DIR, "Data", "mnist-png")

def load_trainings_data():
    data_matrix = []

    for i in range(10):
        for filename in os.listdir(os.path.join(DATA_DIR, "train", str(i))):
            path = os.path.join(DATA_DIR, "train", str(i), filename)

            img = Image.open(path).convert("L")
            # Normalize pixel intensities to [0, 1] to stabilize training
            pixels = [p / 255.0 for p in list(img.getdata())]

            data_matrix.append([i, *pixels])
        
    return data_matrix
