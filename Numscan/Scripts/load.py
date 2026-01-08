
# This file loads all images and converts the pixelvalues into a matrix

from PIL import Image
import os

fixed_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_dir = os.path.join(fixed_path, "Data", "mnist-png")

def load_trainings_data():
    data_matrix = []

    for i in range(10):

        for filename in os.listdir(os.path.join(data_dir, "train", str(i))):
            path = os.path.join(data_dir, "train", str(i), filename)

            img = Image.open(path).convert("L")
            pixels = [p / 255.0 for p in list(img.getdata())]

            data_matrix.append([i, *pixels])
        
    return data_matrix
