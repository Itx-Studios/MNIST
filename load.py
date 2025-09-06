from PIL import Image
import os

def load_trainings_data():
    data_matrix = []

    for i in range(10):
        for filename in os.listdir(os.path.join("mnist-png", "train", str(i))):
            path = os.path.join("mnist-png", "train", str(i), filename)

            img = Image.open(path).convert("L")
            pixels = list(img.getdata())

            data_matrix.append([i, *pixels])
        
    return data_matrix
