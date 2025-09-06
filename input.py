from PIL import Image
import os

def load_images_to_matrix(folder):
    matrix = []
    for filename in os.listdir(folder):
        if filename.endswith(".png"):
            path = os.path.join(folder, filename)

            img = Image.open(path).convert("L")

            pixels = list(img.getdata())


            matrix.append(pixels)

    matrix_T = list(map(list, zip(*matrix)))
    return matrix


data_matrix = load_images_to_matrix("./png")
