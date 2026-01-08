
# This file uses the Tensorflow library to train a Convolutional Neural Network model

import os
import pickle

import tensorflow as tf
from tensorflow.keras import datasets, layers, models

MODEL_PATH = "Numscan 2\Models\model.pkl"

# Data stuff
def load_mnist_data():
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1)).astype("float32") / 255.0
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)).astype("float32") / 255.0
    return (x_train, y_train), (x_test, y_test)

# Model stuff
def build_model():
    model = models.Sequential(
        [
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

# Save / Load model 
def save_model_pickle(model, path=MODEL_PATH):
    with open(path, "wb") as handle:
        pickle.dump(model.get_weights(), handle)

def load_model_pickle(path=MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Pickle file '{path}' does not exist.")
    
    model = build_model()
    with open(path, "rb") as handle:
        weights = pickle.load(handle)
        
    model.set_weights(weights)
    return model

# Main loop
if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = load_mnist_data()

    if os.path.exists(MODEL_PATH):
        print(f"Loading model weights from '{MODEL_PATH}'")
        model = load_model_pickle(MODEL_PATH)
    else:
        print("Training new model...")
        model = build_model()
        model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1)
        save_model_pickle(model, MODEL_PATH)

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"\nTest accuracy: {test_acc:.4f}")
