from load import load_trainings_data
from predict import feed_forward

def test():
    data_matrix = load_trainings_data()

    data_sets = [(row[0], row[1:]) for row in data_matrix]
    
    right_tries = 0

    for data_set in data_sets:
        y, X = data_set

        y_p = feed_forward(X)
        if y == y_p:
            right_tries += 1
        print(f"Label: {y}, Predicted: {y_p}")

    print(f"Accuracy: {right_tries / len(data_sets)}")

test()
