from predict import exec
from network import nn

def MSE(result: list[int], label: int):
    error = 0
    for (i, item) in enumerate(result):
        if i == label:
            error += (item - 1) ** 2
        elif i != label:
            error += (item - 0) ** 2
    return error / len(result)

def ground_truth_vec(label):
    return [1 if i == label else 0 for i in range(10)]

def vec_sub(v1, v2):
    for i in range(len(v1) - 1):
        v1[i] -= v2[i]

def vec_mul(v1, v2):
    for i in range(len(v1) - 1):
        v1[i] *= v2[i]

def vec_factor(v, x):
    for y in v:
        y *= x

def outer_p(v1, v2):
    return [[a * b for b in v2] for a in v1]

def T(m):
    return [list(x) for x in zip(*m)]

def mm(matrix, vector):
    result = []
    for row in matrix:
        sum_val = sum(row[j] * vector[j] for j in range(len(vector)))
        result.append(sum_val)
    return result

def relu_der(z):
    return [1 if x > 0 else 0 for x in z]

def update_w(weigths, gradients, learning_rate=0.01):
    updated = []
    for w_row, g_row in zip(weigths, gradients):
        updated_row = [w - learning_rate * g for w, g in zip(w_row, g_row)]
        updated.append(updated_row)
    return updated

def update_b(biases, gradients, learning_rate=0.01):
    return [b - learning_rate * g for b, g in zip(biases, gradients)]

def back_propagation(input_layer, label):
    a, z = exec(input_layer)

    result = a[-1]
    error = MSE(result, label)
    print(error)

    # Layer -1
    delta_4 = vec_mul(vec_factor(vec_sub(result, ground_truth_vec(label)), 2), relu_der(z[-1]))

    grad_4 = outer_p(delta_4, T(a[-2]))
    grad_bias_4 = delta_4

    weigths = nn.get_weights(2)
    nn.set_weights(2, update_w(weigths, grad_4))

    biases = nn.get_biases(2)
    nn.set_biases(2, update_b(biases, grad_bias_4))


    # Layer -2
    delta_3 = vec_mul(mm(T(nn.get_weights(2)), delta_4), relu_der(z[-2]))

    grad_3 = outer_p(delta_3, T(a[-3]))
    grad_bias_3 = delta_3

    weigths = nn.get_weights(1)
    nn.set_weights(1, update_w(weigths, grad_3))

    biases = nn.get_biases(1)
    nn.set_biases(1, update_b(biases, grad_bias_3))

    # Layer -3
    delta_2 = vec_mul(mm(T(nn.get_weights(2)), delta_3), relu_der(z[-3]))

    grad_2 = outer_p(delta_2, T(a[-4]))
    grad_bias_2 = delta_2

    weigths = nn.get_weights(0)
    nn.set_weights(0, update_w(weigths, grad_2))

    biases = nn.get_biases(0)
    nn.set_biases(0, update_b(biases, grad_bias_2))


