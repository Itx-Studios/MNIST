
# This file runs a single feed forward prediction

import math, random
from network import nn

def make_matrix(m: int, n: int):
    return [[random.random() for _ in range(m)] for _ in range(n)]

def make_vector(n: int):
    return [random.random() for _ in range(n)]

def mm(matrix, vector):
    result = []
    for row in matrix:
        sum_val = sum(row[j] * vector[j] for j in range(len(vector)))
        result.append(sum_val)
    return result

def ma(v1, v2):
    return [a + b for a, b in zip(v1, v2)]

def activation_fn(v):
    return [max(0, x) for x in v]

def calc_layer(input, layer):
    weights = nn.get_weights(layer)
    biases = nn.get_biases(layer)

    z = ma(mm(weights, input), biases)
    return activation_fn(z), z

def soft_max(vector):
    max_val = max(vector)
    xps = [math.exp(x - max_val) for x in vector]
    sum_exps = sum(xps)
    return [val / sum_exps for val in xps]

def exec(input):
    x2, z2 = calc_layer(input, 0)
    x3, z3 = calc_layer(x2, 1)
    x4, z4 = calc_layer(x3, 2)

    return [x2, x3, x4], [z2, z3, z4]

def feed_forward(input_layer):
    logits = exec(input_layer)[1][-1]
    out_y = soft_max(logits)
    return out_y.index(max(out_y))
