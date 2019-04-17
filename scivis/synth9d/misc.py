import numpy as np

def two2nine(input):
    out = []
    for i in range(input.shape[0]):
        x = input[i, 0]
        y = input[i, 1]
        out.append(np.array([x+y, x-y, x*y, x*x, y*y, x*x*y, x*y*y, x*x*x, y*y*y]))
    return np.array(out)
