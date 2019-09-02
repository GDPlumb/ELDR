
import numpy as np
from sklearn.datasets import load_boston

def load_data():
    data = load_boston()
    
    x = data.data
    x = x - np.min(x, axis = 0)
    x = x / np.max(x, axis = 0)
    
    y = data.target
    y = y - np.mean(y)
    y = y / np.sqrt(np.var(y))
    y = np.expand_dims(y, 1)

    return x, y
