import numpy as np
import pickle

def generate(n = 1000):

    x = np.random.normal(size = (n, 5))
    
    l2norm = np.reshape(np.sqrt((x * x).sum(axis = 1)), (n, 1))
    
    x /= l2norm
    
    x[np.int(n / 2):, ] *= 2.0
    
    y = np.zeros((n, 1))
    y[np.int(n / 2):, ] = 1.0
    
    with open("data.pkl", "wb") as f:
        pickle.dump((x, y), f)

if __name__ == "__main__":
    generate()
