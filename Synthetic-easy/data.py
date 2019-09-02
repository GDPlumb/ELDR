import numpy as np
import pickle

def generate(n = 100, d = 5):

    x = np.random.uniform(size = (n, d))
    
    x[np.int(n / 2):, ] += 2.0
    
    with open("data.pkl", "wb") as f:
        pickle.dump(x, f)

if __name__ == "__main__":
    generate()
