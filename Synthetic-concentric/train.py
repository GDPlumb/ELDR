
import matplotlib as mpl
mpl.use('Agg')
import pickle

import sys
sys.path.insert(0, "../Code/")
from train_class import train_class

if __name__ == "__main__":
    x, y = pickle.load(open("data.pkl", "rb"))
    train_class(x, y, batch_size = 32, recon_weight = 1.0)
