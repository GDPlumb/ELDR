
import matplotlib as mpl
mpl.use('Agg')
import pickle

import sys
sys.path.insert(0, "../Code/")
from train_ae import train_ae

if __name__ == "__main__":
    x = pickle.load(open("data.pkl", "rb"))
    train_ae(x)
