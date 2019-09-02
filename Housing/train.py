
import matplotlib as mpl
mpl.use('Agg')
import numpy as np

import sys
sys.path.insert(0, "../Code/")
from train_reg import train_reg

from data import load_data

if __name__ == "__main__":
    x, y = load_data()
    train_reg(x, y, batch_size = 64, min_epochs = 500, recon_weight = 5.0)
