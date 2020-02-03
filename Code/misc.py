
from matplotlib.path import Path
import numpy as np

def poly2labels(data_rep, vertices):

    m = data_rep.shape[0]
    num_clusters = len(vertices)
    
    labels = -1.0 * np.ones((m))
    
    for i in range(num_clusters):
        path = Path(vertices[i])
        for j in range(m):
            if path.contains_points(data_rep[j, :].reshape((1,2))):
                labels[j] = i
                
    return labels    
    
def truncate(values, k):
    values = np.squeeze(values)
    idx = (-np.abs(values)).argsort()[:k]
    values_aprox = np.zeros(values.shape)
    values_aprox[idx] = values[idx]
    return values_aprox
    
def load(deltas, k, initial, target):

    if initial == 0:
        d = deltas[target - 1]
    elif target == 0:
        d = -1.0 * deltas[initial - 1]
    else:
        d = -1.0 * deltas[initial - 1] + deltas[target - 1]
        
    d = truncate(d, k)

    return d

# e_more should be a sparser vector than e_less
# counts the percentage of e_more's explanation that is in features chosen by e_less
def stability(e_more, e_less):
    difference = 0
    for i in range(e_more.shape[0]):
        if e_less[i] != 0:
            difference += np.abs(e_more[i])
    return difference / np.sum(np.abs(e_more))

