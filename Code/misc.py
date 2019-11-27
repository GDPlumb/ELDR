
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
