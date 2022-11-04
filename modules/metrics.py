import numpy as np

def rmse_loss(a,b):
    return np.sqrt(((a-b)**2).mean())
def mse_loss(a,b):
    return ((a-b)**2).mean()