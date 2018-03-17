import numpy as np

def sgd(w, dw, config=None):
    """
    Performs vanilla stochastic gradient descent.

    config format:
    - learning_rate: Scalar learning rate.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)
    #print ("w",w)
    #print ("dw",dw)
    #print (config['learning_rate'])
    w -= config['learning_rate'] * dw
    return w, config