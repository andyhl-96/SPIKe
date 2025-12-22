import spike
import numpy as np
import matplotlib.pyplot as plt

def circle(t):
    x = np.array([0, 0, 0, 0, np.cos(t), np.sin(t), 0])
    return x

X = spike.compute_points(circle, 100)
print(X)