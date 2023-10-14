# importing the required module 
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sigmoid import sigmoid

def cost(x, y, theta):
    # Calculating the loss or cost
    m = x.shape[0]
    cost = -(1 / m) * np.sum(y * np.log(sigmoid(np.dot(x, theta))) + (1-y) * np.log(1 - sigmoid(np.dot(x, theta))))
    return cost

def main():
    return

if __name__ == "__main__":
    main()