import numpy as np
import pandas as pd
import sklearn as sk
import scipy.optimize as scipy
import matplotlib as plot


def hypothesis(x, theta):
    return sigmoid(np.dot(x, theta))


def cost(x, y, theta):

    # calculate hypothesis (y - predicted probabilities)
    h = hypothesis(x, theta)
    # m = number of training examples
    m = x.shape[0]

    return -(1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))


def gradient(x, y, theta):

    for j in range(x.shape[1]):
        gradients[j] = (1 / m) * np.sum((hypothesis(x, theta) - y) * x[:,j])
    return gradients


def gradient_decent(x, y, theta):
    weights = scipy.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(x, y))
    return weights

def confusion_matrix(y_predict, y_actual, theta):

    tp, tn, fp, fn = 0, 0, 0, 0

    for i in range(y.shape[0]):
        if y_predict[i] == 1 and y_actual == 1:
            tp += 1
        if y_predict[i] == 0 and y_actual == 0:
            tn += 1
        if y_predict[i] == 1 and y_actual == 0:
            fp += 1
        if y_predict[i] == 0 and y_actual == 1:
            fn += 1

    return tp, tn, fp, fn


def get_data(file):
    data = pd.read_data(file)



def main():
    # stuff here



if __name__ == "__main__":
    main()