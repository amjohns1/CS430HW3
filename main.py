import numpy as np
import pandas as pd
import scipy.optimize as scipy
from sklearn.model_selection import train_test_split
from sigmoid import sigmoid

def prepare_data(target_class, data):
    class_data = {}
    for class_name in data['class'].unique():
        class_data[class_name] = data[data['class'] == class_name]

    X_train_data = []
    X_test_data = []
    y_train_data = []
    y_test_data = []

    for class_name, class_df in class_data.items():
        # Change the 'class' column to 1 if it matches the target class, 0 otherwise
        class_df['class'] = class_df['class'].apply(lambda x: 1 if x == target_class else 0)
        
        # Perform an 80/20 split on each class
        X = class_df.drop("class", axis=1)
        y = class_df["class"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)
        
        X_train_data.append(X_train)
        X_test_data.append(X_test)
        y_train_data.append(y_train)
        y_test_data.append(y_test)

    # Concatenate the data of all three classes for training and testing
    X_train = pd.concat(X_train_data)
    X_test = pd.concat(X_test_data)
    y_train = pd.concat(y_train_data)
    y_test = pd.concat(y_test_data)

    return X_train, X_test, y_train, y_test

def cost(theta, x, y):
    # Calculating the loss or cost
    m = x.shape[0]

    h = sigmoid(np.dot(x, theta))
    cost = -(1 / m) * np.sum(y * np.log(h) + (1-y) * np.log(1 - h))

    return cost

def gradient(theta, x, y):
    m = x.shape[0]
    return (1 / m) * np.dot(x.T, sigmoid(np.dot(x, theta)) - y)

def gradient_descent(theta, x, y):
    weights = scipy.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(x, y))
    return weights[0]

def confusion_matrix(y_predict, y_actual):
    y_actual = np.asarray(y_actual)
    tp, tn, fp, fn = 0, 0, 0, 0

    for i in range(len(y_predict)):
        if y_predict[i] < 0.5:
            y_predict[i] = 0
        else:
            y_predict[i] = 1

    for i in range(y_actual.shape[0]):
        if y_predict[i] == 1 and y_actual[i] == 1:
            tp += 1
        if y_predict[i] == 0 and y_actual[i] == 0:
            tn += 1
        if y_predict[i] == 1 and y_actual[i] == 0:
            fp += 1
        if y_predict[i] == 0 and y_actual[i] == 1:
            fn += 1

    confusion_matrix = pd.DataFrame({
        "Predicted Positive": [tp, fp],
        "Predicted Negative": [fn, tn]
    }, index=["Actual Positive", "Actual Negative"])

    def precision():
        if tp + fp == 0:
            return 0
        return float(tp / (tp + fp))

    def accuracy():
        total = tp + tn + fp + fn
        if total == 0:
            return 0
        return float((tp + tn) / total)

    return {
        "Confusion Matrix": confusion_matrix,
        "Precision": precision(),
        "Accuracy": accuracy()
    }

def main():
    # Define the column names for the dataset
    column_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]

    # Load the dataset into a pandas DataFrame
    data = pd.read_csv("iris.data", names=column_names)

    # For Setosa
    X_train_setosa, X_test_setosa, y_train_setosa, y_test_setosa = prepare_data('Iris-setosa', data)
    X_train_setosa = np.column_stack((np.ones(X_train_setosa.shape[0]), X_train_setosa))
    X_test_setosa = np.column_stack((np.ones(X_test_setosa.shape[0]), X_test_setosa))

    theta = np.zeros((X_train_setosa.shape[1], 1))
    weights_setosa = gradient_descent(theta, X_train_setosa, y_train_setosa)
    predicted_setosa = sigmoid(np.dot(X_test_setosa, weights_setosa))

    metrics_setosa = confusion_matrix(predicted_setosa, y_test_setosa)

    # For Versicolor
    X_train_versicolor, X_test_versicolor, y_train_versicolor, y_test_versicolor = prepare_data('Iris-versicolor', data)
    X_train_versicolor = np.column_stack((np.ones(X_train_versicolor.shape[0]), X_train_versicolor))
    X_test_versicolor = np.column_stack((np.ones(X_test_versicolor.shape[0]), X_test_versicolor))

    theta = np.zeros((X_train_versicolor.shape[1], 1))
    weights_versicolor = gradient_descent(theta, X_train_versicolor, y_train_versicolor)
    predicted_versicolor = sigmoid(np.dot(X_test_versicolor, weights_versicolor))

    metrics_versicolor = confusion_matrix(predicted_versicolor, y_test_versicolor)

    # For Versicolor
    X_train_virginica, X_test_virginica, y_train_virginica, y_test_virginica = prepare_data('Iris-virginica', data)
    X_train_virginica = np.column_stack((np.ones(X_train_virginica.shape[0]), X_train_virginica))
    X_test_virginica = np.column_stack((np.ones(X_test_virginica.shape[0]), X_test_virginica))

    theta = np.zeros((X_train_virginica.shape[1], 1))
    weights_virginica = gradient_descent(theta, X_train_virginica, y_train_virginica)
    predicted_virginica = sigmoid(np.dot(X_test_virginica, weights_virginica))

    metrics_virginica = confusion_matrix(predicted_virginica, y_test_virginica)
    
    # Print the metrics for Setosa
    print("\nMetrics for Setosa:")
    for metric, value in metrics_setosa.items():
        if metric == "Confusion Matrix":
            print(f"{metric}:\n{value}")
        else:
            print(f"{metric}: {value}")

    # Print the metrics for Versicolor
    print("\nMetrics for Versicolor:")
    for metric, value in metrics_versicolor.items():
        if metric == "Confusion Matrix":
            print(f"{metric}:\n{value}")
        else:
            print(f"{metric}: {value}")

    # Print the metrics for Virginica
    print("\nMetrics for Virginica:")
    for metric, value in metrics_virginica.items():
        if metric == "Confusion Matrix":
            print(f"{metric}:\n{value}")
        else:
            print(f"{metric}: {value}")

    # Plot matrices and perform further analysis here
    return

if __name__ == "__main__":
    main()
