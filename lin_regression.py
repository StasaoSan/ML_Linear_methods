import numpy as np

def transform_labels(y):
    return np.where(y == 1, 1, -1).reshape(-1, 1)

def add_intercept(X):
    intercept = np.ones((X.shape[0], 1))
    return np.hstack((intercept, X))

def ridge_regression(X, y, lambda_reg):
    n_features = X.shape[1]
    I = np.eye(n_features)
    I[0, 0] = 0
    w = np.linalg.inv(X.T @ X + lambda_reg * I) @ X.T @ y

    return w

def predict(X, w):
    return np.sign(X @ w)

def evaluate_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)