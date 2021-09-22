import numpy as np


def regularized_linear_regression_parameters(X_set, y_set, reg_param):
    # Calculate the regularized pseudo_inverse of A
    id_size = np.shape(X_set)[1]
    pinv = np.matmul(np.linalg.inv(np.add(np.matmul(X_set.T, X_set), reg_param*np.identity(id_size))), X_set.T)
    # fit the regularized polynomial to find optimal theta matrix
    return np.matmul(pinv, y_set)


def predict_footprint(X_set, theta):
    return np.matmul(X_set, theta)
