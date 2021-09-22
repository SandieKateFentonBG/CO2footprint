import numpy as np


def power_up_feature(featureArray_column, powerList):
    list_of_virtual_features = []
    for power in powerList:
        list_of_virtual_features.append(np.power(featureArray_column, power).reshape([-1, 1]))
    return np.hstack(list_of_virtual_features)


def create_polynomial_features(X, xlabels, powerdict):
    polynomial_features = np.ones((X.shape[0], 1))
    for feature_index in range(X.shape[1]):
        new_columns = power_up_feature(X[:, feature_index], powerdict[xlabels[feature_index]])
        polynomial_features = np.hstack((polynomial_features, new_columns))
    return polynomial_features


def split_dataset(X, y, train_ratio, shuffle=False):
    if shuffle:
        pass  # TODO
    cutoff = int(X.shape[0] * train_ratio)
    return (X[:cutoff, :], y[:cutoff]), (X[cutoff:, :], y[cutoff:])
