import matplotlib.pyplot as plt
import numpy as np


def MSE(prediction, reference):
    # Calculate the mean square error between the prediction and reference vectors
    return 0.5 * np.mean(np.square(prediction - reference))


def MAE(prediction, reference):
    # Calculate the mean absolute error between the prediction and reference vectors
    return np.mean(np.abs(prediction - reference))


def evaluate_cost(prediction, reference, method='mse'):
    if method == 'mse':
        return MSE(prediction, reference) / len(prediction)
    elif method == 'mae':
        return MAE(prediction, reference) / len(prediction)


def plot_comparative_results(y_te, y_pred, folder=None, plot=False, figure_name=None):
    # plot the ground truth and the predicted
    x_axis = np.linspace(1,len(y_te),len(y_te))
    plt.plot(x_axis,y_te,'b',x_axis,y_pred,'r')
    plt.legend(('Ground truth'+ figure_name,'Predicted'+ figure_name))
    if plot:
        plt.show()
    if folder:
        plt.savefig(folder + '/' + figure_name + 'truth_vs_pred' + '.png')
