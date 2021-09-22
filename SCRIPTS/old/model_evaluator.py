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


def plot_comparative_results(y_te, y_pred,figure_size=(12,15),
                             title="Predicted results vs Ground truth", folder=None, plot=False,
                             figure_name=None, reference=""):
    # plot the ground truth and the predicted
    fig, ax = plt.subplots(figsize=figure_size)
    ax.set_title(title+" "+reference)
    x_axis = np.linspace(1,len(y_te),len(y_te))
    plt.ylabel(figure_name)
    plt.xlabel("Test samples")
    plt.plot(x_axis,y_te,'b',x_axis,y_pred,'r')
    plt.legend(('Ground truth - value from PM database','Predicted - value from linear regression'))
    if folder:
        plt.savefig(folder + '/' + figure_name + '_truth_vs_pred' + '.png')
    if plot:
        plt.show()



