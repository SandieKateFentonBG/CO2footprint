import matplotlib.pyplot as plt
import numpy as np

class linear_regression():

    def __init__(self, linear_regression):

        self.linear_regression = linear_regression
        self.proc_data = self.linear_regression.proc_data

    def plot_comparative_results(self, figure_size=(12, 15),
                                 title="Predicted results vs Ground truth", folder=None, plot=False,
                                 figure_name=None, reference=""):
        # plot the ground truth and the predicted
        y_te = self.proc_data.y_te
        y_pred = self.linear_regression.predict_footprint()
        fig, ax = plt.subplots(figsize=figure_size)
        ax.set_title(title + " " + reference)
        x_axis = np.linspace(1, len(y_te), len(y_te))
        plt.ylabel(figure_name)
        plt.xlabel("Test samples")
        plt.plot(x_axis, y_te, 'b', x_axis, y_pred, 'r')
        plt.legend(('Ground truth - value from PM database', 'Predicted - value from linear regression'))
        if folder:
            plt.savefig(folder + '/' + figure_name + '_truth_vs_pred' + '.png')
        if plot:
            plt.show()