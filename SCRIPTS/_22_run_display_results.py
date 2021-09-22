import matplotlib.pyplot as plt
import numpy as np

class results_displayer():

    def __init__(self, linear_regression):

        self.linear_regression = linear_regression
        self.proc_data = self.linear_regression.proc_data #todo: need parenthseis?
        self.data_model = self.proc_data.preprocessed_data.Model_Structural_Embodied_CO2

    def plot_comparative_results(self, figure_size=(12, 15),
                                 title="Predicted results vs Ground truth", folder=False, plot=True, reference=""):
        # plot the ground truth and the predicted
        y_te = self.proc_data.y_te
        y_pred = self.linear_regression.predict_footprint()
        fig, ax = plt.subplots(figsize=figure_size)
        ax.set_title(title + " " + reference)
        x_axis = np.linspace(1, len(y_te), len(y_te))
        plt.ylabel(self.data_model.y_features[self.data_model.tCO2e_per_m2])
        plt.xlabel("Test samples")
        plt.plot(x_axis, y_te, 'b', x_axis, y_pred, 'r')
        plt.legend(('Ground truth - value from PM database', 'Predicted - value from linear regression'))
        if folder:
            plt.savefig(self.data_model.output_path + '/' + self.data_model.y_features[self.data_model.tCO2e_per_m2] + '_truth_vs_pred' + '.png')
        if plot:
            plt.show()

    def format_parameter_matrix(self):

        features = ["bias_term"] + self.proc_data.selected_features
        powers =  self.proc_data.selected_powers
        flat_powers = [1] + [item for sublist in powers for item in sublist]
        theta_opt = self.linear_regression.optimize_parameters()
        return features, flat_powers, theta_opt

    def display_parameter_matrix(self):
        features, flat_powers, theta_opt = self.format_parameter_matrix()
        print("Parameter Matrix")
        print("Feature ^ Power : Optimized Parameter")
        for feature, power, parameter in zip(features, flat_powers, theta_opt):
            print(feature, "^", power, ":", parameter)

    def display_parameter_ascending(self):
        features, flat_powers, theta_opt = self.format_parameter_matrix()
        abs_theta_opt = abs(theta_opt)
        zipped_lists = zip(abs_theta_opt, theta_opt, features, flat_powers)
        sorted_zipped_lists = sorted(zipped_lists)
        sorted_param = [element for _, element, _, _ in sorted_zipped_lists]
        sorted_features = [element for _, _, element, _ in sorted_zipped_lists]
        sorted_powers = [element for _, _, _, element in sorted_zipped_lists]
        print("Sorted Parameter Matrix")
        print("Feature ^ Power : Optimized Parameter")
        for feature, power, parameter in zip(sorted_features, sorted_powers, sorted_param):
            print(feature, "^", power, ":", parameter)

    def dictionary_of_results(self):
        results = dict()
        results["object of study"] = [self.data_model.y_features[self.data_model.tCO2e_per_m2]]
        results["regularization term"] = [self.data_model.reg]
        results["Mean square error"] = [self.linear_regression.MSE()]
        results["Mean absolute error"] = [self.linear_regression.MAE()]
        results["Cost"] = [self.linear_regression.evaluate_cost()]
        features, flat_powers, theta_opt = self.format_parameter_matrix()
        results["features"] = features
        results["flat_powers"] = flat_powers
        results["theta_opt"] = theta_opt
        return results

    def print_results(self):
        res_dict = self.dictionary_of_results()
        for k, v in res_dict.items():
            print(k, v)