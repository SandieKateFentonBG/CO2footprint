import numpy as np
import matplotlib.pyplot as plt

class linear_regression():

    def __init__(self, processed_data):

        self.proc_data = processed_data

    def regularized_linear_regression_parameters(self):
        # Calculate the regularized pseudo_inverse of A
        X_set = self.proc_data.x_tr
        y_set = self.proc_data.y_tr
        reg_param = self.proc_data.preprocessed_data.Model_Structural_Embodied_CO2.reg
        id_size = np.shape(X_set)[1]
        pinv = np.matmul(np.linalg.inv(np.add(np.matmul(X_set.T, X_set), reg_param*np.identity(id_size))), X_set.T)
        # fit the regularized polynomial to find optimal theta matrix
        return np.matmul(pinv, y_set)

    def predict_footprint(self):
        X_set = self.proc_data.x_te
        theta = self.regularized_linear_regression_parameters()
        return np.matmul(X_set, theta)

    def MSE(self):
        # Calculate the mean square error between the prediction and reference vectors
        prediction = self.predict_footprint()
        reference = self.proc_data.y_te
        return 0.5 * np.mean(np.square(prediction - reference))

    def MAE(self):
        # Calculate the mean absolute error between the prediction and reference vectors
        prediction = self.predict_footprint()
        reference = self.proc_data.y_te
        return np.mean(np.abs(prediction - reference))

    def evaluate_cost(self, method='mse'):
        prediction = self.predict_footprint()
        reference = self.proc_data.y_te
        if method == 'mse':
            return self.MSE() / len(prediction)
        elif method == 'mae':
            return self.MAE() / len(prediction)



