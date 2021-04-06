import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
import seaborn as sns
import folium

### 1. load & visualize full data

def load_dat(filename):
    with open(filename, 'r') as fin:
        lines = fin.readlines()
        dim = len(lines[0].strip().split(";"))
        num_samples = len(lines)
        data = np.zeros((num_samples, dim))
        for i in range(num_samples):
            data[i, :] = np.array([float(x.replace(",",".")) for x in lines[i].strip().split(";")])
        return data

def build_dictionary(X_values, y_values, x_labels, y_labels):
    dict = {}
    for i in range(len(X_values[0])):
        dict[x_labels[i]] = X_values[:, i]
    for i in range(len(y_values[0])):
        dict[y_labels[i]] = y_values[:, i]
    return dict

def plot_graph(dataframe, x_label, y_label, folder):
    sns.scatterplot(data=dataframe, x=x_label, y=y_label, hue=y_label)
    plt.savefig(folder + '/'+ x_label + '-' + y_label +'.png')
    plt.show()

def draw_table(dataframe, folder, table_name):
    fig, ax = plt.subplots()
    # hide axes
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    ax.table(cellText=dataframe.values, colLabels=dataframe.columns, loc='center')
    fig.tight_layout()
    plt.savefig(folder + '/'+ table_name +'.png')
    plt.show()

X = load_dat("210406_cs_pm_co2/xdata_gifa_storey_span_load.csv")
y = load_dat("210406_cs_pm_co2/ydata_totCO2.csv")
x_labels = ['GIFA', 'STOREY', 'SPAN', 'LOAD']
y_labels = ['CO2eq']
all_labels = x_labels + y_labels
CO2_dict = build_dictionary(X, y, x_labels, y_labels)
df = pd.DataFrame(CO2_dict, columns=all_labels)
display(df)
#plot_graph(df, "GIFA", "CO2eq", '210406_cs_pm_co2')
#draw_table(df, '210406_cs_pm_co2', 'CO2_values')
#plot_graph(df, "STOREY", "CO2eq", '210406_cs_pm_co2')
#plot_graph(df, "SPAN", "CO2eq", '210406_cs_pm_co2')
#plot_graph(df, "LOAD", "CO2eq", '210406_cs_pm_co2')

### 2. format & visualize training data


def create_polynomial_function(X):
    # create virtual features
    #  raise all variables to second and third degree
    X_virtual = []
    for i in range(len(X[0])):
        X_virtual_i = [np.power(X[:,i], 2).reshape([-1,1]),
                       np.power(X[:,i], 3).reshape([-1,1])]
        X_virtual += X_virtual_i
    X_virtual = np.hstack(X_virtual)
    X = np.hstack((X, X_virtual))
    interc = np.ones((X.shape[0], 1))
    X = np.hstack((interc, X))
    print(X.shape)
    return X

create_polynomial_function(X)

def split_dataset(X, train_ratio):
# split training and testing dataset
    cutoff = int(X.shape[0] * train_ratio)
    X_tr = X[:cutoff, :]
    y_tr = y[:cutoff]
    X_te = X[cutoff:,:]
    y_te = y[cutoff:]
    print('Train/Test: %d/%d' %(X_tr.shape[0], X_te.shape[0]))
    return X_tr, y_tr, X_te, y_te

X_tr, y_tr, X_te, y_te = split_dataset(X, 0.8)

### 3. regularized linear regression

def regularized_linear_regression_parameters(X_set, y_set, reg_param):
    # Calculate the regularized pseudo_inverse of A
    id_size = np.shape(X_set)[1]
    pinv = np.matmul(np.linalg.inv(np.add(np.matmul(X_set.T, X_set), reg_param*np.identity(id_size))), X_set.T)
    # fit the regularized polynomial to find optimal theta matrix
    return np.matmul(pinv, y_set)

theta_opt = regularized_linear_regression_parameters(X_tr, y_tr, 0.5)

def predict_footprint(X_set, theta):
    # make prediction on the testing set
    return np.matmul(X_set, theta)

pred = predict_footprint(X_te, theta_opt)

### 4. evaluation

def MSE(prediction,reference):
    # Calculate the mean square error between the prediction and reference vectors
    mse = 0.5 * np.mean(np.square(prediction - reference))
    return mse

def MAE(prediction, reference):
    # Calculate the mean absolute error between the prediction and reference vectors
    mae = np.mean(np.abs(prediction - reference))
    return mae

mse = MSE(pred, y_te)
mae = MAE(pred, y_te)
print(mse)
print(mae)

### 5. Display results
def compare_results(y_te, y_pred, folder, figure_name):
    # plot the ground truth and the predicted
    x_axis = np.linspace(1,len(y_te),len(y_te))
    plt.plot(x_axis,y_te,'b',x_axis,y_pred,'r')
    plt.legend(('Ground truth','Predicted'))
    plt.savefig(folder + '/' + figure_name + '.png')
    plt.show()


compare_results(y_te, pred, '210406_cs_pm_co2', 'truth_vs_pred')

"""
next : 
-compute cost
-predict for 1 user - value : "set one gifa", "set one span"
-other computation ways (gradient descent,...)
- present results/parameters associated
-compare results/display
-create bg data
-analyse graphs

"""


