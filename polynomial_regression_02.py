import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
import seaborn as sns
import folium


"""
------------------------------------------------------------------------------------------------------------------------
1. DEFINITIONS
------------------------------------------------------------------------------------------------------------------------
"""

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

def plot_graph(dataframe, x_label, y_label, folder=None):
    sns.scatterplot(data=dataframe, x=x_label, y=y_label, hue=y_label)
    if folder :
        plt.savefig(folder + '/'+ x_label + '-' + y_label +'.png')
    plt.show()

def draw_table(dataframe, folder=None, table_name=None):
    fig, ax = plt.subplots()
    # hide axes
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    ax.table(cellText=dataframe.values, colLabels=dataframe.columns, loc='center')
    fig.tight_layout()
    if table_name :
        plt.savefig(folder + '/'+ table_name +'.png')
    plt.show()

### 2. format & visualize training data

def scale_features(X):
    # scale features by removing mean and dividing by the standard deviation
    x_bar = np.mean(X, axis=0)
    x_std = np.std(X, axis=0)
    return (X - x_bar) / x_std

def create_polynomial_features(X):
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
<<<<<<< HEAD
    print(X,type(X), X.shape)
=======
    print(X.shape)
>>>>>>> origin/master
    return X

def split_dataset(X, train_ratio):
# split training and testing dataset
    cutoff = int(X.shape[0] * train_ratio)
    X_tr = X[:cutoff, :]
    y_tr = y[:cutoff]
    X_te = X[cutoff:,:]
    y_te = y[cutoff:]
    print('Train/Test: %d/%d' %(X_tr.shape[0], X_te.shape[0]))
    return X_tr, y_tr, X_te, y_te

### 3. regularized linear regression

def regularized_linear_regression_parameters(X_set, y_set, reg_param):
    # Calculate the regularized pseudo_inverse of A
    id_size = np.shape(X_set)[1]
    pinv = np.matmul(np.linalg.inv(np.add(np.matmul(X_set.T, X_set), reg_param*np.identity(id_size))), X_set.T)
    # fit the regularized polynomial to find optimal theta matrix
    return np.matmul(pinv, y_set)

def predict_footprint(X_set, theta):
    # make prediction on the testing set
    return np.matmul(X_set, theta)

### 4. evaluation

def MSE(prediction,reference):
    # Calculate the mean square error between the prediction and reference vectors
    mse = 0.5 * np.mean(np.square(prediction - reference))
    return mse

def cost_from_mse(prediction,reference): #check!!!!!!
    mse = MSE(prediction,reference)
    n = len(prediction)
    return 0.5*mse / n

def MAE(prediction, reference):
    # Calculate the mean absolute error between the prediction and reference vectors
    mae = np.mean(np.abs(prediction - reference))
    return mae

### 5. Display results

def compare_results(y_te, y_pred, folder=None, figure_name=None):
    # plot the ground truth and the predicted
    x_axis = np.linspace(1,len(y_te),len(y_te))
    plt.plot(x_axis,y_te,'b',x_axis,y_pred,'r')
    plt.legend(('Ground truth','Predicted'))
    if figure_name:
        plt.savefig(folder + '/' + figure_name + '.png')
    plt.show()


"""
------------------------------------------------------------------------------------------------------------------------
2. APPLICATION
------------------------------------------------------------------------------------------------------------------------
"""

### 1. load & visualize full data

X = load_dat("210406_cs_pm_co2/xdata_gifa_storey_span_load.csv")
y = load_dat("210406_cs_pm_co2/ydata_totCO2.csv")
x_labels = ['GIFA', 'STOREY', 'SPAN', 'LOAD']
y_labels = ['CO2eq']
all_labels = x_labels + y_labels
CO2_dict = build_dictionary(X, y, x_labels, y_labels)
<<<<<<< HEAD

=======
>>>>>>> origin/master
df = pd.DataFrame(CO2_dict, columns=all_labels)
#display(df)
#draw_table(df)
#plot_graph(df, "GIFA", "CO2eq")
#plot_graph(df, "STOREY", "CO2eq", '210406_cs_pm_co2')
#plot_graph(df, "SPAN", "CO2eq", '210406_cs_pm_co2')
#plot_graph(df, "LOAD", "CO2eq", '210406_cs_pm_co2')

### 2. format & visualize training data

print ("before", X[0])
X = scale_features(X)
print ("after", X[0])
y = scale_features(y)
create_polynomial_features(X)
X_tr, y_tr, X_te, y_te = split_dataset(X, 0.8)

### 3. regularized linear regression

reg = 1
theta_opt = regularized_linear_regression_parameters(X_tr, y_tr, reg)
pred = predict_footprint(X_te, theta_opt)
print("regularization param : ", reg)

### 4. evaluation

mse = MSE(pred, y_te)
mae = MAE(pred, y_te)
cost = cost_from_mse(pred,y_te)
print("Mean square error: ", mse)
print("Mean absolute error: ", mae)
print("Cost: ", cost)

### 5. Display results

compare_results(y_te, pred)
#compare_results(y_te, pred, '210406_cs_pm_co2', 'truth_vs_pred')

"""
next : 
-compute cost -- check
-predict for 1 user - value : "set one gifa", "set one span"
-other computation ways (gradient descent,...)
- present results/parameters associated
-compare results/display
-create bg data
-analyse graphs
- try with different parameters
 
 
 evaluation :
 how do we know if these values are good/bad? 
 is this overfiting ?
 order : split train/test - add intercept - scale - create function... ?
<<<<<<< HEAD
print the obtained beta - will say relations to CO2

"""


def test_print(text, obj, showobj=False, isarray=True):
    print(text)
    print(type(obj), len(obj))
    if showobj:
        print(obj)
    if isarray:
        print(obj.shape)
    print()

test = load_dat("210406_cs_pm_co2/xdata_gifa_storey_span_load.csv")
m = test[:,0]
b = np.power(m, 2).reshape(-1,1)

things = (b, b.copy(), b.copy())
good = np.hstack(things)
test_print("good", good)

things = (b, np.hstack([x for x in range(80)]).reshape(-1,1), b.copy())
test = np.hstack(things)
test_print("test", test)
=======


"""


>>>>>>> origin/master
