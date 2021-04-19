import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def load_file_data(filename):
    with open(filename, 'r') as fin:
        lines = fin.readlines()
        dim = len(lines[0].strip().split(";"))
        num_samples = len(lines)
        data = np.zeros((num_samples, dim))
        for i in range(num_samples):
            data[i, :] = np.array([float(x.replace(",",".")) for x in lines[i].strip().split(";")])
        return data


data = load_file_data("210406_cs_pm_co2/xdata_gifa_storey_span_load.csv")
print('data[0]', data[0], type(data[0]), len(data[0]))

def scale_features(X):
    x_bar = np.mean(X, axis=0)
    x_std = np.std(X, axis=0)
    return (X - x_bar) / x_std


def load_data(xpath, ypath, scale=True):
    if scale:
        return scale_features(load_file_data(xpath)), scale_features(load_file_data(ypath))
    return load_file_data(xpath), load_file_data(ypath)


def build_dictionary(X_values, y_values, x_labels, y_labels):  # TODO : def build_data_by_headers():
    dico = dict()
    for i in range(X_values.shape[1]):
        dico[x_labels[i]] = X_values[:, i]
    for i in range(y_values.shape[1]):
        dico[y_labels[i]] = y_values[:, i]
    return dico


def visualize_data_table(X, y, x_labels, y_labels, disp=False):
    import pandas as pd
    from IPython.display import display
    all_labels = x_labels + y_labels
    CO2_dict = build_dictionary(X, y, x_labels, y_labels)
    df = pd.DataFrame(CO2_dict, columns=all_labels)
    if disp:
        display(df)
    return df


def plot_graph(dataframe, x_label, y_label, folder=None, plot=False):
    sns.scatterplot(data=dataframe, x=x_label, y=y_label, hue=y_label)
    if folder :
        plt.savefig(folder + '/'+ x_label + '-' + y_label +'.png')
    if plot:
        plt.show()
