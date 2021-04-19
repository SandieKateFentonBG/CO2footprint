import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

STR_FEATURES = ['Sector', 'Type', 'Basement', 'Foundations', 'Ground Floor', 'Superstructure', 'Cladding', 'BREEAM Rating']
INT_FEATURES = ['GIFA (m2)', 'Storeys', 'Typical Span (m)', 'Typ Qk (kN_per_m2)']
FEATURES_NAMES = STR_FEATURES + INT_FEATURES
OUTPUT_NAMES = ['Calculated Total tCO2e', 'Calculated tCO2e_per_m2']


"""
Questions : 
What happens if blank spaces in excel? > replace with blank elem!!
DF = only numbers ? no strings?

"""

def open_csv_at_given_line(filename, first_line, delimiter):
    import csv
    reader = csv.reader(open(filename + '.csv', mode='r'), delimiter=delimiter)
    for i in range(first_line):
        reader.__next__()
    header = reader.__next__()
    return header, reader


def index_dict_from_csv(filename, first_line, delimiter=';'):
    header, reader = open_csv_at_given_line(filename, first_line, delimiter)
    CST = dict()
    for f in FEATURES_NAMES:
        CST[f] = []
    for line in reader:
        for f in FEATURES_NAMES:
            index = header.index(f)
            if line[index] not in CST[f]:
                CST[f].append(line[index])
    return CST


def split_X_Y_values(X_NAMES, Y_NAMES, filename, first_line, delimiter=';'):
    header, reader = open_csv_at_given_line(filename, first_line, delimiter)
    X_values, Y_values = [], []
    for line in reader:
        for (names, values) in [(X_NAMES, X_values), (Y_NAMES, Y_values)]:
            values.append([line[header.index(name)] for name in names])
    return X_values, Y_values
    # line[header.index(name)] = value in that column


def build_dictionary(X_values, y_values, x_labels, y_labels):  # TODO : shorten
    dico = dict()
    for i in range(len(X_values[0])):  # 12
        dico[x_labels[i]]=[]
    for i in range(len(y_values[0])):  # 12
        dico[y_labels[i]] = []
    for j in range(len(X_values)): #80
        for i in range(len(X_values[0])): #12
            dico[x_labels[i]].append(X_values[j][i])
        for k in range(len(y_values[0])):
            dico[y_labels[k]].append(y_values[j][k])
    return dico





def qualitative_str_feature_to_int(index_dictionary, feature_name, feature_value):

    return index_dictionary[feature_name].index(feature_value)

def quantitative_str_feature_to_float(string):
    """
    input : decimal number in string with "," for decimal separation
    output : decimal number in float with "." for decimal separation
    """

    try:
        return float(string.replace(',', '.'))
    except:
        print(string, ": this should be a number")
        return False

def string_dict_to_number_dict(index_dict,feature_dict, qualitative_features, quantitative_features ):
    number_dict = dict()
    for ql_feature in qualitative_features:
        number_dict[ql_feature] = []
        for ql_value in feature_dict[ql_feature]:
            number_dict[ql_feature].append(qualitative_str_feature_to_int(index_dict, ql_feature, ql_value))
    for qn_feature in quantitative_features:
        number_dict[qn_feature] = []
        for qn_value in feature_dict[qn_feature]:
            number_dict[qn_feature].append(quantitative_str_feature_to_float(qn_value))
    return number_dict

def extract_feature_df_from_dict(data_dict, feature_labels, scale=True):
    num_samples, num_features = len(data_dict[feature_labels[0]]), len(feature_labels)
    data = np.zeros((num_samples, num_features))
    for i in range(num_samples): #80
        data[i, :] = np.array([data_dict[f][i] for f in feature_labels])
        #data[i, :] = np.array([value[i]] for value in data_dict[value])
        #data[i, :] = np.array([float(x) for x in data_matrix[i].strip().split(";")])
    if scale:
        return scale_features(data)
    return data

#def split_dict_into_X_Y_df(data_dict, x_labels, y_labels, scale)
#    return extract_feature_df_from_dict(data_dict, x_labels, scale), extract_feature_df_from_dict(data_dict, y_labels, scale)

def print_dictionary(dict):
    print("keys", len(dict.keys()), dict.keys())
    for key in dict.keys():
        print("key", key, len(dict[key]), type(dict[key]), dict[key])
        print("item", type(dict[key][0]))

def scale_features(X):
    x_bar = np.mean(X, axis=0)
    x_std = np.std(X, axis=0)
    return (X - x_bar) / x_std

def dataframe_from_dict(dict, disp=False):
    import pandas as pd
    from IPython.display import display
    df = pd.DataFrame(dict, columns=dict.keys())
    if disp:
        display(df)
    return df

def plot_graph(dataframe, x_label, y_label,
               title="Features influencing CO2 footprint of Structures - Datasource : Price & Myers",
               figure_size=(12,15), folder=None, plot=False):
    fig, ax = plt.subplots(figsize=figure_size)
    ax.set_title("Features influencing CO2 footprint of Structures - Datasource : Price & Myers")
    sns.scatterplot(data=dataframe, x=x_label, y=y_label, hue=y_label)
    if folder :
        plt.savefig(folder + '/'+ x_label + '-' + y_label +'.png')
    if plot:
        plt.show()

def plot_graph_advanced(dataframe, x_label, y_label, label_dict,
                        title="Features influencing CO2 footprint of Structures - Datasource : Price & Myers",
                        figure_size=(12,15), existing_folder=None, new_folder_path=None, plot=True, descriptive_features=STR_FEATURES):
    labels = label_dict[x_label]
    fig, ax = plt.subplots(figsize=figure_size)
    ax.set_title(title)
    if x_label in descriptive_features:
        x = np.arange(len(labels))
        ax.set_ylabel(y_label)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        plt.setp(ax.get_xticklabels(), rotation=25, ha="right",
                 rotation_mode="anchor")
    sns.scatterplot(data=dataframe, x=x_label, y=y_label, hue=y_label, ax=ax)
    if existing_folder:
        plt.savefig(existing_folder + '/'+ x_label + '-' + y_label +'.png')
    if new_folder_path:
        # Create new directory
        output_dir = new_folder_path
        mkdir_p(output_dir)
        plt.savefig(new_folder_path + x_label + '-' + y_label + '.png')
    if plot:
        plt.show()


def mkdir_p(mypath):
    '''Creates a directory. equivalent to using mkdir -p on the command line'''

    from errno import EEXIST
    from os import makedirs,path

    try:
        makedirs(mypath)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else: raise


xd, yd = split_X_Y_values(FEATURES_NAMES, OUTPUT_NAMES, "DATA/210413_PM_CO2_data", 5, delimiter=';')

#STR_FEATURES = ['Sector', 'Type', 'Basement', 'Foundations', 'Ground Floor', 'Superstructure', 'Cladding', 'BREEAM Rating']
#INT_FEATURES = ['GIFA (m2)', 'Storeys', 'Typical Span (m)', 'Typ Qk (kN_per_m2)']


index_dict = index_dict_from_csv("DATA/210413_PM_CO2_data", 5)
data_dict = build_dictionary(xd, yd, FEATURES_NAMES, OUTPUT_NAMES)
qualitative_features = STR_FEATURES
quantitative_features = INT_FEATURES + OUTPUT_NAMES
nb_dict = string_dict_to_number_dict(index_dict,data_dict, qualitative_features, quantitative_features )
output_path = 'test3/'



df = extract_feature_df_from_dict(nb_dict, INT_FEATURES)
daf = dataframe_from_dict(nb_dict, disp=False)

#for x in FEATURES_NAMES:
#    plot_graph_advanced(daf, x, OUTPUT_NAMES[0], index_dict, figure_size=(12,15), folder=output_path+OUTPUT_NAMES[0], plot=True)

my_existing_folder="test"
my_new_folder_path=output_path
#my_new_folder_path=output_path+"test2"

#plot_graph_advanced(daf, 'Sector', OUTPUT_NAMES[0], index_dict, existing_folder=my_existing_folder, plot=True)
plot_graph_advanced(daf, 'Sector', OUTPUT_NAMES[0], index_dict, new_folder_path=my_new_folder_path, plot=True)

