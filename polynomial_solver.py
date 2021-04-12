train_ratio = 0.8
reg = 1
powerdict = {
    'GIFA': [1, 2],
    'LOAD': [1, 2],
    'SPAN': [1, 2],
    'STOREY': [1, 2]
}

study = ""
save_visu = False

x_labels = ['GIFA', 'STOREY', 'SPAN', 'LOAD']  # TODO : [label for label in powerdict.keys()]
y_labels = ['CO2eq' + study]
input_path = "210406_cs_pm_co2/"
output_path = '210413_cs_pm_co2/'


from data_accessor import *
X_data, y = load_data(input_path + "xdata_gifa_storey_span_load.csv", "%sydata_totCO2%s.csv" % (input_path, study))
if save_visu:
    df = visualize_data_table(X_data, y, x_labels, y_labels)
    for x in x_labels:
        plot_graph(df, x, y_labels[0], output_path)

from data_adapter import *
X = create_polynomial_features(X_data, x_labels, powerdict)
training, test = split_dataset(X, y, train_ratio)

from linear_regression import *
X_tr, y_tr = training
theta_opt = regularized_linear_regression_parameters(X_tr, y_tr, reg)

from model_evaluator import *
X_te, y_te = test
prediction = predict_footprint(X_te, theta_opt)


def from_dict_to_list_of_features_powers(dico):
    out = ['ones']
    for feature in x_labels:
        for power in dico[feature]:
            out.append(feature + " ^" + str(power))
    return out


def runandprint():
    print("object of study : ", y_labels[0])
    print("regularization param : ", reg)
    print("theta_opt : ")
    labels = from_dict_to_list_of_features_powers(powerdict)
    for index in range(len(theta_opt)):
        print(labels[index], theta_opt[index])
    print()
    print("Mean square error: ", MSE(prediction, y_te))
    print("Mean absolute error: ", MAE(prediction, y_te))
    print("Cost: ", evaluate_cost(prediction, y_te))
    plot_comparative_results(y_te, prediction, output_path if save_visu else None, False, y_labels[0])


runandprint()

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
print the obtained beta - will say relations to CO2

"""
