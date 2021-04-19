"""
1. DISPLAY PARAMETERS
"""
#to update
date = "'210416"
test_count = str(1)
save_visu = True
display_plots = False

#Default
reference = date +'_results_' + test_count
input_path = "DATA/210413_PM_CO2_data"
output_path = date + '_results/'
STR_FEATURES = ['Sector', 'Type', 'Basement', 'Foundations', 'Ground Floor', 'Superstructure', 'Cladding', 'BREEAM Rating']
INT_FEATURES = ['GIFA (m2)', 'Storeys', 'Typical Span (m)', 'Typ Qk (kN_per_m2)']
FEATURES_NAMES = STR_FEATURES + INT_FEATURES
OUTPUT_NAMES = ['Calculated Total tCO2e', 'Calculated tCO2e_per_m2']

"""
2. MODEL PARAMETERS
"""
study = 1  # 0 ='Calculated Total tCO2e'; 1 ='Calculated tCO2e_per_m2'
train_ratio = 0.8
reg = 1
f_scaling = True
power_dict_int_features = dict()
for f in INT_FEATURES:
    power_dict_int_features[f] = [1, 2, 3]

powerdicto = { #TODO : will be used later
    'GIFA (m2)': [1, 2, 3],
    'Storeys': [1, 2, 3],
    'Typical Span (m)': [1, 2, 3],
    'Typ Qk (kN_per_m2)': [1, 2, 3],
    'Sector': [1],
    'Type': [1],
    'Basement': [1],
    'Foundations': [1],
    'Ground Floor': [1],
    'Superstructure': [1],
    'Cladding': [1],
    'BREEAM Rating': [1]
}

"""
3. IMPORT SCRIPTS
"""

from data_handler import *
X_data, y = split_X_Y_values(FEATURES_NAMES, OUTPUT_NAMES, input_path, 5, delimiter=';')
id_dict = index_dict_from_csv(input_path, 5)
data_dict = build_dictionary(X_data, y, FEATURES_NAMES, OUTPUT_NAMES)
nb_dict = string_dict_to_number_dict(id_dict, data_dict, STR_FEATURES, INT_FEATURES + OUTPUT_NAMES)
x_qlt_df, x_qtt_df, y_df = extract_X_y_df_from_dict(nb_dict, STR_FEATURES, INT_FEATURES, OUTPUT_NAMES, scale_int=f_scaling)

folder_path = output_path if save_visu else None
if display_plots or save_visu :
    full_df = view_dataframe_from_dict(nb_dict, disp=False)
    for x in FEATURES_NAMES:
        plot_graph_adv(full_df, x, OUTPUT_NAMES[study],label_dict=id_dict, qual_features=STR_FEATURES,
                       new_folder_path=folder_path, plot=display_plots, reference=reference)

from data_adapter import *
#X = create_polynomial_features(X_df, FEATURES_NAMES, powerdicto) #TODO : integrate qlt features to model
X = create_polynomial_features(x_qtt_df, INT_FEATURES, power_dict_int_features)
training, test = split_dataset(X, y_df, train_ratio)

from linear_regression import *
X_tr, y_tr = training
y_tr_study = y_tr[:, study] #TODO : check I can only predict one y type at a time?
theta_opt = regularized_linear_regression_parameters(X_tr, y_tr_study, reg)

from model_evaluator import *
X_te, y_te = test
y_te_study = y_te[:, study]
prediction = predict_footprint(X_te, theta_opt)

"""
4. RUN SCRIPTS
"""


def from_dict_to_list_of_features_powers(dico):
    out = ['ones']
    for feature in INT_FEATURES:
        for power in dico[feature]:
            out.append(feature + " ^" + str(power))
    return out



def runandprint():
    print("object of study : ", OUTPUT_NAMES[study])
    print("regularization param : ", reg)
    print("theta_opt : ")
    labels = from_dict_to_list_of_features_powers(power_dict_int_features)
    for index in range(len(theta_opt)):
        print(labels[index], theta_opt[index])
    print()
    print("Mean square error: ", MSE(prediction, y_te_study))
    print("Mean absolute error: ", MAE(prediction, y_te_study))
    print("Cost: ", evaluate_cost(prediction, y_te_study))
    if display_plots or save_visu:
        plot_comparative_results(y_te_study, prediction, figure_name=OUTPUT_NAMES[study],
                                 folder=folder_path, plot=display_plots, reference=reference)


runandprint()






"""
next : 
- export results as text file 
- plot average of qualitative values / excluding extreme samples > identify treds 

- convert qualitative list of int to boolean 1 vs 0 > to implement in polynomial function/ multiplicatively or additionaly
- shuffle values
-update powerdict - quality/quantity
-compute cost -- check
-predict for 1 user - value : "set one gifa", "set one span"
- other computation ways (gradient descent,...)
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
"""
Questions : 
What happens if blank spaces in excel? > replace with blank elem!!
DF = only numbers ? no strings?

"""