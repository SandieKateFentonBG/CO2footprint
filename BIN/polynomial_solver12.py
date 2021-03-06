train_ratio = 0.8
reg = 1
powerdicto = {
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

study = 1  # 0 ='Calculated Total tCO2e'; 1 ='Calculated tCO2e_per_m2'
save_visu = False
display_plots = True

STR_FEATURES = ['Sector', 'Type', 'Basement', 'Foundations', 'Ground Floor', 'Superstructure', 'Cladding', 'BREEAM Rating']
INT_FEATURES = ['GIFA (m2)', 'Storeys', 'Typical Span (m)', 'Typ Qk (kN_per_m2)']
FEATURES_NAMES = STR_FEATURES + INT_FEATURES
OUTPUT_NAMES = ['Calculated Total tCO2e', 'Calculated tCO2e_per_m2']

powerdict = dict()
for f in INT_FEATURES:
    powerdict[f] = [1, 2, 3]

date = "'210415"
input_path = "DATA/210413_PM_CO2_data"
output_path = date + '_results/'


from data_handler import *
X_data, y = split_X_Y_values(FEATURES_NAMES, OUTPUT_NAMES, input_path, 5, delimiter=';')
id_dict = index_dict_from_csv(input_path, 5)
data_dict = build_dictionary(X_data, y, FEATURES_NAMES, OUTPUT_NAMES)
nb_dict = string_dict_to_number_dict(id_dict, data_dict, STR_FEATURES, INT_FEATURES + OUTPUT_NAMES)
x_qlt_df, x_qtt_df, y_df = extract_X_y_df_from_dict(nb_dict, STR_FEATURES, INT_FEATURES, OUTPUT_NAMES, scale_int=True)

folder_path = output_path if save_visu, None
plot_display = True if display_plots, False
if display :
    full_df = view_dataframe_from_dict(nb_dict, disp=False)
    for x in FEATURES_NAMES:
        plot_graph_adv(full_df, x, OUTPUT_NAMES[study],new_folder_path=folder_path, id_dict, qual_features=STR_FEATURES)

from data_adapter import *
#X = create_polynomial_features(X_df, FEATURES_NAMES, powerdicto)
X = create_polynomial_features(x_qtt_df, INT_FEATURES, powerdict)
training, test = split_dataset(X, y_df, train_ratio)

from linear_regression import *
X_tr, y_tr = training
y_tr_study = y_tr[:, study] #i can only predict one a t a time?
theta_opt = regularized_linear_regression_parameters(X_tr, y_tr_study, reg)

from model_evaluator import *
X_te, y_te = test
y_te_study = y_te[:, study]
prediction = predict_footprint(X_te, theta_opt)
#plot_comparative_results2(y_te, prediction, plot=True)  # folder=folder

print(type(y_te_study), y_te_study.shape)
print("1", y_te_study)
print("2", y_te)
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
    labels = from_dict_to_list_of_features_powers(powerdict)
    for index in range(len(theta_opt)):
        print(labels[index], theta_opt[index])
    print()
    print("Mean square error: ", MSE(prediction, y_te_study))
    print("Mean absolute error: ", MAE(prediction, y_te_study))
    print("Cost: ", evaluate_cost(prediction, y_te_study))
    folder = None
    if save_visu:
        folder = output_path
    plot_comparative_results(y_te_study, prediction, figure_name=OUTPUT_NAMES[study], plot=True) #folder=folder


runandprint()

"""
next : 

- convert qualitative list of int to boolean 1 vs 0 > to implement in polynomial function/ multiplicatively or additionaly
- plot average of qualitative values / excluding extreme samples > identify treds 
-shuffle values
-update powerdict - quality/quantity
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
"""
Questions : 
What happens if blank spaces in excel? > replace with blank elem!!
DF = only numbers ? no strings?

"""