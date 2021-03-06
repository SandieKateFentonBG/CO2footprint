"""
1. DISPLAY PARAMETERS
"""
#to update
date = "'210416"
test_count = str(1)
save_visu = False
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
mse = MSE(prediction, y_te_study)
mae = MAE(prediction, y_te_study)
cost = evaluate_cost(prediction, y_te_study)

from results_displayer import *
polynomial_labels = from_dict_to_list_of_features_powers(power_dict_int_features, INT_FEATURES)
res_dict = dictionary_of_results(OUTPUT_NAMES[study], reg, polynomial_labels, theta_opt, mse, mae, cost)
pr = print_results(res_dict)

lists = [res_dict[key] for key in res_dict.keys()]
print(lists)
print(res_dict.keys(), type(res_dict.keys()))
"""
4. RUN SCRIPTS
"""


def runandprint():
    print("object of study : ", OUTPUT_NAMES[study])
    print("regularization param : ", reg)
    print("theta_opt : ")
    for index in range(len(theta_opt)):
        print(polynomial_labels[index], theta_opt[index])
    print()
    print("Mean square error: ", MSE(prediction, y_te_study))
    print("Mean absolute error: ", MAE(prediction, y_te_study))
    print("Cost: ", evaluate_cost(prediction, y_te_study))

#runandprint()

"""
5. DISPLAY RESULTS
"""

folder_path = output_path if save_visu else None
if display_plots or save_visu :
    full_df = view_dataframe_from_dict(nb_dict, disp=False)
    for x in FEATURES_NAMES:
        plot_graph_adv(full_df, x, OUTPUT_NAMES[study],label_dict=id_dict, qual_features=STR_FEATURES,
                       new_folder_path=folder_path, plot=display_plots, reference=reference)

if display_plots or save_visu:
    plot_comparative_results(y_te_study, prediction, figure_name=OUTPUT_NAMES[study],
                             folder=folder_path, plot=display_plots, reference=reference)


"""
5. DISPLAY RESULTS
"""

def export_results_as_json(filename, res_dict):
    import json
    with open(filename + "results_as_json.txt", 'w') as file:
        file.write(json.dumps(res_dict))
    file.close()

def export_results_as_text(filename, res_dict):

    fo = open(filename + "results.txt", 'a')
    for k, v in res_dict.items():
        if len(res_dict[k]) <= 1:
            fo.write(str(k) + ' >>> '+ str(v) + '\n\n')
        if len(res_dict[k]) > 1:
            fo.write(str(k) + ' >>> ' + '\n\n')
            for i in range(len(res_dict[k])):
                fo.write(str(i) + '  '+ str(res_dict[k][i]) + '\n\n')
    fo.close()


def export_results_as_text_adv(path, res_dict, key_a, key_b, filename="results.txt"):

    fo = open(path + filename, 'w')
    for k, v in res_dict.items():
        if len(res_dict[k]) <= 1:
            fo.write(str(k) + ' : '+ str(res_dict[k][0]) + '\n\n')

    a = [key_a]+res_dict[key_a]
    b = [key_b]+res_dict[key_b]
    c = [a, b]
    for x in zip(*c):
        fo.write("{0}\t{1}\n".format(*x))
    fo.close()

"""
def export_results_as_csv(path, res_dict, filename="results.txt"):
    import csv

    with open(path + filename + '.csv', 'w', ) as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        my_lists=[]
        for key in res_dict.keys():
            my_list = [key]+res_dict[key]
            #wr.writerow(my_list)
            for v in my_list:
                wr.writerow([v])

    with open(, 'w', ) as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        my_lists=[]
        for key in res_dict.keys():
            my_list = [key]+res_dict[key]
            #wr.writerow(my_list)
            for v in my_list:
                wr.writerow([v])
                
                
                
"""






test = export_results_as_json(output_path, res_dict)
test2 = export_results_as_text_adv(output_path, res_dict,"polynomial_exponents", "theta_opt", filename=reference)
test3 = export_results_as_csv(output_path, res_dict, filename = reference)
test4 = csv_into_columns(output_path + reference,output_path, reference + "_col" )

"""
def open_csv_at_given_line(filename, first_line, delimiter):


def export_results_to_database():
    pass

    """


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