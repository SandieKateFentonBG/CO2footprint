from _11_setup_define_model import *
from _12_setup_preprocess_data import *
from _13_setup_process_data import *
from _14_setup_tune_regression import *
from _21_run_display_data import *
from _22_run_display_results import *
from _23_run_archive_implementation import *

"""
1. SET UP
"""
"""
1.1. DEFINE MODEL
"""
my_model = Model_Structural_Embodied_CO2()
"""
1.2. PREPROCESS DATA
"""
my_prep_data = data_preprocessing(my_model)
labels = my_prep_data.create_long_label_list()
condense_array = my_prep_data.create_condense_array()
long_label_dict = my_prep_data.create_long_label_dict()

print("labels", labels, len(labels))
print("condense_array", len(condense_array), condense_array)
print("long_label_dict", len(long_label_dict), long_label_dict)

"""
1.3. PROCESS DATA
"""
my_proc_data = data_processing(my_prep_data)
"""
1.4. TUNE REGRESSION
"""
my_linear_regression = linear_regression(my_proc_data)

"""
2. RUN
"""
"""
2.1. DISPLAY DATA
"""
my_data = data_display(my_prep_data)
"""
2.2. DISPLAY RESULTS
"""
my_results = results_displayer(my_linear_regression)
"""
2.3. ARCHIVE RESULTS 
"""
#TODO: HERE
my_exports = archive_implementation(my_model, my_prep_data, my_proc_data, my_linear_regression, my_data, my_results)

"""my_model.export_model_data(format='.txt')
my_model.export_model_data(format='.csv')
my_exports.export_results_as_text()
my_exports.export_results_as_csv()"""









"""a = my_prep_data.dictionary_of_labels()
m = dict()

b = my_prep_data.dictionary_of_values()
c = my_prep_data.Model_Structural_Embodied_CO2.x_features_str
d= my_prep_data.full_model_dataframe()[0]
for k in c : #a.keys()
    m[k] = len(a[k])
print("a", a)
print("m", m)
print("b",b)
print("c",c)
print("d",d, type(d), d.shape)
print("len", b.values())
max_list = m.values()
print("l", max_list)

empty_matrixx = []
for condensed_row in d:
    empty_line = []
    #print("cr", condensed_row)
    for max, val in zip(max_list, condensed_row):
        empty_spot = [0]*max
        empty_spot[int(val)] = 1
        empty_line.append(empty_spot)
    #print("t", empty_line)
    empty_matrixx.append(empty_line)

print(len(b['Sector']))
long_labels = dict()
for k, v in a.items():
    new_val_list = []
    if k in c:
        for value in v:
            new_val = k + '_' + value
            new_val_list.append(new_val)
        long_labels[k] = new_val_list

print(long_labels)

long_labels_list = []
for l in long_labels.values():
    long_labels_list.append(l)
print(long_labels_list)
print(empty_matrixx)

long_label_dict = dict()
labels = []
for sublist in long_labels_list:
    for label in sublist:
        labels.append(label)
print("labels", len(labels), labels)
condense_array = []
for row in empty_matrixx:
    condense_list = []
    for sublist in row:
        for value in sublist:
            condense_list.append(value)
    condense_array.append(condense_list)
print("condense_array", condense_array)

for i in range(len(labels)):
    long_label_dict[labels[i]] = []
    for row in condense_array:
        long_label_dict[labels[i]].append(row[i])

print("long_label_dict", long_label_dict)
"""














preprocessed_df = my_prep_data.full_model_dataframe()
polynomial_features = my_proc_data.create_polynomial_features()
#print(preprocessed_df[0])
#print(polynomial_features)
#parameters = my_linear_regression.optimize_parameters() #20 on polynomial size
#prediction = my_linear_regression.predict_footprint() #16 on testing set
#for x in my_data.preprocessed_data.Model_Structural_Embodied_CO2.x_features:
#    my_data.plot_graph_adv(x, my_data.preprocessed_data.Model_Structural_Embodied_CO2.y_features[my_data.preprocessed_data.Model_Structural_Embodied_CO2.tCO2e_per_m2])
#my_plot = my_results.plot_comparative_results()
#param_matrix = my_results.display_parameter_matrix()
#my_results.print_results()
#txt_file_adv = my_exports.export_results_as_text_adv()
#json_file = my_exports.export_results_as_json()