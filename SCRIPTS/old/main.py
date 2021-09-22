"""
1. SET MODEL FEATURES
"""
from model_structural_embodied_co2 import *
mymodel = Model_Structural_Embodied_CO2()

"""
2. IMPORT DATA
"""
from data_preprocessing import *
my_prep_data = data_preprocessing(mymodel)


print("0", my_prep_data.dictionary_of_data)
print("1", my_prep_data.dictionary_of_labels())
print("2", my_prep_data.separate_X_Y_values())
print("3", my_prep_data.dictionary_of_values())
print("4", my_prep_data.dictionary_of_data())
print("5", my_prep_data.dataframe_from_feature(my_prep_data.Model_Structural_Embodied_CO2.x_features_str))
print("6", my_prep_data.full_model_dataframe())

"""
3. DATA DISPLAY
"""
from display_preprocessed import *
#dict = my_data.string_dict_to_number_dict()
#print("a", dict)

my_matrix = matrix_display(my_prep_data)
unit = my_matrix.preprocessed_data.Model_Structural_Embodied_CO2.tCO2e_per_m2




#for x in my_matrix.preprocessed_data.Model_Structural_Embodied_CO2.x_features:
#    my_matrix.plot_graph_adv(x, my_matrix.preprocessed_data.Model_Structural_Embodied_CO2.y_features[unit])

#print("7", my_matrix)
#print("8", my_matrix.preprocessed_data.delimiter)
#print("9", my_matrix.preprocessed_data.Model_Structural_Embodied_CO2.input_path)
print(my_matrix.view_dataframe_from_dict(my_prep_data.dictionary_of_data()))

"""
4. MAP DATA TO OBJECTIVE FUNCTION
"""

from data_processing import *

my_proc_data = data_processing(my_prep_data)
print("10", my_proc_data.construct_power_dictionary())
df,polf = my_proc_data.create_polynomial_features()
print("df", df)
print("polf", polf)
#print("df", type(df), df.shape, "line", df[1, :], "col", df[:, 1])
#print("df", type(df), df.shape, "line", df[-1, :], "col", df[:, -1])
#print("polf", type(polf), polf.shape, "line", polf[0, :], "col", polf[:, 0])
#print("polf", type(polf), polf.shape, "line", polf[-1, :], "col", polf[:, -1])


"""
4. EVALUATE MODEL
"""

"""
5. HANDLE RESULTS
"""

"""
from data_handler import *

#mymodel.printMe("a") #TODO : why does this not print?
#print('a', mymodel.__dict__.keys())
#print("o", type(mymodel.printMe))
#print("u", type(mymodel.printMe('a')))

#for k, v in mymodel.__dict__.items():
#    print(' ', k, ' : ', v)


csv_firstline = 5
X_data, y = split_X_Y_values(mymodel.FEATURES_NAMES, mymodel.OUTPUT_NAMES, mymodel.input_path, csv_firstline, delimiter=';')
id_dict = index_dict_from_csv(mymodel.input_path, csv_firstline)
data_dict = build_dictionary(X_data, y, mymodel.FEATURES_NAMES, mymodel.OUTPUT_NAMES)
nb_dict = string_dict_to_number_dict(id_dict, data_dict, mymodel.STR_FEATURES, mymodel.INT_FEATURES + mymodel.OUTPUT_NAMES)
x_qlt_df, x_qtt_df, y_df = extract_X_y_df_from_dict\
    (nb_dict, mymodel.STR_FEATURES, mymodel.INT_FEATURES, OUTPUT_NAMES, scale_int=f_scaling)
"""
