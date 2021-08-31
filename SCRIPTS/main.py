"""
1. IMPORT MODEL FEATURES
"""

from model_features import *
mymodel = model_features()

"""
2. IMPORT DATA
"""

from data_preprocessing import *
my_data = data_preprocessing(mymodel, mymodel.FEATURES_NAMES, mymodel.OUTPUT_NAMES)
print(my_data.string_dict_to_number_dict)
print(type(my_data.string_dict_to_number_dict))


"""def op_csv_at_given_line():
    import csv
    reader = csv.reader(open('C:/Users/sfenton/Code/Repositories/CO2footprint/DATA/210413_PM_CO2_data' + '.csv', mode='r'),
                        delimiter=';')
    for i in range(5):
        reader.__next__()
    header = reader.__next__()
    return header, reader
print 
"""

print("1", my_data.index_dict_from_csv())
print("2", my_data.split_X_Y_values())
print("3", my_data.build_dictionary())
print("4", my_data.string_dict_to_number_dict())
print("5", my_data.extract_feature_df_from_dict(my_data.features.STR_FEATURES))
print("6", my_data.extract_X_y_df_from_dict())

"""
3. MAP DATA TO OBJECTIVE FUNCTION
"""
from matrix_display import *
dict = my_data.string_dict_to_number_dict()
print("a", dict)
#my_matrix = matrix_display(data_preprocessing(mymodel, mymodel.FEATURES_NAMES, mymodel.OUTPUT_NAMES))
my_matrix = matrix_display(my_data)
print("7", my_matrix)
print("8", my_matrix.data.delimiter)
print("9", my_matrix.data.features.input_path)


unit = my_matrix.data.features.tCO2e_per_m2
for x in my_matrix.data.features.FEATURES_NAMES:
    my_matrix.plot_graph_adv(x, my_matrix.data.features.OUTPUT_NAMES[unit])

print(my_matrix.view_dataframe_from_dict(my_data.string_dict_to_number_dict()))


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
