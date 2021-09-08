"""
1. SET MODEL FEATURES
"""
from model_structural_embodied_co2 import *
mymodel = Model_Structural_Embodied_CO2()

"""
2. PREPROCESS DATA
"""
from data_preprocessing import *
my_prep_data = data_preprocessing(mymodel)

"""
3. VISUALIZE DATA
"""
from display_preprocessed import *
my_matrix = matrix_display(my_prep_data)
#for x in my_matrix.preprocessed_data.Model_Structural_Embodied_CO2.x_features:
#    my_matrix.plot_graph_adv(x, my_matrix.preprocessed_data.Model_Structural_Embodied_CO2.y_features[my_matrix.preprocessed_data.Model_Structural_Embodied_CO2.tCO2e_per_m2])

"""
4. MAP DATA TO OBJECTIVE FUNCTION
"""
from data_processing import *
my_proc_data = data_processing(my_prep_data)
polynomial_df = my_proc_data.create_polynomial_features()

"""
5. OPTIMIZE/RUN/EVALUATE OBJECTIVE FUNCTION
"""
from linear_regression import *
my_linear_regression = linear_regression(my_proc_data)
prediction = my_linear_regression.predict_footprint()
mse = my_linear_regression.MSE()
mae = my_linear_regression.MAE()
cost = my_linear_regression.evaluate_cost()
print(prediction, prediction.shape)


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
