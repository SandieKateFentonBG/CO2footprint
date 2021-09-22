"""
1. SET UP
"""
from _11_setup_define_model import *
from _12_setup_preprocess_data import *
from _13_setup_process_data import *
from _14_setup_tune_regression import *
"""
1.1. DEFINE MODEL
"""
my_model = Model_Structural_Embodied_CO2()
"""
1.2. PREPROCESS DATA
"""
my_prep_data = data_preprocessing(my_model)
"""
1.3. PROCESS DATA
"""
my_proc_data = data_processing(my_prep_data)
"""
1.4. TUNE REGRESSION
"""
my_linear_regression = linear_regression(my_proc_data)

#my_model.printMe()
#preprocessed_df = my_prep_data.full_model_dataframe()
#polynomial_features = my_proc_data.create_polynomial_features()
#parameters = my_linear_regression.optimize_parameters() #20 on polynomial size
#prediction = my_linear_regression.predict_footprint() #16 on testing set

"""
2. RUN
"""
from _21_run_display_data import *
from _22_run_display_results import *
from _23_run_archive_implementation import *

"""
2.1. DISPLAY DATA
"""
my_data = data_display(my_prep_data)
"""
2.2. DISPLAY RESULTS
"""
my_results = results_displayer(my_linear_regression)


#for x in my_data.preprocessed_data.Model_Structural_Embodied_CO2.x_features:
#    my_data.plot_graph_adv(x, my_data.preprocessed_data.Model_Structural_Embodied_CO2.y_features[my_data.preprocessed_data.Model_Structural_Embodied_CO2.tCO2e_per_m2])

#my_plot = my_results.plot_comparative_results()
#param_matrix = my_results.display_parameter_matrix()
#my_results.print_results()


"""
2.3. ARCHIVE RESULTS 
"""

#TODO: HERE
my_exports = archive_implementation(my_model, my_prep_data, my_proc_data, my_linear_regression, my_data, my_results)


txt_file = my_exports.export_results_as_text()
csv_file = my_exports.export_results_as_csv()
csv_test = my_exports.csv_into_columns("C:/Users/sfenton/Code/Repositories/CO2footprint/RESULTS/210910_results/results.csv")

#txt_file_adv = my_exports.export_results_as_text_adv()
#json_file = my_exports.export_results_as_json()