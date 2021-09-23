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
full_df = my_prep_data.full_model_dataframe()
print("my_prep_data.full_df", full_df.shape, full_df)
# print("my_prep_data.y_df", my_prep_data.y_df.shape, my_prep_data.y_df)
#
# """
# 1.3. PROCESS DATA
# """
# my_proc_data = data_processing(my_prep_data)
#
# # #TODO : works without logit = true/ error otherwise
# # #TODO : why does it exceed occurences? too many "self." (especially at begining of class 'init'; look into super class?
# # #TODO : integrate logit format to steps > provide option "as int" (for raphs and NN) or "as logit" (for polynomial),
#  #TODO : NO later :enable display of graphs ?
#
# """
# 1.4. TUNE REGRESSION
# """
# my_linear_regression = linear_regression(my_proc_data)
#
# """
# 2. RUN
# """
# """
# 2.1. DISPLAY DATA
# """
# my_data = data_display(my_prep_data)
#
# for x in my_data.preprocessed_data.Model_Structural_Embodied_CO2.x_features:
#    my_data.plot_graph_adv(x, my_data.preprocessed_data.Model_Structural_Embodied_CO2.y_features[my_data.preprocessed_data.Model_Structural_Embodied_CO2.tCO2e_per_m2])
#
# """
# 2.2. DISPLAY RESULTS
# """
# my_results = results_displayer(my_linear_regression)
# my_results.print_results()
# """
# 2.3. ARCHIVE RESULTS
# """
# my_exports = archive_implementation(my_model, my_prep_data, my_proc_data, my_linear_regression, my_data, my_results)
#


 labels = my_prep_data.create_long_label_list()
# #merge = my_prep_data.merge_long_label_list()
# merge = my_prep_data.flatten(labels)
# lengths = my_prep_data.length_list(labels)
# expand = my_prep_data.expand_list(merge,lengths)
#
# condense_array = my_prep_data.create_condense_array()
# long_label_dict = my_prep_data.create_long_label_dict()
# df = my_prep_data.dataframe_from_subfeature()
# dff = my_prep_data.x_df
#
# print("c")
# print("labels", labels, len(labels))
# print("merge", merge, len(merge))
# print("lengths", lengths, len(lengths))
# # print("expand", expand, len(expand))
# # print("condense_array", len(condense_array), condense_array) #80
# # print("long_label_dict", len(long_label_dict), long_label_dict) #50
# # print("dictionary_of_data", len(my_prep_data.dictionary_of_data()),  my_prep_data.dictionary_of_data()) #14 include X and Y
# # print("d")
# #