from _12_setup_preprocess_data import *




def covert_feature_int_to_logit(self):
    labels = self.dictionary_of_labels()
    qual_labels = self.Model_Structural_Embodied_CO2.x_features_str
    long_labels = dict()
    for k,v in labels.items():
        new_val_list = []
        if k in qual_labels:
            for value in v:
                new_val = k +'_' + value
                new_val_list.append(new_val)
            long_labels[k] = new_val_list

    long_labels_list = []
    for l in long_labels.values():
        long_labels_list.append(l)

    long_label_dict = dict ()
    labels = []
    for sublist in long_labels_list :
        for label in sublist:
            labels.append(label)
    #for label in labels:
    #    long_label_dict[label] = []
    condense_array = []
    for row in empty_matrix:
        condense_list = []
        for sublist in row:
            for value in sublist:
                condense_list.append(value)
        condense_array.append(condense_list)

    for i in range(len(labels)):
        long_label_dict[labels[i]] = []
        for row in condense_array:
            long_label_dict[labels[i]].append(row[i])




    values = self.dictionary_of_values()
    qual_labels = self.Model_Structural_Embodied_CO2.x_features_str
    data = self.full_model_dataframe()[0]
    lengths = dict()
    for k in qual_labels:
        lengths[k] = len(labels[k])
        names[k] = k +
    max_list = lengths.values()

    empty_matrix = []
    for condensed_row in data:
        empty_line = []
        #print("cr", condensed_row)
        for max, val in zip(max_list, condensed_row):
            empty_spot = [0]*max
            empty_spot[int(val)] = 1
            empty_line.append(empty_spot)
        #print("t", empty_line)
        empty_matrix.append(empty_line)


"""def x_qualitative_df(self, scale_int=True):
    preprocessed_df = my_prep_data.full_model_dataframe()
    single_list = self.full_model_dataframe()[0]
    CST = self.index_dict_from_csv()
    for project in single_list:
        for i in range(len(single_list[0])):
        max = max
    fractured_list =


    def qual_to_logits(self):
        number_dict = dict()
        max_dict = dict()
        index_dict = self.index_dict_from_csv()
        for k in index_dict:
            max_dict[k] = len(index_dict[k])
        feature_dict = self.build_dictionary()
        qualitative_features = self.Model_Structural_Embodied_CO2.x_features_str
        single_list = self.full_model_dataframe()[0]



        for ql_feature in qualitative_features:
            number_dict[ql_feature] = []
            for ql_value in feature_dict[ql_feature]:
                number_dict[ql_feature].append(self.remap_qual_feature_as_int(index_dict, ql_feature, ql_value))



def max_value(self):
    max = 1
    new_list = []
    for feature in feature_list:
        for property in feature:

    max = max


def feature_as_binary(self, count):
    count =
"""