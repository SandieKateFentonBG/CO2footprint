from model_structural_embodied_co2 import *

class data_preprocessing():

    def __init__(self, Model_Structural_Embodied_CO2, first_line=5, delimiter=';'):

        #super().__init__() #TODO: ?
        self.delimiter = delimiter
        self.first_line = first_line
        self.Model_Structural_Embodied_CO2 = Model_Structural_Embodied_CO2
        self.unit = [self.Model_Structural_Embodied_CO2.y_features[self.Model_Structural_Embodied_CO2.tCO2e_per_m2]]
        self.x_qual_df = self.dataframe_from_feature(self.Model_Structural_Embodied_CO2.x_features_str)
        self.x_quant_df = self.dataframe_from_feature(self.Model_Structural_Embodied_CO2.x_features_int)
        self.x_df = self.dataframe_from_feature(self.Model_Structural_Embodied_CO2.x_features)
        self.y_df = self.dataframe_from_feature(self.unit)

    def open_csv_at_given_line(self):
        import csv
        reader = csv.reader(open(self.Model_Structural_Embodied_CO2.input_path + '.csv', mode='r'), delimiter=self.delimiter)
        for i in range(self.first_line):
            reader.__next__()
        header = reader.__next__()
        return header, reader

    def index_dict_from_csv(self):
        header, reader = self.open_csv_at_given_line()
        CST = dict()
        for f in self.Model_Structural_Embodied_CO2.x_features:
            CST[f] = []
        for line in reader:
            for f in self.Model_Structural_Embodied_CO2.x_features:
                index = header.index(f)
                if line[index] not in CST[f]:
                    CST[f].append(line[index])
        return CST

    def separate_X_Y_values(self):
        header, reader = self.open_csv_at_given_line()
        X_values, Y_values = [], []
        for line in reader:
            for (names, values) in [(self.Model_Structural_Embodied_CO2.x_features, X_values),
                                    (self.Model_Structural_Embodied_CO2.y_features, Y_values)]:
                values.append([line[header.index(name)] for name in names])
        return X_values, Y_values
        # line[header.index(name)] = value in that column

    def build_dictionary(self):  # TODO : shorten
        dico = dict()
        X_values, Y_values = self.separate_X_Y_values()
        for i in range(len(X_values[0])):  # 12
            dico[self.Model_Structural_Embodied_CO2.x_features[i]] = []
        for i in range(len(Y_values[0])):  # 12
            dico[self.Model_Structural_Embodied_CO2.y_features[i]] = []
        for j in range(len(X_values)):  # 80
            for i in range(len(X_values[0])):  # 12
                dico[self.Model_Structural_Embodied_CO2.x_features[i]].append(X_values[j][i])
            for k in range(len(Y_values[0])):
                dico[self.Model_Structural_Embodied_CO2.y_features[k]].append(Y_values[j][k])
        return dico

    def remap_qual_feature_as_int(self, index_dictionary, feature_name, feature_value):

        return index_dictionary[feature_name].index(feature_value)

    def format_str_feature_to_float(self, string):
        """
        input : decimal number in string with "," for decimal separation
        output : decimal number in float with "." for decimal separation
        """

        try:
            return float(string.replace(',', '.'))
        except:
            print(string, ": this should be a number")
            return False

    def string_dict_to_number_dict(self):
        number_dict = dict()
        index_dict = self.index_dict_from_csv()
        feature_dict = self.build_dictionary()
        qualitative_features = self.Model_Structural_Embodied_CO2.x_features_str
        quantitative_features = self.Model_Structural_Embodied_CO2.x_features_int + self.Model_Structural_Embodied_CO2.y_features
        for ql_feature in qualitative_features:
            number_dict[ql_feature] = []
            for ql_value in feature_dict[ql_feature]:
                number_dict[ql_feature].append(self.remap_qual_feature_as_int(index_dict, ql_feature, ql_value))
        for qn_feature in quantitative_features:
            number_dict[qn_feature] = []
            for qn_value in feature_dict[qn_feature]:
                number_dict[qn_feature].append(self.format_str_feature_to_float(qn_value))
        return number_dict

    def dataframe_from_feature(self, feature_labels, scale = False):
        import numpy as np
        data_dict = self.string_dict_to_number_dict()
        num_samples = len(data_dict[feature_labels[0]])
        num_features = len(feature_labels)
        data = np.zeros((num_samples, num_features))
        for i in range(num_samples):  # 80
            data[i, :] = np.array([data_dict[f][i] for f in feature_labels])
        if scale:
            data = self.scale_features(data)
        return data

    def full_model_dataframe(self, scale_qual=True, scale_int=True, scale_y=True,):

        x_qual_df = self.dataframe_from_feature(self.Model_Structural_Embodied_CO2.x_features_str)
        x_quant_df = self.dataframe_from_feature(self.Model_Structural_Embodied_CO2.x_features_int)
        y_df = self.dataframe_from_feature(self.Model_Structural_Embodied_CO2.y_features)
        """
        if scale_qual:
            x_qual_df = self.scale_features(x_qual_df)
        if scale_int:
            x_quant_df = self.scale_features(x_quant_df)
        if scale_y:
            y_df = self.scale_features(y_df)"""

        return x_qual_df, x_quant_df, y_df

    def x_qualitative_df(self, scale_int=True):
        x_qual_df, x_quant_df, y_df = extract_X_y_df_from_dict

    def scale_features(self, X):
        import numpy as np
        x_bar = np.mean(X, axis=0)
        x_std = np.std(X, axis=0)
        return (X - x_bar) / x_std


