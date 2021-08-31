class data_preprocessing:
    def __init__(self, my_model_features, x_names, y_names, first_line=5, delimiter=';'):
        #super().__init__() #TODO: ?
        self.delimiter = delimiter
        self.first_line = first_line
        self.features = my_model_features
        self.x_names = x_names
        self.y_names = y_names

    def open_csv_at_given_line(self):
        import csv
        reader = csv.reader(open(self.features.input_path + '.csv', mode='r'), delimiter=self.delimiter)
        for i in range(self.first_line):
            reader.__next__()
        header = reader.__next__()
        return header, reader

    def index_dict_from_csv(self):
        header, reader = self.open_csv_at_given_line()  # TODO: parenthesis needed here?
        CST = dict()
        for f in self.features.FEATURES_NAMES:
            CST[f] = []
        for line in reader:
            for f in self.features.FEATURES_NAMES:
                index = header.index(f)
                if line[index] not in CST[f]:
                    CST[f].append(line[index])
        return CST

    def split_X_Y_values(self):
        header, reader = self.open_csv_at_given_line()
        X_values, Y_values = [], []
        for line in reader:
            for (names, values) in [(self.x_names, X_values), (self.y_names, Y_values)]:
                values.append([line[header.index(name)] for name in names])
        return X_values, Y_values
        # line[header.index(name)] = value in that column

    def build_dictionary(self):  # TODO : shorten
        dico = dict()
        X_values, Y_values = self.split_X_Y_values()
        for i in range(len(X_values[0])):  # 12
            dico[self.features.FEATURES_NAMES[i]] = []
        for i in range(len(Y_values[0])):  # 12
            dico[self.features.OUTPUT_NAMES[i]] = []
        for j in range(len(X_values)):  # 80
            for i in range(len(X_values[0])):  # 12
                dico[self.features.FEATURES_NAMES[i]].append(X_values[j][i])
            for k in range(len(Y_values[0])):
                dico[self.features.OUTPUT_NAMES[k]].append(Y_values[j][k])
        return dico

    def qualitative_str_feature_to_int(self, index_dictionary, feature_name, feature_value):

        return index_dictionary[feature_name].index(feature_value)

    def quantitative_str_feature_to_float(self, string):
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
        qualitative_features = self.features.STR_FEATURES
        quantitative_features = self.features.INT_FEATURES + self.features.OUTPUT_NAMES
        for ql_feature in qualitative_features:
            number_dict[ql_feature] = []
            for ql_value in feature_dict[ql_feature]:
                number_dict[ql_feature].append(self.qualitative_str_feature_to_int(index_dict, ql_feature, ql_value))
        for qn_feature in quantitative_features:
            number_dict[qn_feature] = []
            for qn_value in feature_dict[qn_feature]:
                number_dict[qn_feature].append(self.quantitative_str_feature_to_float(qn_value))
        return number_dict

    def extract_feature_df_from_dict(self, feature_labels):
        import numpy as np
        data_dict = self.string_dict_to_number_dict()
        num_samples = len(data_dict[feature_labels[0]])
        num_features = len(feature_labels)
        data = np.zeros((num_samples, num_features))
        for i in range(num_samples):  # 80
            data[i, :] = np.array([data_dict[f][i] for f in feature_labels])
        return data

    def extract_X_y_df_from_dict(self, scale_int=True):  # TODO : USELESS

        x_qual_df = self.extract_feature_df_from_dict(self.features.STR_FEATURES)
        x_quant_df = self.extract_feature_df_from_dict(self.features.INT_FEATURES)
        if scale_int:
            x_quant_df = self.scale_features(x_quant_df)
        y_df = self.extract_feature_df_from_dict(self.features.OUTPUT_NAMES)
        return x_qual_df, x_quant_df, y_df

    def scale_features(self, X):
        import numpy as np
        x_bar = np.mean(X, axis=0)
        x_std = np.std(X, axis=0)
        return (X - x_bar) / x_std


"""
class data_preprocessing:
    def __init__(self, first_line=5, delimiter=';'):
        super().__init__()
        self.str = model_features.STR_FEATURES
        self.int = model_features.INT_FEATURES
        self.features = model_features.FEATURES_NAMES
        self.outputs = model_features.OUTPUT_NAMES
        self.filename = model_features.input_path
        self.delimiter = delimiter
        self.first_line = first_line




"""