from _11_setup_define_model import *
import numpy as np

class data_preprocessing():

    def __init__(self, my_model, first_line=5, delimiter=';', logit=True):

        #super().__init__() #TODO: ?
        self.delimiter = delimiter
        self.first_line = first_line
        self.Model_Structural_Embodied_CO2 = my_model
        self.unit = [self.Model_Structural_Embodied_CO2.y_features[self.Model_Structural_Embodied_CO2.tCO2e_per_m2]]
        # self.x_qual_df = self.full_model_dataframe()[0]
        # self.x_quant_df = self.full_model_dataframe()[1]
        # self.x_df = np.concatenate((self.x_qual_df, self.x_quant_df), axis=1)
        # self.y_df = self.dataframe_from_feature(self.unit)

    def open_csv_at_given_line(self):
        import csv
        reader = csv.reader(open(self.Model_Structural_Embodied_CO2.input_path + '.csv', mode='r'), delimiter=self.delimiter)
        for i in range(self.first_line):
            reader.__next__()
        header = reader.__next__()
        return header, reader

    def dictionary_of_labels(self):
        """
        {'Sector': ['Other', 'Residential', 'Cultural', 'Educational', 'Mixed Use', 'Commercial', 'Industrial'],
         'Type': ['New Build (Brownfield)', 'New Build (Greenfield)', 'Mixed New Build/Refurb'],
         'Basement': ['None', 'Partial Footprint', 'Full Footprint'],
         'Foundations': ['Piled Ground Beams', 'Mass Pads/Strips', 'Raft', 'Piles (Pile Caps)', 'Reinforced Pads/Strips', ''],
         'Ground Floor': ['Suspended RC', 'Ground Bearing RC', 'Suspended Precast', 'Raft', 'Other'],
         'Superstructure': ['In-Situ RC', 'CLT Frame', 'Steel Frame, Precast', 'Masonry, Concrete', 'Steel Frame, Composite', 'Steel Frame, Other', 'Masonry, Timber', 'Other', 'Timber Frame', 'Steel Frame, Timber'],
         'Cladding': ['Masonry + SFS', 'Lightweight Only', 'Stone + Masonry', 'Glazed/Curtain Wall', 'Masonry Only', 'Stone + SFS', 'Other', 'Lightweight + SFS', 'Timber + SFS', 'Timber Only'],
         'BREEAM Rating': ['Unknown', 'Very Good', 'Excellent', 'Good', 'Passivhaus', 'Outstanding'],
         'GIFA (m2)': ['4631', '4640', '742', '1534', '7284', '4180', '15145', '4500', '3900', '1724', '2772', '904', '1050', '1555', '8118', '1874', '1305', '2349', '7755', '2100', '15985', '3932', '3460', '11500', '9483', '31774', '2276', '220', '1311', '11736', '7922', '544', '5620', '1712', '37115', '14029', '1360', '1154', '862', '3499', '10108', '9862', '1290', '5428', '125', '19512', '2395', '10220', '5500', '1729', '2054', '1210', '2769', '3600', '2400', '746', '810', '648', '500', '1100', '1900', '3023', '867', '21500', '3997', '3657', '1830', '517', '1643', '2368', '2404', '1930', '8748', '1163', '2575', '4787', '3851', '307', '340'],
         'Storeys': ['3', '7', '2', '6', '5', '10', '4', '1', '9', '8', '19', '17', '24'],
         'Typical Span (m)': ['7,00', '4,00', '2,50', '7,50', '4,50', '6,50', '8,00', '3,00', '6,00', '45,00', '5,74', '5,00', '6,40', '7,20', '8,10', '7,80', '0,00', '3,50', '35,00', '2,90', '5,05', '3,20', '8,50', '15,00', '5,50', '12,00', '3,85', '6,24', '20,50'],
         'Typ Qk (kN_per_m2)': ['4,00', '2,50', '3,00', '0,75', '5,00', '3,50', '2,00', '1,50', '10,00', '5,50', '0,85']}
        """

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

    def dictionary_of_values(self):  # TODO : shorten
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

    def convert_feature_qual_to_int(self, index_dictionary, feature_name, feature_value):

        return index_dictionary[feature_name].index(feature_value)



    def create_long_labels(self):

        labels = self.dictionary_of_labels()
        qual_labels = self.Model_Structural_Embodied_CO2.x_features_str
        long_labels = dict()
        for k, v in labels.items():
            new_val_list = []
            if k in qual_labels:
                for value in v:
                    new_val = k + '_' + value
                    new_val_list.append(new_val)
                long_labels[k] = new_val_list
        return long_labels

    def create_long_label_list(self):
        long_labels = self.create_long_labels()
        long_labels_list = []
        for l in long_labels.values():
            long_labels_list.append(l)
        return long_labels_list

    def flatten(self, list):
        return [item for sublist in list for item in sublist]

    def length_list(self, list):
        return [len(item) for item in list]

    def expand_list(self, list, shape):

        return self.flatten([[item]*size for item, size in zip(list, shape)])

    def create_long_label_dict(self):
        long_labels_list = self.create_long_label_list()
        long_label_dict = dict()
        labels = []
        for sublist in long_labels_list:
            for label in sublist:
                labels.append(label)
        condense_array = self.create_condense_array()
        for i in range(len(labels)):
            long_label_dict[labels[i]] = []
            for row in condense_array:
                long_label_dict[labels[i]].append(row[i])
        return long_label_dict

    def create_condense_array(self):

        logit_matrix = self.create_logit_matrix()
        condense_array = []
        for row in logit_matrix:
            condense_list = []
            for sublist in row:
                for value in sublist:
                    condense_list.append(value)
            condense_array.append(condense_list)

        return condense_array

    def create_max_list(self):
        qual_labels = self.Model_Structural_Embodied_CO2.x_features_str
        labels = self.dictionary_of_labels()
        lengths = dict()
        for k in qual_labels:
            lengths[k] = len(labels[k])
        max_list = lengths.values()
        return max_list

    def create_logit_matrix(self):
        logit_matrix = []
        data = self.full_model_dataframe()[0]
        max_list = self.create_max_list()
        for condensed_row in data:
            logit_line = []
            for max, val in zip(max_list, condensed_row):
                empty_spot = [0] * max
                empty_spot[int(val)] = 1
                logit_line.append(empty_spot)
            logit_matrix.append(logit_line)

        return logit_matrix


    def convert_feature_str_to_float(self, string):
        """
        input : decimal number in string with "," for decimal separation
        output : decimal number in float with "." for decimal separation
        """

        try:
            return float(string.replace(',', '.'))
        except:
            print(string, ": this should be a number")
            return False

    def dictionary_of_data(self):
        number_dict = dict()
        index_dict = self.dictionary_of_labels()
        feature_dict = self.dictionary_of_values()
        qualitative_features = self.Model_Structural_Embodied_CO2.x_features_str
        quantitative_features = self.Model_Structural_Embodied_CO2.x_features_int + self.Model_Structural_Embodied_CO2.y_features
        for ql_feature in qualitative_features:
            number_dict[ql_feature] = []
            for ql_value in feature_dict[ql_feature]:
                number_dict[ql_feature].append(self.convert_feature_qual_to_int(index_dict, ql_feature, ql_value))
        for qn_feature in quantitative_features:
            number_dict[qn_feature] = []
            for qn_value in feature_dict[qn_feature]:
                number_dict[qn_feature].append(self.convert_feature_str_to_float(qn_value))
        return number_dict

    def dataframe_from_feature(self, feature_labels):
        scale = self.Model_Structural_Embodied_CO2.f_scaling
        import numpy as np
        data_dict = self.dictionary_of_data()
        num_samples = len(data_dict[feature_labels[0]])
        num_features = len(feature_labels)
        data = np.zeros((num_samples, num_features))
        for i in range(num_samples):  # 80
            data[i, :] = np.array([data_dict[f][i] for f in feature_labels])
        if scale:
            data = self.scale_features(data)
        return data

    def dataframe_from_subfeature(self):
        import numpy as np

        long_label_dict = self.create_long_label_dict()
        condense_array = self.create_condense_array()
        num_features = len(long_label_dict) #50
        num_samples = len(condense_array)
        data = np.zeros((num_samples, num_features))
        for i in range(num_samples):  # 80
            data[i, :] = np.array([long_label_dict[f][i] for f in long_label_dict.keys()])
        return data

    def full_model_dataframe(self): #TODO: update default here
        if self.Model_Structural_Embodied_CO2.logit:
            x_qual_df = self.dataframe_from_subfeature()
        else :
            x_qual_df = self.dataframe_from_feature(self.Model_Structural_Embodied_CO2.x_features_str)
        x_quant_df = self.dataframe_from_feature(self.Model_Structural_Embodied_CO2.x_features_int)
        y_df = self.dataframe_from_feature(self.Model_Structural_Embodied_CO2.y_features)

        return x_qual_df, x_quant_df, y_df


    def scale_features(self, X):
        import numpy as np
        x_bar = np.mean(X, axis=0)
        x_std = np.std(X, axis=0)
        return (X - x_bar) / x_std


