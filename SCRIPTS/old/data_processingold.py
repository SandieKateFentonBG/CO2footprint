import numpy as np

class data_processing():

    def __init__(self, preprocessed_data):
        self.preprocessed_data = preprocessed_data

    def longest(self, list):
        return max(len(elem) for elem in list)

    def power_dictionary(self):
        dictio = dict()
        all_features = self.preprocessed_data.Model_Structural_Embodied_CO2.x_features_str + self.preprocessed_data.Model_Structural_Embodied_CO2.x_features_int
        all_powers = [self.preprocessed_data.Model_Structural_Embodied_CO2.Sector_power,
                      self.preprocessed_data.Model_Structural_Embodied_CO2.Type_power,
                      self.preprocessed_data.Model_Structural_Embodied_CO2.Basement_power,
                      self.preprocessed_data.Model_Structural_Embodied_CO2.Foundations_power,
                      self.preprocessed_data.Model_Structural_Embodied_CO2.Groundfloor_power,
                      self.preprocessed_data.Model_Structural_Embodied_CO2.Superstructure_power,
                      self.preprocessed_data.Model_Structural_Embodied_CO2.Cladding_power,
                      self.preprocessed_data.Model_Structural_Embodied_CO2.Rating_power,
                      self.preprocessed_data.Model_Structural_Embodied_CO2.GIFA_power,
                      self.preprocessed_data.Model_Structural_Embodied_CO2.Storeys_power,
                      self.preprocessed_data.Model_Structural_Embodied_CO2.Span_power,
                      self.preprocessed_data.Model_Structural_Embodied_CO2.Qk_power]
        selected_features = []
        selected_powers = []
        max_power = self.longest(all_powers)

        for feature, power_list in zip(all_features, all_powers):
            if power_list:
                dictio[feature] = power_list + [0]*(max_power-len(power_list))
                selected_features.append(feature)
                selected_powers.append(power_list + [0]*(max_power-len(power_list)))

        return dictio, selected_features, selected_powers


    def power_up_feature(self, featureArray_column, powerList):
        list_of_virtual_features = []
        for power in powerList:
            if power == 0:
                replacer = np.zeros(len(featureArray_column))
                list_of_virtual_features.append(replacer.reshape([-1, 1]))
            else:
                list_of_virtual_features.append(np.power(featureArray_column, power).reshape([-1, 1]))
                #list_of_virtual_features.append(np.power(featureArray_column, power).reshape([-1, 1]))

        return np.hstack(list_of_virtual_features)

    def create_polynomial_features(self, scale = False):
        #TODO : exposant 0 = 1 < on veut un resultat = 0 ; why dim = 37, where is this extra from?
        #TODO : utiliser un masque !!
        #verifier le in range
        powerdict, selected_features, selected_powers = self.power_dictionary()
        x_dataframe = self.preprocessed_data.dataframe_from_feature(selected_features,scale)
        polynomial_features = np.ones((x_dataframe.shape[0], 1))
        for feature_index in range(x_dataframe.shape[1]):
            old_column = x_dataframe[:, feature_index] #(80x1)
            new_columns = self.power_up_feature(old_column, powerdict[selected_features[feature_index]]) #todo: does this order match/dictionary doesn't keep order..
            polynomial_features = np.hstack((polynomial_features, new_columns))
        return x_dataframe, polynomial_features



"""    def create_power_dict(self):
        power_dict= dict()
        if self.int_power :
            for f in self.preprocessed_data.Model_Structural_Embodied_CO2.x_features_int:
                power_dict[f] = list(range(1, self.int_power+1))
        if self.str_power :
            for f in self.preprocessed_data.Model_Structural_Embodied_CO2.x_features_str:
                power_dict[f] = list(range(1, self.str_power+1))
        return power_dict



    def virtual_features(self):
        dictio, selected_features, selected_powers = self.power_dictionary()
        return np.power(selected_features, selected_powers).reshape([-1, 1])

    def dataframe(self):
        dictio, selected_features, selected_powers = self.power_dictionary()
        dim_o = len(selected_features)
        df_ones = np.ones((dim_o, 1))
        df_zeros = np.zeros((dim_o, max_power-1))

        df_base = np.concatenate(df_ones, df_zeros)
        for feature_index in range(len()):    #(X.shape[1]):


        X = self.dataframe_from_feature(self.Model_Structural_Embodied_CO2.x_features_int,scale=True), xlabels, powerdict)
        power_dict = self.create_power_dict()
        dataframe =
        polynomial_features = np.ones((X.shape[0], 1))
        for feature_index in range(X.shape[1]):
            new_columns = self.power_up_feature(X[:, feature_index], powerdict[xlabels[feature_index]])
            polynomial_features = np.hstack((polynomial_features, new_columns))
        return polynomial_features


    def create_polynomial_features(self):
        X=x_qtt_df
        xlabels=
        powerdict=

        X = self.dataframe_from_feature(self.Model_Structural_Embodied_CO2.x_features_int,scale=True), xlabels, powerdict)
        power_dict = self.create_power_dict()
        dataframe =
        polynomial_features = np.ones((X.shape[0], 1))
        for feature_index in range(X.shape[1]):
            new_columns = self.power_up_feature(X[:, feature_index], powerdict[xlabels[feature_index]])
            polynomial_features = np.hstack((polynomial_features, new_columns))
        return polynomial_features


    def split_dataset(self, X, y, shuffle=False):
        train_ratio = self.preprocessed_data.Model_Structural_Embodied_CO2.train_ratio
        if shuffle:
            pass  # TODO
        cutoff = int(X.shape[0] * train_ratio)
        return (X[:cutoff, :], y[:cutoff]), (X[cutoff:, :], y[cutoff:])"""