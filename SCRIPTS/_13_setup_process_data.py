import numpy as np

class data_processing():

    def __init__(self, preprocessed_data):
        self.preprocessed_data = preprocessed_data
        self.power_dictionary = self.construct_power_dictionary()[0]
        self.selected_features = self.construct_power_dictionary()[1]
        self.selected_powers = self.construct_power_dictionary()[2]
        self.input_df = self.preprocessed_data.dataframe_from_feature(self.selected_features,
                                                    self.preprocessed_data.Model_Structural_Embodied_CO2.f_scaling)
        self.output_df = self.preprocessed_data.y_df
        #self.output_df = self.preprocessed_data.dataframe_from_feature(
        #    self.preprocessed_data.Model_Structural_Embodied_CO2.y_features,
        #    self.preprocessed_data.Model_Structural_Embodied_CO2.f_scaling)
        (self.x_tr, self.y_tr), (self.x_te, self.y_te) = self.split_training_testing_data()

    def longest(self, list):
        return max(len(elem) for elem in list)

    def construct_power_dictionary(self):
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

        for feature, power_list in zip(all_features, all_powers):
            if power_list:
                dictio[feature] = power_list
                selected_features.append(feature)
                selected_powers.append(power_list)

        return [dictio, selected_features, selected_powers] #todo: do dictionaries keep orders?if so keeping all this is useless


    def power_up_feature(self, featureArray_column, powerList):
        list_of_virtual_features = []
        for power in powerList: #(1,2,3)
            list_of_virtual_features.append(np.power(featureArray_column, power).reshape([-1, 1]))

        return np.hstack(list_of_virtual_features)


    def create_polynomial_features(self):

        polynomial_features = np.ones((self.input_df.shape[0], 1))
        for feature, feature_index in zip(self.selected_features, list(range(len(self.selected_features)))):
            old_column = self.input_df[:, feature_index] #(80x1)
            new_columns = self.power_up_feature(old_column, self.power_dictionary[feature]) #todo: does this order match/dictionary doesn't keep order..
            polynomial_features = np.hstack((polynomial_features, new_columns))
        return polynomial_features

    def split_training_testing_data(self,shuffle=False):
        X = self.create_polynomial_features() #TODO : check split before or after power up to polynomial data?
        y = self.output_df
        train_ratio = self.preprocessed_data.Model_Structural_Embodied_CO2.train_ratio
        if shuffle:
            pass  # TODO
        cutoff = int(X.shape[0] * train_ratio)
        return (X[:cutoff, :], y[:cutoff]), (X[cutoff:, :], y[cutoff:])

    def cross_validation(self):
        pass # TODO : split data in 5 groups, rotate between groups when training