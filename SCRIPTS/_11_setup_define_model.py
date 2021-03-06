

class Model_Structural_Embodied_CO2:
    def __init__(self, date="210910", test_count='1', archive=True, display_plots=True,
                 display_features=True, logit = True,
                 input_path="C:/Users/sfenton/Code/Repositories/CO2footprint/DATA/210413_PM_CO2_data",
                 output_path='C:/Users/sfenton/Code/Repositories/CO2footprint/RESULTS/',
                 x_features_str=['Sector', 'Type', 'Basement', 'Foundations', 'Ground Floor', 'Superstructure', 'Cladding', 'BREEAM Rating'],
                 x_features_int=['GIFA (m2)', 'Storeys', 'Typical Span (m)', 'Typ Qk (kN_per_m2)'],
                 y_features=['Calculated Total tCO2e', 'Calculated tCO2e_per_m2'],
                 tCO2e_per_m2 = 1, train_ratio = 0.8, reg=1, f_scaling=False, GIFA_power=[1, 2, 3],
                 Storeys_power=[1, 2], Span_power=[1, 2], Qk_power = [1, 2, 3], Sector_power=[1],
                 Type_power=[1, 2], Basement_power=[1], Foundations_power=[1], Groundfloor_power=[1],
                 Superstructure_power=[1], Cladding_power=[1], Rating_power=[1]):

        """
        1. DISPLAY PARAMETERS
        """
        #to update
        self.date = date
        self.test_count = test_count
        self.archive = archive #Save_visu
        self.display_features = display_features
        self.display_plots = display_plots

        #Default
        self.reference = date +'_results_' + test_count
        self.input_path = input_path
        self.output_path = output_path + date + '_results/'
        self.x_features_str = x_features_str  #STR_FEATURES
        self.x_features_int = x_features_int    #INT_FEATURES
        self.x_features = x_features_str + x_features_int   #FEATURES_NAMES
        self.y_features = y_features    #OUTPUT_NAMES

        """
        2. MODEL PARAMETERS
        """
        self.tCO2e_per_m2 = tCO2e_per_m2  # 0 ='Calculated Total tCO2e'; 1 ='Calculated tCO2e_per_m2' #TODO : change this name
        self.train_ratio = train_ratio
        self.reg = reg
        self.f_scaling = f_scaling
        self.logit = logit


        """
        3. POLYNOMIAL PARAMETERS
        """

        self.GIFA_power = GIFA_power
        self.Storeys_power = Storeys_power
        self.Span_power = Span_power
        self.Qk_power = Qk_power
        self.Sector_power = Sector_power
        self.Type_power = Type_power
        self.Basement_power = Basement_power
        self.Foundations_power = Foundations_power
        self.Groundfloor_power = Groundfloor_power
        self.Superstructure_power = Superstructure_power
        self.Cladding_power = Cladding_power
        self.Rating_power = Rating_power


    def export_model_data(self, format = '.txt', title="data", save=True, show=True):
        import os
        if save and not os.path.isdir(self.output_path):
            os.makedirs(self.output_path)
        for k, v in self.__dict__.items():
            if show:
                print(' ', k, ' : ', v)
            if save:
                print(' ', k, ' : ', v, file=open(self.output_path + title + format, 'a'))


