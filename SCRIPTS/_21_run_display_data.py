import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from _12_setup_preprocess_data import *


class data_display():

    def __init__(self, preprocessed_data):

        #super().__init__() #TODO: ?
        self.preprocessed_data = preprocessed_data

    def view_dataframe_from_dict(self, mydict, disp=False):
        import pandas as pd
        from IPython.display import display
        df = pd.DataFrame(mydict, columns=mydict.keys())
        if disp:
            display(df)
        return df

    def view_dataframe_from_dict_keys(self, mydict, keys, disp=False):
        import pandas as pd
        from IPython.display import display
        df = pd.DataFrame(mydict, columns=keys)
        if disp:
            display(df)
        return df

    def plot_graph(self, dataframe, x_label, y_label,
                   title="Features influencing CO2 footprint of Structures - Datasource : Price & Myers",
                   figure_size=(12, 15), folder=None, plot=False):
        fig, ax = plt.subplots(figsize=figure_size)
        ax.set_title(title)
        sns.scatterplot(data=dataframe, x=x_label, y=y_label, hue=y_label)
        if folder:
            plt.savefig(folder + '/' + x_label + '-' + y_label + '.png')
        if plot:
            plt.show() #TODO: delete this?

    def plot_graph_adv(self, x_label, y_label,
                       title="Features influencing CO2 footprint of Structures - Datasource : Price & Myers",
                       reference="", figure_size=(12, 15), save=False, show=True):
        import os
        dataframe = self.view_dataframe_from_dict(self.preprocessed_data.dictionary_of_data())
        label_dict = self.preprocessed_data.dictionary_of_labels()
        labels = label_dict[x_label]
        fig, ax = plt.subplots(figsize=figure_size)
        ax.set_title(title + " " + reference)
        if x_label in self.preprocessed_data.Model_Structural_Embodied_CO2.x_features_str:
            x = np.arange(len(labels))
            ax.set_ylabel(y_label)
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            plt.setp(ax.get_xticklabels(), rotation=25, ha="right",
                     rotation_mode="anchor")
        sns.scatterplot(data=dataframe, x=x_label, y=y_label, hue=y_label, ax=ax)

        if save and not os.path.isdir(self.preprocessed_data.Model_Structural_Embodied_CO2.output_path):
            os.makedirs(self.preprocessed_data.Model_Structural_Embodied_CO2.output_path)
            plt.savefig(self.preprocessed_data.Model_Structural_Embodied_CO2.output_path + '/' + x_label + '-' + y_label + '.png')

        if show:
            plt.show()


