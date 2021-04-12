import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
import seaborn as sns
import folium

### this function load data from .dat file
def load_dat(filename):
    with open(filename, 'r') as fin:
        lines = fin.readlines()
        dim = len(lines[0].strip().split(";"))
        num_samples = len(lines)
        data = np.zeros((num_samples, dim))
        for i in range(num_samples):
            data[i, :] = np.array([float(x.replace(",",".")) for x in lines[i].strip().split(";")])
        return data

### load data
# call the load_dat function to load X and Y from the corresponding input files
X = load_dat("210406_cs_pm_co2/xdata_gifa_storey_span_load.csv")
y = load_dat("210406_cs_pm_co2/ydata_totCO2.csv")
# get some statistics of the data
num_samples = X.shape[0] # get the first dimension of X (i.e. number of rows)
dim = X.shape[1] # get the second dimension of X (i.e. number of columns)
print('X (%d x %d)' %(num_samples, dim))
print('y (%d)' %(num_samples))

###make dictionary and plot data
CO2_dict = {}
CO2_dict['GIFA']=X[:,0]
CO2_dict['STOREY']=X[:,1]
CO2_dict['SPAN']=X[:,2]
CO2_dict['LOAD']=X[:,3]
CO2_dict['CO2eq']=y[:,0]

df = pd.DataFrame(CO2_dict, columns=['GIFA', 'STOREY', 'SPAN', 'LOAD', 'CO2eq'])
display(df)

sns.scatterplot(data=df, x="GIFA", y="CO2eq", hue="CO2eq")
plt.ylim(0,5000)



"""
df1=df
df1.plot.scatter(x="GIFA", y="CO2eq")
pd.show()
#display(df1)
#df.plot()
#sns.scatterplot(data=df, x="building", y="GHG/Capita", hue="City category")
#plt.ylim(0,30000)


#pd.plotting.table(ax,CO2_dict)
#table_values = X+y
#table_labels = ['GIFA', 'STOREY', 'SPAN', 'LOAD', 'CO2Eq']
#import matplotlib.pyplot as plt
#plt.table(table_values, cellColours=None, cellLoc='right', colWidths=None, rowLabels=None, rowColours=None, rowLoc='left', colLabels=table_labels, colColours=None, colLoc='center', loc='bottom', bbox=None, edges='closed')
#plt.show()

#print(CO2_dict.values())
#print(type(CO2_dict.values))
#print(CO2_dict['GIFA'])
#print(len(CO2_dict['GIFA']))
"""