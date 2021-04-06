import pandas as pd

df = pd.read_csv('210406_cs_pm_co2/xdata_gifa_storey_span_load.csv')
df.replace(to_replace={',':'.'}, regex=True, inplace=True)

display(df)