import pandas as pd
from pandas import DataFrame
import sys

filename = sys.argv[1]
df = pd.read_csv(filename)
del df["YEAR"]
del df["RECIPIENT ZIPCODE"]

# Change the categorical feature to 0-1 s
df2 = pd.get_dummies(df)
df2 = df2.astype(int)

# Move the label to the last column
cols = list(df2.columns.values)
cols.remove("TREATMENT")
cols.append("TREATMENT")
df2 = df2[cols]

df2.to_csv("data_processed/to_dummies.csv")



