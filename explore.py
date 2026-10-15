import pandas as pd
d=pd.read_csv("dataset/tourism.csv")
print(d.head(),"\n");d.info();print(d.describe());print("Duplicates:",d.duplicated().sum());print(d.isna().sum())
