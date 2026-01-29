import pandas as pd
df = pd.read_csv("UCI dataset/garments_worker_productivity.csv")
#print(df.head())

print(df.shape)
print(df.columns)

print(df.isnull().sum())
print(df.describe())



