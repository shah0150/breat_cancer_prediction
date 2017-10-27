import numpy as np
import pandas as pd

# Read the CSV file
data = pd.read_csv('data/data.csv', index_col=False)

# id col is redundant and not useful
data.drop('id', axis=1, inplace=True)

print("After deleting the ID " ,data.head(2))

print("Shape of the data ", data.shape)

print("Review data types " , data.info())

print("Review number of columns of each data type " , data.get_dtype_counts())

print("check for missing variables " , data.isnull().any())

print(data.diagnosis.unique())

#save the cleaner version of dataframe for future analyis
data.to_csv('data/clean-data.csv')