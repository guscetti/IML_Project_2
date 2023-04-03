import numpy as np
import pandas as pd

data = pd.read_csv("train.csv")
data_season = data.drop(columns=["season"])

# create a new DataFrame with the first 10 rows dropped
data_season = data_season.drop(range(10))

print(data_season)
