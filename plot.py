import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def process(data):
    data = data.drop(columns=["season"])

    #delete entries with y = NaN
    for i in list(data.index.values):
        if pd.isna(data.loc[i, 'price_CHF']):
            data = data.drop(i)

    #set entries with X = NaA to X = X_mean
    iX = []
    for index, row in data.iterrows():
        mean = row.mean()
        #data.loc[index] = row.fillna(mean)
        iX.append(mean)

    y = data["price_CHF"].to_numpy()
    data = data.drop(columns=('price_CHF'))
    X = data.to_numpy()
    return iX, y


def interpolate_drop_empty_y(data, johresziit):
    data_season = data[data['season'] == johresziit]
    data_season = data_season.drop(columns=["season"])

    #delete entries with y = NaN
    for i in list(data_season.index.values):
        if pd.isna(data_season.loc[i, 'price_CHF']):
            data_season = data_season.drop(i)

    #set entries with X = NaA to X = X_mean
    for spaltenname in data_season.columns:
        mean = data_season[spaltenname].mean()
        for i in list(data_season.index.values):
            if pd.isna(data_season.loc[i, spaltenname]):
                data_season.loc[i, spaltenname] = mean

    y = data_season["price_CHF"].to_numpy()
    data_season = data.drop(columns=('price_CHF'))
    X = data_season.to_numpy()
    return X, y


datamata = pd.read_csv("train.csv")
testimesti = pd.read_csv("test.csv")


ex, wy = process(datamata)

zäme = zip(ex, wy)
zäme_sort = sorted(zäme, key=lambda x: x[1])
res_x, res_y = [[i for i, j in zäme_sort],
       [j for i, j in zäme_sort]]

#dt = pd.DataFrame(ex)
#dt.to_csv('shit.csv', index=False)

plt.plot(res_y, res_x)
plt.show()