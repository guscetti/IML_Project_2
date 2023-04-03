import numpy as np
import pandas as pd



data_mata = pd.read_csv("train.csv")

def interpolate_x_and_y(data, johresziit):
    data_season = data[data['season'] == johresziit]
    data_season = data_season.drop(columns=["season"])
    for spaltenname in data_season.columns:
        mean = data_season[spaltenname].mean()
        for i in list(data_season.index.values):
            if pd.isna(data_season.loc[i, spaltenname]):
                data_season.loc[i, spaltenname] = mean
    return data_season


def interpolate_x(data, johresziit):
    data_season = data[data['season'] == johresziit]
    data_season = data_season.drop(columns=["season"])

    #delete entries with y = NaN
    for i in list(data_season.index.values):
        print(i)
        print(data_season.loc[i, 'price_CHF'])
        #if pd.isna(data_season.loc[i, 'price_CHF']):
            #data_season.drop([i])

    return 0


prices_spring_x_y = interpolate_x_and_y(data_mata, 'spring')
prices_spring_x = interpolate_x(data_mata, 'spring')
#print(prices_spring_x)