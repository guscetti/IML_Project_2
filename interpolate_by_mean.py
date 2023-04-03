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
        if pd.isna(data_season.loc[i, 'price_CHF']):
            data_season = data_season.drop(i)

    #set entries with X = NaA to X = X_mean
    for spaltenname in data_season.columns:
        mean = data_season[spaltenname].mean()
        for i in list(data_season.index.values):
            if pd.isna(data_season.loc[i, spaltenname]):
                data_season.loc[i, spaltenname] = mean
    return data_season


#split the interpolated data frames into numpy array of X and y
def x_to_np(data, season, interp):
    if interp == 'only_x':
        prices_season = interpolate_x(data, season)
    elif interp == 'x_and_y':
        prices_season = interpolate_x_and_y(data, season)
    data = prices_season.drop(columns=('price_CHF'))
    X = data.to_numpy()
    return X

def y_to_np(data, season, interp):
    if interp == 'only_x':
        prices_season = interpolate_x(data, season)
    elif interp == 'x_and_y':
        prices_season = interpolate_x_and_y(data, season)
    y = prices_season["price_CHF"].to_numpy()
    return y

