import pandas as pd


def interpolate_both(data, johresziit):
    data_season = data[data['season'] == johresziit]
    data_season = data_season.drop(columns=["season"])
    for spaltenname in data_season.columns:
        mean = data_season[spaltenname].mean()
        for i in list(data_season.index.values):
            if pd.isna(data_season.loc[i, spaltenname]):
                data_season.loc[i, spaltenname] = mean
    y = data_season["price_CHF"].to_numpy()
    data_season = data_season.drop(columns=('price_CHF'))
    X = data_season.to_numpy()
    return X, y

def interpolate_both_no_season(data):
    data = data.drop(columns=["season"])
    for spaltenname in data.columns:
        mean = data[spaltenname].mean()
        for i in list(data.index.values):
            if pd.isna(data.loc[i, spaltenname]):
                data.loc[i, spaltenname] = mean
    y = data["price_CHF"].to_numpy()
    data = data.drop(columns=('price_CHF'))
    X = data.to_numpy()
    return X, y


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


# the two functions interpolate the train or test data by mean and delete rows with missing y_values. The feature season
# is not taken into account.
def interpolate_train_no_season(data):
    data = data.drop(columns=["season"])

    #delete entries with y = NaN
    for i in list(data.index.values):
        if pd.isna(data.loc[i, 'price_CHF']):
            data = data.drop(i)

    #set entries with X = NaA to X = X_mean
    for spaltenname in data.columns:
        mean = data[spaltenname].mean()
        for i in list(data.index.values):
            if pd.isna(data.loc[i, spaltenname]):
                data.loc[i, spaltenname] = mean

    y = data["price_CHF"].to_numpy()
    data = data.drop(columns=('price_CHF'))
    X = data.to_numpy()
    return X, y


#for the test function (only return x)
def interpolate_test_no_season(data):
    data = data.drop(columns=["season"])

    # set entries with X = NaA to X = X_mean
    for spaltenname in data.columns:
        mean = data[spaltenname].mean()
        for i in list(data.index.values):
            if pd.isna(data.loc[i, spaltenname]):
                data.loc[i, spaltenname] = mean

    x = data.to_numpy()
    return x

#----------------------------------------------------------------------------------------------

def interpolate_train_no_season_v2(data):
    data = data.drop(columns=["season"])

    #delete entries with y = NaN
    for i in list(data.index.values):
        if pd.isna(data.loc[i, 'price_CHF']):
            data = data.drop(i)

    #set entries with X = NaA to X = X_mean
    for index, row in data.iterrows():
        mean = row.mean()
        data.loc[index] = row.fillna(mean)

    y = data["price_CHF"].to_numpy()
    data = data.drop(columns=('price_CHF'))
    X = data.to_numpy()
    return X, y


#for the test function (only return x)
def interpolate_test_no_season_v2(data):
    data = data.drop(columns=["season"])

    # set entries with X = NaA to X = X_mean
    for index, row in data.iterrows():
        mean = row.mean()
        data.loc[index] = row.fillna(mean)

    x = data.to_numpy()
    return x




datamata = pd.read_csv("train.csv")
testimesti = pd.read_csv("test.csv")

ix = interpolate_test_no_season(datamata)



