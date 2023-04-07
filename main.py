# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, RBF, Matern, RationalQuadratic
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


# the two functions interpolate the train or test data by mean and delete rows with missing y_values. The feature season
# is not taken into account.
def interpolate_train_no_season(train):
    train = train.drop(columns=["season"])

    # delete entries with y = NaN
    for i in list(train.index.values):
        if pd.isna(train.loc[i, 'price_CHF']):
            train = train.drop(i)

    # stuff to np
    y = train["price_CHF"].to_numpy()
    train = train.drop(columns=('price_CHF'))
    X = train.to_numpy()

    # insert column mean using simple inputer
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(X)
    X_interp = imp.transform(X)

    return X_interp, y


# for the test function (only return x)
def interpolate_test_no_season(train):
    train = train.drop(columns=["season"])

    # stuff to np
    X = train.to_numpy()

    # insert column mean using simple inputer
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(X)
    X_interp = imp.transform(X)

    return X_interp
#-------------------------------------------------------------------------------------------------------------
#multivariant feature inputation

def multi_variant_train(train):
    train = train.drop(columns=["season"])

    # delete entries with y = NaN
    for i in list(train.index.values):
        if pd.isna(train.loc[i, 'price_CHF']):
            train = train.drop(i)

    # stuff to np
    y = train["price_CHF"].to_numpy()
    train = train.drop(columns=('price_CHF'))
    X = train.to_numpy()

    # insert column mean using a multivariant inputer
    imp = IterativeImputer(max_iter=100, random_state=0)
    imp.fit(X)
    IterativeImputer(random_state=0)
    X_interp = imp.transform(X)

    return X_interp, y

def multi_variant_test(train):
    train = train.drop(columns=["season"])

    # stuff to np
    X = train.to_numpy()

    # insert column mean using a multivariant inputer
    imp = IterativeImputer(max_iter=100, random_state=0)
    imp.fit(X)
    IterativeImputer(random_state=0)
    X_interp = imp.transform(X)

    return X_interp

# ---------------------------------------------------------------------------------------------------------------------------------
def data_loading():
    """
    This function loads the training and test data, preprocesses it, removes the NaN values and interpolates the missing
    data using imputation

    Parameters
    ----------
    Returns
    ----------
    X_train: matrix of floats, training input with features
    y_train: array of floats, training output with labels
    X_test: matrix of floats: dim = (100, ?), test input with features
    """
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    #X_tr, y_tr = interpolate_train_no_season(train)
    #X_tst = interpolate_test_no_season(test)
    X_tr, y_tr = multi_variant_train(train)
    X_tst = multi_variant_test(test)


    assert (X_tr.shape[1] == X_tst.shape[1]) and (X_tr.shape[0] == y_tr.shape[0]) and (
            X_tst.shape[0] == 100), "Invalid data shape"
    return X_tr, y_tr, X_tst


def modeling_and_prediction(X_train, y_train, X_test, Kernel):
    """
    This function defines the model, fits training data and then does the prediction with the test data

    Parameters
    ----------
    X_train: matrix of floats, training input with 10 features
    y_train: array of floats, training output
    X_test: matrix of floats: dim = (100, ?), test input with 10 features

    Returns
    ----------
    y_test: array of floats: dim = (100,), predictions on test set
    """
    # y_pred = np.zeros(X_test.shape[0])
    # TODO: Define the model and fit it using training data. Then, use test data to make predictions

    gpr = GaussianProcessRegressor(kernel=Kernel, alpha=0.0001)
    gpr.fit(X_train, y_train)
    y_pred = gpr.predict(X_test)
    return np.array(y_pred)


def calculate_RMSE(y_predicted, y):
    RMSE = np.sqrt(sum((y_predicted - y) ** 2) / len(y))
    assert np.isscalar(RMSE)
    return RMSE


def average_RMSE(X, y, Kernels, n_folds):
    """
    Main cross-validation loop, implementing 10-fold CV. In every iteration (for every train-test split), the RMSE for every lambda is calculated,
    and then averaged over iterations.

    Parameters
    ----------
    X: matrix of floats, dim = (150, 13), inputs with 13 features
    y: array of floats, dim = (150, ), input labels
    Kernels: list of floats, len = 5, values of lambda for which ridge regression is fitted and RMSE estimated
    n_folds: int, number of foflds (pieces in which we split the dataset), parameter K in KFold CV

    Returns
    ----------
    avg_RMSE: array of floats: dim = (5,), average RMSE value for every lambda
    """
    RMSE_mat = np.zeros((n_folds, len(Kernels)))
    for j in range(len(Kernels)):
        kf = KFold(n_splits=n_folds, shuffle=True)
        for i, (train_index, test_index) in enumerate(kf.split(X)):
            y_hat = modeling_and_prediction(X[train_index], y[train_index], X[test_index], Kernels[j])
            RMSE_mat[i, j] = (calculate_RMSE(y_hat, y[test_index]))

    avg_RMSE = np.mean(RMSE_mat, axis=0)
    return avg_RMSE


# Main function. You don't have to change this
if __name__ == "__main__":
    # Data loading
    X_train, y_train, X_test = data_loading()

    # cross validation for the kernels
    """
    körnels = [DotProduct(), RBF(), Matern(), RationalQuadratic()]
    n_folds = 10
    cross_val = average_RMSE(X_train, y_train, körnels, n_folds)
    print(cross_val)
    """

    # predict y
    y_pred = modeling_and_prediction(X_train, y_train, X_test, RationalQuadratic())

    # Save results in the required format
    dt = pd.DataFrame(y_pred)
    dt.columns = ['price_CHF']
    dt.to_csv('results.csv', index=False)
    print("\nResults file successfully generated!")
