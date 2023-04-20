# Program for predicting sales prices and practice feature engineering, RFs,
# and gradient boosting based on the Kaggle competition :
# House Prices: Advanced Regression Techniques. Link:
# (https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
#
#

###### Start of import packages ######
import sys
import matplotlib as mpl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from statistics import mean, stdev
from sklearn import linear_model
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn import ensemble
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from matplotlib import cm as cm
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
import xgboost as xgb


pd.set_option('display.max_rows', 100)
###### To ignore warnings ######
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

###### End of import packages ######

sns.set()
# Drops the features that consist of mostly Null values and fills in remaining
# Null values with either the mean of the feature if it's numeric or the most
# popular value if it's string.


def removeNulls(dataset):
    dataset.drop(['LotFrontage', 'Alley', 'FireplaceQu', 'PoolQC',
                 'Fence', 'MiscFeature'], axis=1, inplace=True)
    for x in range(0, dataset.shape[1]):
        col = dataset.columns[x]
        if (dataset[col].dtype == object):
            dataset[col] = dataset[col].fillna(dataset[col].mode()[0])
        else:
            dataset[col] = dataset[col].fillna(dataset[col].mean())
    return dataset

# Transform all string features to numeric type


def transformObjects(dataset):
    for x in range(0, dataset.shape[1]):
        if (dataset[dataset.columns[x]].dtype == object):
            le = preprocessing.LabelEncoder()
            le.fit(dataset[dataset.columns[x]])
            dataset[dataset.columns[x]] = le.transform(
                dataset[dataset.columns[x]])
    return dataset


def removeAllOutliers(dataset):
    dataset = dataset[(dataset['LotArea'] < 100000)]
    dataset = dataset[(dataset['BsmtFinSF1'] < 4000)]
    dataset = dataset[(dataset['TotalBsmtSF'] < 5000)]
    dataset = dataset[(dataset['GrLivArea'] < 4600)]
    return dataset

# Prints R2 and RMSE scores


def get_score(prediction, labels):
    print('\tR2: {}'.format(r2_score(prediction, labels)))
    print('\tRMSE: {}'.format(np.sqrt(mean_squared_error(prediction, labels))))


def chunks(l, n):
    return [l[i:i + n] for i in range(0, len(l), n)]


def plot_features_to_SalePrice(dataset):
    cols = dataset.columns
    for x in chunks(cols, 5):
        sns.pairplot(dataset, y_vars=['SalePrice'], x_vars=x)


def correlation(dataset):
    corr = dataset.corr()
    cols = dataset.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            attr1 = cols[i]
            attr2 = cols[j]
            if corr[attr1][attr2] > 0.7:
                print("(%s, %s): %.2f" % (attr1, attr2, corr[attr1][attr2]))


def drop_highly_correlated(dataset):
    dataset.drop(['GarageCars', 'TotRmsAbvGrd', '1stFlrSF', 'Exterior2nd',
                 'GarageYrBlt', 'MSSubClass'], axis=1, inplace=True)
    return dataset


def scale_data(dataset):
    dataset.BsmtFinSF1 = np.log1p(dataset.BsmtFinSF1)
    dataset.BsmtUnfSF = np.log1p(dataset.BsmtUnfSF)
    dataset.TotalBsmtSF = np.log1p(dataset.TotalBsmtSF)
    dataset.GarageArea = np.log1p(dataset.GarageArea)
    dataset.GrLivArea = np.log1p(dataset.GrLivArea)
    dataset.LotArea = np.log1p(dataset.LotArea)
    dataset.MasVnrArea = np.log1p(dataset.MasVnrArea)
    dataset.OpenPorchSF = np.log1p(dataset.OpenPorchSF)
    dataset.EnclosedPorch = np.log1p(dataset.EnclosedPorch)
    dataset.WoodDeckSF = np.log1p(dataset.WoodDeckSF)
    return dataset


def drop_lowly_correlated(dataset):
    drops = ["Street", "LandContour", "Utilities", "LotConfig", "LandSlope",
             "Condition2", "MasVnrType", "BsmtCond", "BsmtFinType2", "BsmtFinSF2",
             "LowQualFinSF", "BsmtHalfBath", "3SsnPorch", "MiscVal", "MoSold",
             "YrSold", "SaleType"]
    dataset = dataset.drop(drops, axis=1)
    return dataset


def drop_not_important(dataset):
    drops = ["RoofMatl", "Heating", "CentralAir", "Electrical", "Functional",
             "GarageQual", "GarageCond", "PavedDrive"]
    dataset = dataset.drop(drops, axis=1)
    return dataset


dataset = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
# correlation(dataset)

dataset = removeNulls(dataset)
dataset = drop_highly_correlated(dataset)
dataset = drop_lowly_correlated(dataset)
dataset = removeAllOutliers(dataset)
dataset = drop_not_important(dataset)
dataset = transformObjects(dataset)

dataset.SalePrice = np.log1p(dataset.SalePrice)
plot_features_to_SalePrice(dataset)
dataset = scale_data(dataset)

test = removeNulls(test)
test = drop_highly_correlated(test)
test = drop_lowly_correlated(test)
test = drop_not_important(test)
test = transformObjects(test)
test = scale_data(test)

print(dataset.shape)

array = dataset.values
testarray = test.values
n = dataset.shape[1]
n = n-1

test_X = testarray[:, 0:n]  # features
X = array[:, 0:n]  # features
Y = array[:, n]  # target

x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.5, random_state=200)  # test = 40%, train = 60%





print("\nElasticNet")
ens_test = linear_model.ElasticNetCV(alphas=[0.0001, 0.0005, 0.001, 0.01, 0.1, 1, 10], l1_ratio=[
                                     0.01, 0.1, 0.5, 0.9, 0.99], max_iter=5000).fit(x_train, y_train)
ens_result = ens_test.predict(x_test)
get_score(ens_result, y_test)
scores = cross_val_score(ens_test, X, Y, cv=5)
print("\tAccuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))



print("\nXgBoost")
xg_reg = xgb.XGBRegressor(objective='reg:linear', eval_metric='logloss',
                          scoring='neg_mean_squared_error', subsample=0.95, colsample_bytree=0.3, learning_rate=0.04,
                          max_depth=2, alpha=0.1, n_estimators=1000,)
xg_reg.fit(x_train, y_train)
preds = xg_reg.predict(x_test)
scores = cross_val_score(xg_reg, X, Y, cv=5)
rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))
print("\tAccuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

print("\nGradientBoostingRegressor")
g_best = ensemble.GradientBoostingRegressor(n_estimators=1850, random_state=1234, learning_rate=0.02, max_depth=3,
                                            max_features='log2', min_samples_leaf=11, min_samples_split=15, loss='huber').fit(x_train, y_train)
g_best_result = g_best.predict(x_test)
get_score(g_best_result, y_test)
scores = cross_val_score(g_best, X, Y, cv=5)
print("\tAccuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

print("\nLinearRegression")
lr = linear_model.LinearRegression()
lr.fit(x_train, y_train)
lr_result = lr.predict(x_test)
get_score(lr_result, y_test)
scores = cross_val_score(lr, X, Y, cv=5)
print("\tAccuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

print("\nRidge Regression")
ridge = Ridge(max_iter=50000)
ridge_est = GridSearchCV(
    ridge, param_grid={"alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]})
ridge_est.fit(x_train, y_train)
ridge_result = ridge_est.predict(x_test)
get_score(ridge_result, y_test)
scores = cross_val_score(ridge_est, X, Y, cv=5)
print("\tAccuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# Execution time 2-3 mins
print("\nLasso Regression")
lasso = Lasso(max_iter=50000)
lasso_est = GridSearchCV(
    lasso, param_grid={"alpha": np.arange(0.0005, 0.001, 0.00001)})
lasso_est.fit(x_train, y_train)
lasso_result = lasso_est.predict(x_test)
get_score(lasso_result, y_test)
scores = cross_val_score(lasso_est, X, Y, cv=5)
print("\tAccuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

result = np.exp(g_best.predict(test_X))
output = pd.DataFrame({'Id': test.Id, 'SalePrice': result})
print(output)
output.to_csv('submission.csv', index=False)
