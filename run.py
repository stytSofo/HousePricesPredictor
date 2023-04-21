# %%
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

# %%
dataset = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# %%
dataset.isnull().sum()

# %%
# Remove Nulls
# dataset.dropna(axis=1,inplace=True)
dataset.drop(['LotFrontage', 'Alley', 'FireplaceQu', 'PoolQC',
                 'Fence', 'MiscFeature','GarageType','GarageFinish'], axis=1, inplace=True)
# for x in range(0, dataset.shape[1]):
#     col = dataset.columns[x]
#     if (dataset[col].dtype == object):
#         dataset[col] = dataset[col].fillna(dataset[col].mode()[0])
#     else:
#         dataset[col] = dataset[col].fillna(dataset[col].mean())
for i,column in enumerate(dataset):
        if (dataset[column].dtype == object):
            dataset[column] = dataset[column].fillna(dataset[column].mode()[0])
        else:
            dataset[column] = dataset[column].fillna(dataset[column].mean())

# %%
# Drop Highly Correlated
dataset.drop(['GarageCars', 'TotRmsAbvGrd', '1stFlrSF', 'Exterior2nd',
                 'GarageYrBlt', 'MSSubClass'], axis=1, inplace=True)

# %%
# Drop Low Correlated
drops = ["Street", "LandContour", "Utilities", "LotConfig", "LandSlope",
             "Condition2", "MasVnrType", "BsmtCond", "BsmtFinType2", "BsmtFinSF2",
             "BsmtHalfBath", "LowQualFinSF", "3SsnPorch", "MiscVal", "MoSold",
             "YrSold"]
dataset = dataset.drop(drops, axis=1)

# %%
# Drop outliers
dataset = dataset[(dataset['LotArea'] < 60000)]
dataset = dataset[(dataset['BsmtFinSF1'] < 2300)]
dataset = dataset[(dataset['TotalBsmtSF'] < 5000)]
dataset = dataset[(dataset['GrLivArea'] < 4600)]
dataset = dataset[(dataset['OpenPorchSF'] < 380)]

# %%
# Drop Not Important
drops = ["RoofMatl", "Heating", "Electrical", "Functional",
             "GarageQual", "GarageCond", "PavedDrive"]
dataset = dataset.drop(drops, axis=1)


# %%
dataset

# %%
# dataset["SaleType_WD"] = dataset['SaleType'].apply(lambda x: 1 if x=='WD' else 0)
# dataset["SaleType_CWD"] = dataset['SaleType'].apply(lambda x: 1 if x=='CWD' else 0)
# dataset["SaleType_VWD"] = dataset['SaleType'].apply(lambda x: 1 if x=='VWD' else 0)
dataset["SaleType"] = dataset['SaleType'].apply(lambda x: 1 if x=="New" else 0)
# dataset["SaleType_COD"] = dataset['SaleType'].apply(lambda x: 1 if x=='COD' else 0)
# dataset["SaleType_Con"] = dataset['SaleType'].apply(lambda x: 1 if x=='Con' else 0)
# dataset["SaleType_ConLw"] = dataset['SaleType'].apply(lambda x: 1 if x=='ConLw' else 0)
# dataset["SaleType_ConLI"] = dataset['SaleType'].apply(lambda x: 1 if x=='ConLI' else 0)
# dataset["SaleType_ConLD"] = dataset['SaleType'].apply(lambda x: 1 if x=='ConLD' else 0)
# dataset["SaleType_Oth"] = dataset['SaleType'].apply(lambda x: 1 if x=='Oth' else 0)


# %%
# Transform to Object

# for x in range(0, dataset.shape[1]):
#     if (dataset[dataset.columns[x]].dtype == object):
#         le = preprocessing.LabelEncoder()
#         le.fit(dataset[dataset.columns[x]])
#         dataset[dataset.columns[x]] = le.transform(dataset[dataset.columns[x]])
dataset = pd.get_dummies(dataset)

# %%
# Scale Price
# dataset = np.log1p(dataset)
# dataset.SalePrice = np.log1p(dataset.SalePrice)


# %%
# plot_features_to_SalePrice
def chunks(l, n):
    return [l[i:i + n] for i in range(0, len(l), n)]

cols = dataset.columns
for x in chunks(cols, 5):
    sns.pairplot(dataset, y_vars=['SalePrice'], x_vars=x)

# %%
# Scale Data

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

# dataset = np.log1p(dataset)

# %%
pd.set_option('display.max_columns',None)
dataset

# %%
# corr = dataset.corrwith(dataset['GarageArea']).to_frame()
corr = dataset.corr()
plt.figure(figsize=(50,50))
sns.heatmap(corr,cmap='coolwarm',annot=True, fmt='.2f')
plt.show()

# %%
dataset.SalePrice = np.log1p(dataset.SalePrice)

# %%
# Remove Nulls Test
test.drop(['LotFrontage', 'Alley', 'FireplaceQu', 'PoolQC',
                 'Fence', 'MiscFeature','GarageType','GarageFinish'], axis=1, inplace=True)
# for x in range(0, test.shape[1]):
#     col = test.columns[x]
#     if (test[col].dtype == object):
#         test[col] = test[col].fillna(test[col].mode()[0])
#     else:
#         test[col] = test[col].fillna(test[col].mean())

for i,column in enumerate(test):
        if (test[column].dtype == object):
            test[column] = test[column].fillna(test[column].mode()[0])
        else:
            test[column] = test[column].fillna(test[column].mean())

# %%
# Drop Highly Corr Test
test.drop(['GarageCars', 'TotRmsAbvGrd', '1stFlrSF', 'Exterior2nd',
                 'GarageYrBlt', 'MSSubClass'], axis=1, inplace=True)

# %%
# Drop Low Corr Test
drops = ["Street", "LandContour", "Utilities", "LotConfig", "LandSlope",
             "Condition2", "MasVnrType", "BsmtCond", "BsmtFinType2", "BsmtFinSF2",
             "BsmtHalfBath", "LowQualFinSF", "3SsnPorch", "MiscVal", "MoSold",
             "YrSold"]
test = test.drop(drops, axis=1)

# %%
# Not Important Test
drops = ["RoofMatl", "Heating", "Electrical", "Functional",
             "GarageQual", "GarageCond", "PavedDrive"]
test = test.drop(drops, axis=1)

# %%
# dataset["SaleType_WD"] = dataset['SaleType'].apply(lambda x: 1 if x=='WD' else 0)
# dataset["SaleType_CWD"] = dataset['SaleType'].apply(lambda x: 1 if x=='CWD' else 0)
# dataset["SaleType_VWD"] = dataset['SaleType'].apply(lambda x: 1 if x=='VWD' else 0)
test["SaleType"] = test['SaleType'].apply(lambda x: 1 if x=='New' else 0)
# dataset["SaleType_COD"] = dataset['SaleType'].apply(lambda x: 1 if x=='COD' else 0)
# dataset["SaleType_Con"] = dataset['SaleType'].apply(lambda x: 1 if x=='Con' else 0)
# dataset["SaleType_ConLw"] = dataset['SaleType'].apply(lambda x: 1 if x=='ConLw' else 0)
# dataset["SaleType_ConLI"] = dataset['SaleType'].apply(lambda x: 1 if x=='ConLI' else 0)
# dataset["SaleType_ConLD"] = dataset['SaleType'].apply(lambda x: 1 if x=='ConLD' else 0)
# dataset["SaleType_Oth"] = dataset['SaleType'].apply(lambda x: 1 if x=='Oth' else 0)


# %%
# Transform To Object Test
# for x in range(0, test.shape[1]):
#         if (test[test.columns[x]].dtype == object):
#             le = preprocessing.LabelEncoder()
#             le.fit(test[test.columns[x]])
#             test[test.columns[x]] = le.transform(
#                 test[test.columns[x]])
test = pd.get_dummies(test)

# %%
# Match columns train test
cols = dataset.columns.tolist()
test = test.reindex(columns=cols).fillna(0)

# %%
# Scale Data
# dataset = np.log1p(dataset)


test.BsmtFinSF1 = np.log1p(test.BsmtFinSF1)
test.BsmtUnfSF = np.log1p(test.BsmtUnfSF)
test.TotalBsmtSF = np.log1p(test.TotalBsmtSF)
test.GarageArea = np.log1p(test.GarageArea)
test.GrLivArea = np.log1p(test.GrLivArea)
test.LotArea = np.log1p(test.LotArea)
test.MasVnrArea = np.log1p(test.MasVnrArea)
test.OpenPorchSF = np.log1p(test.OpenPorchSF)
test.EnclosedPorch = np.log1p(test.EnclosedPorch)
test.WoodDeckSF = np.log1p(test.WoodDeckSF)

# test = np.log1p(test)

# %%
# dataset["Cond1_Artery"] = dataset['Condition1'].apply(lambda x: 1 if x=='Artery' else 0)
# dataset["Cond1_Feedr"] = dataset['Condition1'].apply(lambda x: 1 if x== 'Feedr' else 0)
# dataset["Cond1_Norm"] = dataset['Condition1'].apply(lambda x: 1 if x=='Norm' else 0)
# dataset["Cond1_RRNn"] = dataset['Condition1'].apply(lambda x: 1 if x=='RRNn' else 0)
# dataset["Cond1_RRAn"] = dataset['Condition1'].apply(lambda x: 1 if x=='RRAn' else 0)
# dataset["Cond1_PosN"] = dataset['Condition1'].apply(lambda x: 1 if x=='PosN' else 0)
# dataset["Cond1_PosA"] = dataset['Condition1'].apply(lambda x: 1 if x=='PosA' else 0)
# dataset["Cond1_RRNe"] = dataset['Condition1'].apply(lambda x: 1 if x=='RRNe' else 0)
# dataset["Cond1_RRAe"] = dataset['Condition1'].apply(lambda x: 1 if x=='RRAe' else 0)


# %%
print(dataset.shape)

# %%
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer

array = dataset.values
testarray = test.values
n = dataset.shape[1]
n = n-1

test_X = testarray[:, 0:n]  # features
X = array[:, 0:n]  # features
Y = array[:, n]  # target

XX = dataset.iloc[:, ~dataset.columns.isin(['Id','SalePrice'])]
YY = dataset[['SalePrice']]

def mrse(y_true,y_pred):
    return 1 - np.sqrt(mean_squared_error(y_true,y_pred))

scorer = make_scorer(mrse,greater_is_better=True)


rfr = RandomForestRegressor()
sfs_range = SFS(estimator=rfr,
                k_features=(16,18),
                forward=True,
                floating=False,
                scoring=scorer,
                cv=5,n_jobs=8)

sfs_range.fit(XX,YY)

# print the accuracy of the best combination as well as the set of best features
print('best combination (ACC: %.3f): %s\n' % (sfs_range.k_score_, sfs_range.k_feature_idx_))

plt.rcParams["figure.figsize"] = (6,6)
# use the plot_sfs to visualize all accuracies
plot_sfs(sfs_range.get_metric_dict(), kind='std_err')



# %%
X_sfs = sfs_range.transform(XX)
x_train, x_test, y_train, y_test = train_test_split(X_sfs, YY, test_size=0.2, random_state=203)  # test = 40%, train = 60%

# %%
print("\nElasticNet")
ens_test = linear_model.ElasticNetCV(alphas=[0.0001, 0.0005, 0.001, 0.0015, 0.01, 0.015, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], 
                                     l1_ratio=[0.01, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99], max_iter=10000).fit(x_train, y_train)
ens_result = ens_test.predict(x_test)
print('\tR2: {}'.format(r2_score(ens_result, y_test)))
print('\tRMSE: {}'.format(np.sqrt(mean_squared_error(ens_result, y_test))))
scores = cross_val_score(ens_test, XX, YY, cv=5)
print("\tAccuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# %%
print("\nXgBoost")
xg_reg = xgb.XGBRegressor(objective='reg:linear', eval_metric='logloss',
                          scoring='neg_mean_squared_error', subsample=0.95, colsample_bytree=0.3, learning_rate=0.04,
                          max_depth=2, alpha=0.1, n_estimators=1000,)
xg_reg.fit(x_train, y_train)
preds = xg_reg.predict(x_test)
scores = cross_val_score(xg_reg, XX, YY, cv=5)
rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))
print("\tAccuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# %%
print("\nGradientBoostingRegressor")
g_best = ensemble.GradientBoostingRegressor(n_estimators=1850, random_state=1234, learning_rate=0.02, max_depth=3,
                                            max_features='log2', min_samples_leaf=11, min_samples_split=15, loss='huber').fit(x_train, y_train)
g_best_result = g_best.predict(x_test)
print('\tR2: {}'.format(r2_score(g_best_result, y_test)))
print('\tRMSE: {}'.format(np.sqrt(mean_squared_error(g_best_result, y_test))))
scores = cross_val_score(g_best, XX, YY, cv=5)
print("\tAccuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# %%
print("\nLinearRegression")
lr = linear_model.LinearRegression()
lr.fit(x_train, y_train)
lr_result = lr.predict(x_test)
print('\tR2: {}'.format(r2_score(lr_result, y_test)))
print('\tRMSE: {}'.format(np.sqrt(mean_squared_error(lr_result, y_test))))
scores = cross_val_score(lr, XX, YY, cv=5)
print("\tAccuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# %%
print("\nRidge Regression")
ridge = Ridge(max_iter=50000)
ridge_est = GridSearchCV(
    ridge, param_grid={"alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]})
ridge_est.fit(x_train, y_train)
ridge_result = ridge_est.predict(x_test)
print('\tR2: {}'.format(r2_score(ridge_result, y_test)))
print('\tRMSE: {}'.format(np.sqrt(mean_squared_error(ridge_result, y_test))))
scores = cross_val_score(ridge_est, XX, YY, cv=5)
print("\tAccuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# %%
print("\nLasso Regression")
lasso = Lasso(max_iter=50000)
lasso_est = GridSearchCV(
    lasso, param_grid={"alpha": np.arange(0.0005, 0.001, 0.00001)})
lasso_est.fit(x_train, y_train)
lasso_result = lasso_est.predict(x_test)
print('\tR2: {}'.format(r2_score(lasso_result, y_test)))
print('\tRMSE: {}'.format(np.sqrt(mean_squared_error(lasso_result, y_test))))
scores = cross_val_score(lasso_est, XX, YY, cv=5)
print("\tAccuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# %%
result = np.exp(ens_test.predict(test.iloc[:,[1,28]]))
output = pd.DataFrame({'Id': test.Id, 'SalePrice': result})
print(output)
output.to_csv('submission.csv', index=False)


