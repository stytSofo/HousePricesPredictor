import pandas as pd
from sklearn import ensemble
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn import linear_model
import xgboost as xgb
from sklearn.metrics import make_scorer, r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV
import warnings

warnings.simplefilter("ignore")

kfolds = KFold(n_splits=10, shuffle=True, random_state=42)

# Remove correlated column with score over 0.8
def remove_correlated(df):
    to_drop = []
    col_corr = set()
    corr_matrix = df.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i, j] >= 0.8) and (corr_matrix.columns[j] not in col_corr):
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
                if colname in df.columns:
                    to_drop.append(colname)
    df.drop(to_drop, axis=1, inplace=True)
    return df

# Top 14 columns from feature importance. After testings only 3
# columns give as the best answers.
def outliers(data):
    # data = data[(data['BsmtFinSF1'] < 3000)]
    # data = data[(data['BsmtFinSF2'] < 1100)]
    # data = data[(data['BsmtFullBath'] < 2.5)]
    # data = data[(data['BsmtHalfBath'] < 1.25)]
    # data = data[(data['EnclosedPorch'] < 350)]
    # data = data[(data['Fireplaces'] < 2.5)]
    # data = data[(data['GarageCars'] < 3.5)]
    data = data[(data['GrLivArea'] < 4500)] #
    # data = data[(data['HalfBath'] < 1.5)]
    data = data[(data['LotArea'] < 150000)] #
    # data = data[(data['MasVnrArea'] < 1200)]
    # data = data[(data['MiscVal'] < 4000)]
    data = data[(data['OpenPorchSF'] < 380)] #
    # data = data[(data['TotalBsmtSF'] < 4000)]
    return data

# Prints R2 and RMSE scores
def get_score(prediction, labels):
    print('R2: {}'.format(r2_score(prediction, labels)))
    print('RMSE: {}'.format(np.sqrt(mean_squared_error(prediction, labels))))

# Match the columns of dataframe a to b
def matchCols(a,b):
    cols = a.columns.tolist()
    b = b.reindex(columns=cols).fillna(0)
    return b

# Replace all nulls with the average of the exact column
# or the best value for string columns
def removeNulls(df):
    for i,column in enumerate(df):
        if (df[column].dtype == object):
            df[column] = df[column].fillna(df[column].mode()[0])
        else:
            df[column] = df[column].fillna(df[column].mean())
    return df

# Pre process function for the training data
def preProcess(df):
    df.drop(['LotFrontage', 'Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis=1, inplace=True) # NULL COLUMNS
    df.drop(["Id", "Street", "Utilities"], axis=1, inplace=True) # Unnecessary columns
    df = removeNulls(df)
    df = remove_correlated(df)
    df = pd.get_dummies(df)
    df = outliers(df)
    return df

def scale(df):
    return np.log1p(df)

def run(alg,t_X,ids,x_t,y_t,x,y):
    result = np.expm1(alg.predict(t_X))
    output = pd.DataFrame({'Id': ids, 'SalePrice': np.floor(result)})
    alg_result = alg.predict(x_t)
    get_score(alg_result, y_t)
    scores = cross_val_score(alg, x, y, cv=kfolds)
    print("\tAccuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    return output
################################################
##################MAIN CODE#####################
################################################
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train = preProcess(train)

################################################
###############TESTING CODE#####################
################################################
##### TRAIN DATA #####
train = scale(train) #Scale Data
Y = pd.DataFrame(train["SalePrice"].values,columns=["SalePrice"])
train.drop("SalePrice", axis=1, inplace=True)
X = train
##### TEST DATA #####
test_id = test.Id
test = pd.get_dummies(test)
test = matchCols(train,test)
test = scale(test) #Scale Test Data
test_X = test

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=203)  # test = 20%, train = 80%



names = [
         "ElasticNet"
]

classifiers = [
    linear_model.ElasticNetCV()
]

parameters = [
              {'alphas': [np.arange(0.0001,1,0.001),np.arange(0.0001,1,0.005)],
                'l1_ratio': [[0.01, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99],[0.01, 0.05,0.09,0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]],
                'max_iter' : [100000,200000]}
]

def mrse(y_true, y_pred):
    return 1 - np.sqrt(mean_squared_error(y_true, y_pred))


scorer = make_scorer(mrse, greater_is_better=True)

# print results
for name, classifier, params in zip(names, classifiers, parameters):
    gs = GridSearchCV(classifier, param_grid=params, scoring=scorer, n_jobs=-1)
    gs.fit(x_train, y_train)
    print("{} score: {} - best params: {}".format(name, gs.best_score_, gs.best_params_))




print("ElasticNet")
elasticnet = linear_model.ElasticNetCV(
        alphas=[0.0001, 0.0005, 0.001, 0.0015, 0.01, 0.015, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
        l1_ratio=[0.01, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99], max_iter=100000,n_jobs=-1).fit(x_train, y_train)
output = run(elasticnet,test_X,test_id,x_test,y_test,X,Y)
print(output)
output.to_csv('submission.csv',index=False)

print("\nXgBoost")
xg_reg = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='logloss',subsample=0.95, colsample_bytree=0.3, learning_rate=0.04,
                          max_depth=3, alpha=0.1, n_estimators=1000)
xg_reg = xg_reg.fit(x_train, y_train)
output = run(xg_reg,test_X,test_id,x_test,y_test,X,Y)
print(output)

print("\nLasso Regression")
lasso = Lasso(max_iter =  50000)
lasso_est = GridSearchCV(lasso, param_grid={"alpha": np.arange(0.0005, 0.001, 0.00001)})
lasso_est = lasso_est.fit(x_train,y_train)
output = run(lasso_est,test_X,test_id,x_test,y_test,X,Y)
print(output)

print("\nGradientBoostingRegressor")
g_best = ensemble.GradientBoostingRegressor(n_estimators=5000, random_state=200,alpha=0.8 ,learning_rate=0.015, max_depth=3,
                                            max_features="sqrt", min_samples_leaf=15, min_samples_split=10,
                                            loss='huber',warm_start = True)
g_best = g_best.fit(x_train, y_train)
output = run(g_best,test_X,test_id,x_test,y_test,X,Y)
print(output)

