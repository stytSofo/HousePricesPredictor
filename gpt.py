import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# load the data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# preprocess the data
train_df = pd.get_dummies(train_df)
test_df = pd.get_dummies(test_df)

cols = train_df.columns.tolist()
test_df = test_df.reindex(columns=cols).fillna(0)

# handle missing values
train_df = train_df.fillna(train_df.mean())
test_df = test_df.fillna(test_df.mean())

X = train_df.iloc[:, :-1]
Y = train_df['SalePrice']

# split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=200)

# train the model
model = LinearRegression()
model.fit(X_train, y_train)

# evaluate the model on the validation set
y_pred = model.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
rmse = np.sqrt(mse)
print(f'Validation RMSE: {rmse}')

# make predictions on the test set
test_preds = model.predict(test_df)

# create a submission file
submission_df = pd.DataFrame({'Id': test_df['Id'], 'SalePrice': test_preds})
submission_df.to_csv('submission.csv', index=False)
