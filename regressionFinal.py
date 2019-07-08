#import the relevant packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

#read in the file

db = pd.read_csv("train.csv")

#initial view of the data

db.head()
fig,ax = plt.subplots()
ax.scatter(db['SalePrice'], db['LotFrontage'])

#list the frequency of missing values

naValues = db.isnull().sum()
naValues = naValues.drop(naValues[naValues==0].index).sort_values(ascending=False)
print(naValues)

# clean up the missing values

db['Electrical'] = db['Electrical'].fillna(value='SBrkr')
db['MasVnrType'] = db['MasVnrType'].fillna(value='None')
db['MasVnrArea'] = db['MasVnrArea'].fillna(0)
cols = ['BsmtQual','BsmtCond','BsmtFinType1','BsmtExposure','BsmtFinType2','GarageType','GarageYrBlt','GarageFinish','GarageQual','GarageCond','FireplaceQu','Fence','Alley','MiscFeature','PoolQC']
for c in cols:
    db[c] = db[c].fillna('NA')

sumLF = db['LotFrontage'].sum()
naValuesLF = db['LotFrontage'].isnull().sum()
mean = sumLF / (len(db.index) - naValuesLF)
db['LotFrontage'] = db['LotFrontage'].fillna(round(mean,1))

#check to ensure there are no more missing values remaining

naValues = db.isnull().sum()
naValues = naValues.drop(naValues[naValues==0].index).sort_values(ascending=False)
print(naValues)

#some values must be converted into strings, so that they will be proportional when converted back using our label encoder
cols = ['YrSold','MoSold','MSSubClass','OverallCond','OverallQual','GarageYrBlt']
for c in cols:
    db[c] = db[c].astype(str)
#generate and fit the label encoder

columns = ('MoSold', 'MSSubClass', 'OverallCond', 'OverallQual', 'Street', 'Alley', 'MSZoning', 'BsmtQual', 'BsmtCond', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition', 'GarageYrBlt', 'YrSold')
for c in columns:
    encoder = LabelEncoder()
    encoder.fit(list(db[c].values))
    db[c] = encoder.transform(list(db[c].values))

#transform the data to be ready for our split into train and test sets

db = db.drop(['Id'], axis=1)
y = db['SalePrice']
db = db.drop(['SalePrice'], axis=1)

#split the dataset into train and test

X_train, X_test, Y_train, Y_test = train_test_split(db, y, test_size=0.3)
print (X_train.shape, Y_train.shape)
print (X_test.shape, Y_test.shape)

#fit the simple Linear Regression to the data

LR = LinearRegression()
model = LR.fit(X_train, Y_train)
y_predicted = model.predict(X_test)
rmse = mean_squared_error(y_predicted, Y_test)
r2 = r2_score(y_predicted, Y_test)
print("RMSE: {}".format(rmse))
print("r2 Score: {}".format(r2))

#the ridge model

ridge = Ridge()
alphaVals = {'alpha': [1e-15,1e-10,1e-8,1e-4,1e-3,1e-2,1,5,10,20]}
ridge_regressor = GridSearchCV(ridge, alphaVals, scoring="neg_mean_squared_error", cv=10)
regressorModel = ridge_regressor.fit(X_train, Y_train)
print(ridge_regressor.best_params_, ridge_regressor.best_score_)
y_predictionsRidge = regressorModel.predict(X_test)
print("r2 score: {}".format(r2_score(y_predictionsRidge, Y_test)))


#lasso regression

lasso = Lasso()
lasso_regressor = GridSearchCV(lasso, alphaVals, scoring="neg_mean_squared_error", cv=10)
lassoModel = ridge_regressor.fit(X_train, Y_train)
y_predictionsLasso = lassoModel.predict(X_test)
print("r2 score: {}".format(r2_score(y_predictionsLasso, Y_test)))

#bayesian ridge

bayesian = linear_model.BayesianRidge()
modelBayesian = bayesian.fit(X_train, Y_train)
y_predictionsBayesian = modelBayesian.predict(X_test)
print("r2 score: {}".format(r2_score(y_predictionsBayesian, Y_test)))
