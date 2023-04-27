import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ydata_profiling as pp

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold

from sklearn.linear_model import LinearRegression, SGDRegressor, RidgeCV
from sklearn.svm import SVR, LinearSVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.ensemble import BaggingRegressor, AdaBoostRegressor, VotingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
import sklearn.model_selection
from sklearn.model_selection import cross_val_predict as cvp
from sklearn import metrics
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb
import lightgbm as lgb

import warnings
warnings.filterwarnings("ignore")

# Downloading dataset
valid_part = 0.3
#pd.set_option('max_columns',20)
pd.set_option('display.max_columns', None)

train0 = pd.read_csv('vehicles_dataset.csv')
print(train0.head())

numerics = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']
categorical_columns = []
features = train0.columns.values.tolist()
for col in features:
    if train0[col].dtype in numerics: continue
    categorical_columns.append(col)

# Encoding categorical features
for col in categorical_columns:
    if col in train0.columns:
        le = LabelEncoder()
        le.fit(list(train0[col].astype(str).values))
        train0[col] = le.transform(list(train0[col].astype(str).values))

train0['year'] = (train0['year']-1900).astype(int)
train0['odometer'] = train0['odometer'].astype(int)

# EDA

fig, ax = plt.subplots(1,3)

ax[0].hist(x[y==0, 2], color='r')
ax[0].set(title=classes[0])

ax[1].hist(x[y==1, 2], color='b')
ax[1].set(title=classes[1])

ax[2].hist(x[y==2, 2], color ='g')
ax[2].set(title=classes[2])

plt.show()
plt.close()
