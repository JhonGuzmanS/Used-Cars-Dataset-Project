#import opendatasets as od
import math 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import *

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_selection import f_regression, SelectKBest
from scipy.stats import uniform, truncnorm, randint

from sklearn import metrics
from sklearn.metrics import mean_absolute_error as mae
from joblib import dump, load

from sklearn.ensemble import RandomForestRegressor

valid_part = 0.3
pd.set_option('display.max_columns', None)

# Edit the filename to 'final_df.csv to use the proper dataset'
df = pd.read_csv('vehicles.csv')

# Delete lines 27-33, df.drop will cause an error for the edited dataset
print(df.columns)
print("\n\n")

df = df.drop(columns = ['id', 'url', 'region', 'region_url', 'title_status', 'VIN','image_url', 'description', 
                        'county', 'posting_date', 'lat', 'long', 'model', 'transmission','fuel','state','paint_color','drive'])


# Process and Prepare the data
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3-Q1

filtered_df = (df['price'] >= Q1 - 1.5 * IQR)  & (df['price'] <= Q3 + 1.5 * IQR)

old_size = df.count()['price']
df = df.loc[filtered_df]
new_size = df.count()['price']
print(old_size-new_size, '(', '{:.2f}'.format(100*(old_size-new_size)/old_size), '%',')', 'outliers removed from dataset')


Q1 = df['odometer'].quantile(0.25)
Q3 = df['odometer'].quantile(0.75)
IQR = Q3-Q1

filtered_df = (df['odometer'] <= Q3 + 3 * IQR)

old_size = df.count()['odometer']
df = df.loc[filtered_df]
new_size = df.count()['odometer']
print(old_size-new_size, '(', '{:.2f}'.format(100*(old_size-new_size)/old_size), '%',')', 'outliers removed from dataset')
print("\n\n")

df = df[df['year'].between(1970,2020)]

#print(df.head)
print("\n\n")

# Remove rows that contain less than 40% info/data, otherwise replace values
null_values = df.isna().sum()

def na_filter(na, threshold = 0.4):
    column = []
    for i in na.keys():
        if na[i]/df.shape[0] < threshold:
            column.append(i)
    return column

df = df[na_filter(null_values)]

df = df.replace(np.nan, 'other', regex=True)

#print(df.head)
print("\n\n")

catColumns = ['manufacturer','type']
for column in catColumns: 
    column = pd.get_dummies(df[column],drop_first=True) 
    df = pd.concat([df,column],axis=1)
df = df.drop(columns = catColumns)

df = df.astype(int)
print(df.head)   

#print(df.head)
print("\n\n")

X_train, X_test, y_train, y_test= train_test_split(df.drop('price',axis=1), df['price'],test_size=0.20, random_state=5564)

df = X_train.copy()
df_test = X_test.copy()

df_train_labels = y_train.copy()
df_test_labels = y_test.copy()

scaler = StandardScaler()
for column in ['year','odometer']:
    df[column] = scaler.fit_transform(df[column].values.reshape(-1,1))

std_Scaler = StandardScaler()
for column in ['year','odometer']:
    df_test[column] = scaler.fit_transform(df_test[column].values.reshape(-1,1))

print(df_test.head())

#Start of Model

model_params = {
    'n_estimators': randint(4,200),
    'min_samples_split': uniform(0.01, 0.199)
}

rf2 = RandomForestRegressor()
clf = RandomizedSearchCV(rf2, model_params, n_iter=20, cv=5, random_state=5564)
model = clf.fit(df,df_train_labels)

y_pred = model.predict(df_test)

Acc = pd.DataFrame(index=None, columns=['Model','Mean Absolute Error','Root Mean Squared  Error',
                                        'Accuracy on Traing set','Accuracy on Testing set'])

name = 'Random Forest Regressor'
MAE = round(metrics.mean_absolute_error(df_test_labels,y_pred),2)
RMSE = np.sqrt(metrics.mean_squared_error(df_test_labels, y_pred))
ATrS =  model.score(df,df_train_labels)
ATeS = model.score(df_test,df_test_labels)
Acc = Acc.append(pd.Series({'Model':name,'Mean Absolute Error': MAE,'Root Mean Squared  Error': RMSE,
                            'Accuracy on Traing set':ATrS,'Accuracy on Testing set':ATeS}),ignore_index=True)

print(Acc)