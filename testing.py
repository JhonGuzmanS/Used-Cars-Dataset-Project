import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR, LinearSVR
from sklearn.tree import DecisionTreeRegressor

import warnings
warnings.filterwarnings("ignore")

# Loading dataset
valid_part = 0.3
pd.set_option('display.max_columns', None)

df = pd.read_csv('vehicles_dataset.csv')

print(df['price'].value_counts())

price = df['price'].value_counts()
plt.plot(range(len(price)), price)
#plt.show()

drop_columns = ["state"]
df = df.drop(columns=drop_columns)
df = df.drop(columns='Unnamed: 0', axis=1)

print(df.head())

# Determination categorical features - converts categorical featrues(ex: condition, cylinders, transmission)
# into integer values

numerics = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']
categorical_columns = []
features = df.columns.values.tolist()
for col in features:
    if df[col].dtype in numerics: continue
    categorical_columns.append(col)

# Encoding categorical features
for col in categorical_columns:
    if col in df.columns:
        le = LabelEncoder()
        le.fit(list(df[col].astype(str).values))
        df[col] = le.transform(list(df[col].astype(str).values))

df['year'] = (df['year']-1900).astype(int)
df['odometer'] = df['odometer'].astype(int)

print(df.head())

# EDA

df = df[df['price'] > 1000]
df = df[df['price'] < 60000]

print(df.corr())

ds_features = df.drop(["price"], axis=1)
ds_label = df["price"]
X_train, X_test, y_train, y_test = train_test_split(ds_features, ds_label, test_size=0.2, random_state=42)

print(len(X_train))

data = X_train.iloc[:5]
label = y_train.iloc[:5]

line_reg = LinearRegression()
line_reg.fit(X_train, y_train)
print("Prediction: ", line_reg.predict(data))
print("Labels: ", list(label))
print(line_reg.score(X_train, y_train))

print("\n")
dec_tree = DecisionTreeRegressor()
dec_tree.fit(X_train, y_train)
print("Prediction: ", dec_tree.predict(data))
print("Labels: ", list(label))
print(dec_tree.score(X_train, y_train))
