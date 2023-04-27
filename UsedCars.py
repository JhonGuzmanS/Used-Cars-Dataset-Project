import numpy as np
import pandas as pd
# %matplotlib inline

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold


# model tuning

import warnings
warnings.filterwarnings("ignore")

# Downloading dataset
valid_part = 0.3
#pd.set_option('max_columns',20)
pd.set_option('display.max_columns', None)

# Note: 'vehicles.csv' is the original dataset(~2GB) ___ 'vehicles.csv' is the updated one with the removal of colomns listed on line 22 
train0 = pd.read_csv('vehicles.csv')

drop_columns = ['url', 'region_url', 'title_status', 'VIN', 'size', 'image_url', 'description', 'lat', 'long', 'posting_date', 'county', 'type']
train0 = train0.drop(columns=drop_columns)

#train0.info()

train0 = train0.dropna()
print(train0.head())
#train0.to_csv('vehicles_dataset.csv')


