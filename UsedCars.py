import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ydata_profiling as pp

import warnings
warnings.filterwarnings("ignore")

# Downloading dataset
valid_part = 0.3
#pd.set_option('max_columns',20)
pd.set_option('display.max_columns', None)

train0 = pd.read_csv('vehicles.csv')

# compared to model used in Kaggle, these are columns that were recently added into the dataset:
# model, county, state, posting_date
drop_columns = ['id', 'url', 'region_url', 'title_status', 'VIN', 'image_url', 'description', 'lat', 'long',
                 'region', 'county', 'posting_date', 'model']
train0 = train0.drop(columns=drop_columns)

train0.info()

train0 = train0.dropna()
print(train0.head())
train0.to_csv('vehicles_dataset.csv')

print("\n\n\n")
print(train0.head())
print("\n\n\n")



