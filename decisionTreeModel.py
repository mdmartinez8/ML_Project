import pandas as pd

df = pd.read_csv('./archive/fraudTrain_simple.csv')
# This line was uncommented because fraudTrain.csv is too big, fraudTrain_simple.csv is being used in the mean time
# df = pd.read_csv('./archive/fraudTrain.csv')

# Preprocessing the dataset
# trans_date_trans_time needs to be broken down to be taken into consideration for numerical data
df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'], format = '%m/%d/%y %H:%M')

df['hour'] = df['trans_date_trans_time'].dt.hour
df['day'] = df['trans_date_trans_time'].dt.day
df['month'] = df['trans_date_trans_time'].dt.month
df['year'] = df['trans_date_trans_time'].dt.year
df['dayofweek'] = df['trans_date_trans_time'].dt.dayofweek

# One hot encoding on 'category' column:
catergories = pd.get_dummies(df['category'], prefix = 'category')

# Concatenating one hot encoding data frame
df = pd.concat([df, catergories], axis = 1)
df.drop(['category', 'trans_date_trans_time'], axis = 1, inplace = True)