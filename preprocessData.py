import pandas as pd

df = pd.read_csv('./archive/fraudTrain_simple.csv')

# This line was uncommented because fraudTrain.csv is too big, fraudTrain_simple.csv is being used in the mean time
# df = pd.read_csv('./archive/fraudTrain.csv')

df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'], format = '%m/%d/%y %H:%M')

df['hour'] = df['trans_date_trans_time'].dt.hour
df['day'] = df['trans_date_trans_time'].dt.day
df['month'] = df['trans_date_trans_time'].dt.month
df['year'] = df['trans_date_trans_time'].dt.year
df['dayofweek'] = df['trans_date_trans_time'].dt.dayofweek