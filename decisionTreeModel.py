import pandas as pd

df = pd.read_csv('./archive/fraudTrain_simple.csv')
# This line was uncommented because fraudTrain.csv is too big, fraudTrain_simple.csv is being used in the mean time
# df = pd.read_csv('./archive/fraudTrain.csv')

# ----- PREPROCESSING -----
# trans_date_trans_time needs to be broken down to be taken into consideration for numerical data
df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'], format = '%m/%d/%y %H:%M')

df['hour'] = df['trans_date_trans_time'].dt.hour
df['day'] = df['trans_date_trans_time'].dt.day
df['month'] = df['trans_date_trans_time'].dt.month
df['year'] = df['trans_date_trans_time'].dt.year
df['dayofweek'] = df['trans_date_trans_time'].dt.dayofweek

# MERCHANT STILL NEEDS TO BE PRE PROCESSED

# One hot encoding on 'category' column:
catergories = pd.get_dummies(df['category'], prefix = 'category')

# Concatenating one hot encoding data frame
df = pd.concat([df, catergories], axis = 1)
df.drop(['category', 'trans_date_trans_time'], axis = 1, inplace = True)

# Drop unnecessary columns:
df.drop(df.columns[0], axis = 1, inplace = True)
dropCols = ['cc_num', 'first', 'last', 'gender', 'street', 'city', 'state', 'zip', 'job', 'dob', 'trans_num']

# Since we have 2 separate files, one for testing and one for 
# training, we do not have to split the file into train and test

# ----- TRAINING MODEL -----
# In progress:
# X_train = df.drop('is_fraud', axis = 1)
# Y_train = df['is_fraud']
