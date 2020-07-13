import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

# get the stock quote 
df = web.DataReader('INFY', data_source='yahoo', start='2012-01-01', end='2019-12-17') 
# show the data 
df.shape

# plot the closing price
plt.figure(figsize=(16,8))
plt.title('Close Price')
plt.plot(df['Close'])
plt.xlabel('Date',fontsize=18)
plt.ylabel('Close Price USD ($)',fontsize=18)
plt.show()

# create a new dataframe with only the 'Close' column
data = df.filter(['Close']) # converting to numpy array
dataset = data.values # get the number of rows to train the model
training_data_len = math.ceil( len(dataset) *.8)

# scale the data to be between 0 and 1 
scaler = MinMaxScaler(feature_range=(0, 1)) 
scaled_data = scaler.fit_transform(dataset)

# create scaled training set 
train_data = scaled_data[0:training_data_len  , : ]
# split the data 
x_train=[]
y_train = []
for i in range(60,len(train_data)):
    x_train.append(train_data[i-60:i,0])
    y_train.append(train_data[i,0])

# convert to np array
x_train, y_train = np.array(x_train), np.array(y_train)

# reshape for nn
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

# build nn
model = Sequential()
model.add(LSTM(units=50, return_sequences=True,input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))
# compile
model.compile(optimizer='adam', loss='mean_squared_error')
# model fitting
model.fit(x_train, y_train, batch_size=1, epochs=1)

# test data
test_data = scaled_data[training_data_len - 60: , : ]
# create the testing sets
x_test = []
y_test =  dataset[training_data_len : , : ] 

# get all of the rows from index 1603 to the rest and all of the columns 
# (in this case it's only column 'Close'), so 2003 - 1603 = 400 rows of data
for i in range(60,len(test_data)):
    x_test.append(test_data[i-60:i,0])

# convert to a np array 
x_test = np.array(x_test)
# reshape the data into shape accepted by nn
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))

# prediction
predictions = model.predict(x_test) 
# inverse scaling
predictions = scaler.inverse_transform(predictions)

# accuracy
rmse=np.sqrt(np.mean(((predictions- y_test)**2)))
rmse

# plots

train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='upper left')
plt.show()








