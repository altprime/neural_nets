import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout


# import training data
train = pd.read_csv('train_RIL.csv')
training_set = train.iloc[:, 1:2].values

# feature scaling
sc = MinMaxScaler(feature_range = (0, 1))
training_scaled = sc.fit_transform(training_set)

# data structure with 60 time steps and 1 output
xtrain = []
ytrain = []
for i in range(20, 1098):
    X_train.append(training_scaled[i-20:i, 0])
    y_train.append(training_scaled[i, 0])
xtrain, ytrain = np.array(xtrain), np.array(ytrain)

# reshaping
xtrain = np.reshape(xtrain, (xtrain.shape[0], xtrain.shape[1], 1))


# RNN
model = Sequential()
model.add(LSTM(units = 50, return_sequences = True, input_shape = (xtrain.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50))
model.add(Dropout(0.2))
# output layer
model.add(Dense(units = 1))
# compile
model.compile(optimizer = 'adam', loss = 'mean_squared_error')
# fitting
model.fit(xtrain, ytrain, epochs = 100, batch_size = 32)

# import testing data
test = pd.read_csv('test.csv')
actual_price = test.iloc[:, 1:2].values

# predictions
dataset_complete = pd.concat((train['Open'], test['Open']), axis = 0)
inputs = dataset_complete[len(dataset_complete) - len(test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
xtest = []
for i in range(60, 80):
    xtest.append(inputs[i-60:i, 0])
xtest = np.array(xtest)
xtest = np.reshape(xtest, (xtest.shape[0], xtest.shape[1], 1))
predicted_price = regressor.predict(xtest)
predicted_price = sc.inverse_transform(predicted_price)

# visualising the results
plt.plot(actual_price, color = 'red', label = 'Actual Stock Price')
plt.plot(predicted_price, color = 'blue', label = 'Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()



