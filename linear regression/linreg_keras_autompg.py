# required libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense

data = pd.read_csv("./datasets/auto mpg.csv")

# check for NA values
data.isna().sum()
data = data.dropna()

# train test split
train = data.sample(frac=0.8, random_state=0)
test = data.drop(train.index)

'''
                    ./results/plot1-pairwise_relationship.png
'''
sns.pairplot(train[["mpg", "cylinders", "displacement", "weight"]], diag_kind="kde")

# remove the variable 'mpg' since that's what we are going to predict
train_labels = train.pop('mpg')
test_labels = test.pop('mpg')

# normalise the dataset
# this normalised data is what we'll use in the model
train_stats = train.describe()
train_stats = train_stats.transpose()
train_stats

def norm(x):
  return (x - train_stats['mean']) / train_stats['std']

normed_train_data = norm(train)
normed_test_data = norm(test)

# building the model
def build_model():
    model = Sequential()
    model.add(Dense(64, activation="relu", input_shape=[len(train.keys())]))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(1))
    model.compile(loss="mse", optimizer="rmsprop", metrics=["mae", "mse"])

    return model

model = build_model()
model.summary()

# train the model and record training and validation accuracy in history
history = model.fit(normed_train_data, train_labels, epochs=1000, validation_split=0.2, verbose=0)

hist = pd.DataFrame(history.history)
hist["epoch"] = history.epoch
hist.tail()

# making predictions
test_predictions = model.predict(normed_test_data).flatten()

'''
                    ./results/plot2-true_vs_pred.png
'''
a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('true values: mpg')
plt.ylabel('predictions: mpg')
lims = [0, 50]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)


# error distribution
'''
                    ./results/plot3-error_distribution.png
'''
error = test_predictions - test_labels
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error [MPG]")
_ = plt.ylabel("Count")
