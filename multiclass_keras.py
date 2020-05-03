# import libraries
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

# load dataset
df = pd.read_csv("./datasets/iris_data.csv")
df_values = df.values
x = df_values[:,0:4].astype(float)
y = df_values[:,4]

# encode class values as integers
encoded_y = LabelEncoder().fit_transform(y)

# convert integers to dummy variables => one hot encoding
dummy_y = np_utils.to_categorical(encoded_y)

# define base model
def base_model():
    model = Sequential()
    model.add(Dense(8, input_dim=4, activation="relu"))
    model.add(Dense(3, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

estimator = KerasClassifier(build_fn=base_model, epochs=200, batch_size=5, verbose=0)
kfold = KFold(n_splits=10, shuffle=True)

results = cross_val_score(estimator, x, dummy_y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
