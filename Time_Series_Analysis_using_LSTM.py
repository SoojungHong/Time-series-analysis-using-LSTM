#  https://stackabuse.com/time-series-analysis-with-lstm-using-pythons-keras-library/

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# load data
apple_training_complete = pd.read_csv('C:/Users/shong/PycharmProjects/keras_exercise/data/AAPL.csv')
#print('apple_training_complete : \n', apple_training_complete)
apple_training_processed = apple_training_complete.iloc[:, 1:2].values  # only column in index 1 which is 'Open' column
#print('apple_training_processed : \n', apple_training_processed)


# data normalization
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
apple_training_scaled = scaler.fit_transform(apple_training_processed)
#print('apple_training_scaled : \n', apple_training_scaled)


# feature set
features_set = []
labels = []
for i in range(60, 1260):  # to predict the time T's stock price, we use previous 60 days' value
      features_set.append(apple_training_scaled[i-60:i, 0])  # adding one array, this one array contains 60 values (0th ~ 59th value), it is ndarray type
      labels.append(apple_training_scaled[i, 0]) # adding a label, which is 60th apple open stock value is label, it is float64 type

features_set, labels = np.array(features_set), np.array(labels)  # features_set is array of array
#print('feature_set shape :', features_set.shape)  # (1200, 60)
#print('labels : ', labels.shape)  # (1200,)
features_set = np.reshape(features_set, (features_set.shape[0], features_set.shape[1], 1))  # features_set.shape[0] is 1200, features_set.shape[1] is 60
#print('reshaped feature_set shape : \n', features_set.shape)  # shape is three dimensional (1200, 60, 1)
#print('features_set.shape[1] : \n', features_set.shape[1])


# training LSTM
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(features_set.shape[1], 1)))  # the number of neurons is 50, return_sequence is true since we will add more layers to the model
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50))
model.add(Dropout(0.2))

model.add(Dense(units=1))  # the number of neurons in the dense layer will be set to 1 since we want to predict a single value in the output


# model compilation - we need to compile out LSTM before we train the model on training data
model.compile(optimizer='adam', loss='mean_squared_error')


# algorithm training
model.fit(features_set, labels, epochs=200, batch_size=32)  # ToDo : at the moment, set epoch 1


# testing our LSTM
apple_testing_complete = pd.read_csv('C:/Users/shong/PycharmProjects/keras_exercise/data/apple_testing.csv')
apple_testing_processed = apple_testing_complete.iloc[:, 1:2].values


# convert test data to right format
apple_total = pd.concat((apple_training_complete['Open'], apple_testing_complete['Open']), axis=0)  # axis = 0 is concatenate into new row
test_inputs = apple_total[len(apple_total) - len(apple_testing_complete)-60:].values

test_inputs = test_inputs.reshape(-1,1)
test_inputs = scaler.transform(test_inputs)

test_features = []
for i in range(60,80):
      test_features.append(test_inputs[i-60:i, 0])

test_features = np.array(test_features)
test_features = np.reshape(test_features, (test_features.shape[0], test_features.shape[1], 1))


# make prediction
predictions = model.predict(test_features)
# reverse the scaled prediction
predictions = scaler.inverse_transform(predictions)


# plot
plt.figure(figsize=(10,6))
plt.plot(apple_testing_processed, color='blue', label='Actual Apple Stock Price')
plt.plot(predictions , color='red', label='Predicted Apple Stock Price')
plt.title('Apple Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Apple Stock Price')
plt.legend()
plt.show()
