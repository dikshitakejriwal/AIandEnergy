import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

# Load the data from the CSV file
data = pd.read_csv('final proj/solar_weather.csv')
data = data.drop(['Time', 'Energy delta[Wh]', 'sunlightTime', 'dayLength', 'weather_type', 'hour', 'month'], axis=1)

# Split the data into training and testing sets
train_data = data.sample(frac=0.8, random_state=0)
test_data = data.drop(train_data.index)

# Separate the input features (X) and output variable (y) for training and testing sets
train_X = train_data.drop(['GHI'], axis=1)
train_y = train_data['GHI']
test_X = test_data.drop(['GHI'], axis=1)
test_y = test_data['GHI']
# print(train_y.head(10))

# Define the neural network architecture
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=[len(train_X.keys())]),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1)
])

# Compile the model with appropriate loss and optimizer functions
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam())

# Train the model on the training data
model.fit(train_X, train_y, epochs=100)

# Evaluate the model on the testing data
test_loss = model.evaluate(test_X, test_y)

# Print the test loss
print('Test loss:', test_loss)

# Use the trained model to make predictions on new data
new_data = pd.DataFrame({'temp': [27.5], 'pressure': [1008], 'humidity': [52], 'wind_speed': [1.8], 'rain_1h': [0.0], 'snow_1h': [0.0], 'clouds_all': [38], 'isSun': [1], 'SunlightTime/daylength': [0.85]}) #66.1
new_data1 = pd.DataFrame({'temp': [-1.5], 'pressure': [1009], 'humidity': [97], 'wind_speed': [2.0], 'rain_1h': [0.0], 'snow_1h': [0.0], 'clouds_all': [93], 'isSun': [1], 'SunlightTime/daylength': [0.09]}) #11.5
new_data2 = pd.DataFrame({'temp': [12.5], 'pressure': [1020], 'humidity': [81], 'wind_speed': [0.8], 'rain_1h': [0.0], 'snow_1h': [0.0], 'clouds_all': [100], 'isSun': [1], 'SunlightTime/daylength': [0.79]}) #46.4
prediction = model.predict(new_data)
prediction1 = model.predict(new_data1)
prediction2 = model.predict(new_data2)
print('Prediction:', prediction, 'Actual:', 66.1)
print('Prediction:', prediction1, 'Actual:', 11.5)
print('Prediction:', prediction2, 'Actual', 46.4)
print('Accuracy:', (prediction[0]/66.1 + prediction1[0]/11.5 + prediction2[0]/46.4)/3)
