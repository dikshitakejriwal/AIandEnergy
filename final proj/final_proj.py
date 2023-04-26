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
prediction = model.predict(new_data)
# print('Prediction:', prediction, 'Actual:', x)
surface_area = input("Enter surface area of array(m^2): ")
time = input("Enter time in hours: ")
energy = float(prediction) * float(surface_area) * float(time)
efficiency = input("Enter PV panel efficiency: ")
power_output = float(energy) * float(efficiency)
print("Predicted power output over", time, "hours = ", power_output/1000, "kW")
