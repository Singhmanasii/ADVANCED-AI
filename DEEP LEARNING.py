#pip install numpy pandas matplotlib scikit-learn tensorflow


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

# Step 1: Load dataset (using a new example dataset)
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'
data = pd.read_csv(url, header=0, index_col=0, parse_dates=True)

# Show the first few rows of the data
print(data.head())

# Step 2: Preprocessing - Using the dataset's values
data_values = data.values
data_values = data_values.reshape(-1, 1)

# Normalize the data using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data_values)

# Step 3: Create sequences of data for training
def create_sequences(data, time_step=12):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:i + time_step])
        y.append(data[i + time_step])
    return np.array(X), np.array(y)

time_step = 12  # Using 12 previous months to predict the next one
X, y = create_sequences(scaled_data, time_step)

# Reshape the data for LSTM (samples, time steps, features)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Step 4: Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Step 5: Train the model
model.fit(X, y, epochs=10, batch_size=32)

# Step 6: Make predictions
predictions = model.predict(X)

# Step 7: Inverse transform the predictions
predictions = scaler.inverse_transform(predictions)

# Plot the predictions and the true values
plt.figure(figsize=(10, 6))
plt.plot(data_values[time_step:], label="True Values")
plt.plot(predictions, label="Predictions")
plt.legend()
plt.show()
