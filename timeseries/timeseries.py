import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler

# Generate synthetic time series data
np.random.seed(42)
time = np.arange(0, 365, 1)  # 1 year of daily data

# Generate multiple time series variables
var1 = np.sin(time / 50) * 20 + np.random.normal(scale=2, size=len(time)) + 30
var2 = np.cos(time / 40) * 15 + np.random.normal(scale=2, size=len(time)) + 40
var3 = np.sin(time / 30) * 10 + np.random.normal(scale=2, size=len(time)) + 25
var4 = np.cos(time / 20) * 5 + np.random.normal(scale=2, size=len(time)) + 50

# Target variable (can be one of the above with added noise)
target = var1 * 0.6 + var2 * 0.2 + var3 * 0.2 + np.random.normal(scale=3, size=len(time))

# Create DataFrame
df = pd.DataFrame({
    'Date': pd.date_range(start='2019-01-01', periods=len(time), freq='D'),
    'Variable 1': var1,
    'Variable 2': var2,
    'Variable 3': var3,
    'Variable 4': var4,
    'Target': target
})

# Normalize data
scaler = MinMaxScaler()
df_scaled = df.copy()
df_scaled.iloc[:, 1:] = scaler.fit_transform(df.iloc[:, 1:])

# Prepare data for LSTM
seq_length = 20

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length, :-1])  # All variables except target
        y.append(data[i + seq_length, -1])  # Target variable
    return np.array(X), np.array(y)

data_values = df_scaled.iloc[:, 1:].values
X, y = create_sequences(data_values, seq_length)

# Split into train/test sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshape for LSTM
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2]))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2]))

# Build LSTM model
model = keras.Sequential([
    layers.LSTM(50, activation="relu", return_sequences=True, input_shape=(seq_length, X_train.shape[2])),
    layers.LSTM(50, activation="relu"),
    layers.Dense(1)
])

model.compile(optimizer="adam", loss="mse")
model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test))

# Make predictions
predictions = model.predict(X_test)

# Inverse transform predictions
y_test_actual = scaler.inverse_transform(np.c_[np.zeros((len(y_test), 4)), y_test])[:, -1]
predictions_actual = scaler.inverse_transform(np.c_[np.zeros((len(predictions), 4)), predictions])[:, -1]

# Plot time series data (similar to your uploaded image)
plt.figure(figsize=(10, 5))
plt.plot(df['Date'], df['Variable 1'], label="Variable 1", color="blue")
plt.plot(df['Date'], df['Variable 2'], label="Variable 2", color="orange")
plt.plot(df['Date'], df['Variable 3'], label="Variable 3", color="green")
plt.plot(df['Date'], df['Variable 4'], label="Variable 4", color="purple")
plt.plot(df['Date'][train_size + seq_length:], y_test_actual, label="Target", color="red", linewidth=2)

plt.xlabel("Date")
plt.ylabel("Value")
plt.title("Time Series Forecasting - Multiple Variables")
plt.legend()
plt.show()
