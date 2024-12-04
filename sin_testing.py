import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras


# Generate data
x = np.linspace(0, 4 * np.pi, 1000)  # Input values (0 to 2π)
y = np.sin(x - np.pi/4)/2                     # Target values (sine wave)

# Plot the sine wave
# plt.plot(x, y, label="Sine wave")
# plt.xlabel("x")
# plt.ylabel("sin(x)")
# plt.legend()
# plt.show()

from sklearn.model_selection import train_test_split

# Reshape x for the neural network input
x = x.reshape(-1, 1)  # Make it a 2D array (samples, features)

# Split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

# # Define the model
# model = Sequential([
#     Dense(64, activation='relu', input_shape=(1,)),  # Input: 1D (x value)
#     Dense(64, activation='relu'),                    # Hidden layer
#     Dense(1)                                         # Output: 1D (sin(x))
# ])

# # Compile the model
# model.compile(optimizer='adam', loss='mse')  # Mean squared error for regression


model = keras.models.Sequential() 
# model.add(keras.layers.Embedding(1000, 128)) 
# model.add(keras.layers.Bidirectional( 
#     keras.layers.LSTM(64, return_sequences=True))) 
# model.add(keras.layers.Bidirectional(keras.layers.LSTM(64))) 
model.add(keras.layers.Dense(128, activation="relu")) 
model.add(keras.layers.Dense(128, activation="relu"))
model.add(keras.layers.Dense(128, activation="relu"))
model.add(keras.layers.Dense(128, activation="relu"))

model.add(keras.layers.Dense(1, activation="sigmoid")) 

# model.compile("rmsprop", "binary_crossentropy", metrics=["accuracy"]) 
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.summary()
history = model.fit(x_train, y_train, epochs=50, batch_size = 10)

test_loss = model.evaluate(x_test, y_test)
print("Test Loss:", test_loss)
x_test_2 = x_test + 1
y_pred = model.predict(x_test)

# Plot true vs. predicted values
plt.scatter(x_test, y_test, label="True values", color="blue", alpha=0.6)
plt.scatter(x_test, y_pred, label="Predicted values", color="red", alpha=0.6)
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.legend()
plt.show()

y_fit = model.predict(x)

# Plot true sine wave and model's prediction
plt.plot(x, y, label="True sine wave", color="blue")
plt.plot(x, y_fit, label="Fitted curve", color="red", linestyle="--")
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.legend()
plt.show()