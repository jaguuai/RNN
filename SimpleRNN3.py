import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN

# Convert into dataset matrix
def convertToMatrix(data, step):
    X, Y = [], []
    for i in range(len(data) - step):
        d = i + step
        X.append(data[i:d, ])
        Y.append(data[d, ])
    return np.array(X), np.array(Y)

# Load dataset
df = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv", usecols=[1], engine="python")
df.head()
plt.plot(df)
plt.show()

step = 4
N = df.shape[0]
Tp = int(df.shape[0] * 0.8)
values = df.values
train, test = values[0:Tp, :], values[Tp:N, :]

# Add step elements into train and test
test = np.append(test, np.repeat(test[-1, ], step))
train = np.append(train, np.repeat(train[-1, ], step))

trainX, trainY = convertToMatrix(train, step)
testX, testY = convertToMatrix(test, step)
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# Create and compile the SimpleRNN model
model = Sequential()
model.add(SimpleRNN(units=32, input_shape=(1, step), activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(1))
model.compile(loss="mean_squared_error", optimizer="rmsprop")
model.summary()
# Train the model
model.fit(trainX, trainY, epochs=10, batch_size=16, verbose=2)
# Make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
predicted = np.concatenate((trainPredict, testPredict), axis=0)
# Evaluate the model
trainScore = model.evaluate(trainX, trainY, verbose=0)
testScore = model.evaluate(testX, testY, verbose=0)
print(f"Train Score: {trainScore}")
print(f"Test Score: {testScore}")

# Plot data and prediction with a vertical line indicating the train-test split
plt.plot(df, label="Data")
plt.plot(predicted, label="Prediction")
plt.axvline(df.index[Tp], color="r")
plt.legend()
plt.show()



















