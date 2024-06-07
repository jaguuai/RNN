from tensorflow import keras 
from tensorflow.keras import layers
model=keras.Sequential()
model.add(layers.Input(shape=(None,)))
model.add(layers.Embedding(input_dim=100,output_dim=32))
model.add(layers.LSTM(64))
model.add(layers.Dense(1))
model.summary()









