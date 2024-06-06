from keras.models import Sequential
from keras.layers import Dense,SimpleRNN,Activation
from keras import optimizers

# Create a Sequential model
model=Sequential()
# Add a SimpleRNN layer with 50 units
model.add(SimpleRNN(50,input_shape=(49,1),return_sequences=False))
# Add a Dense layer with 46 units
model.add(Dense(46))
# Add a softmax activation layer
model.add(Activation("softmax"))

# Create an Adam optimizer with a learning rate of 0.001
adam=optimizers.Adam(learning_rate=0.001)
# Compile the model with categorical crossentropy loss, Adam optimizer, and accuracy metric
model.compile(loss="categorical_crossentropy",optimizer=adam,metrics=["accuracy"])
# Print the summary of the model
print(model.summary())




