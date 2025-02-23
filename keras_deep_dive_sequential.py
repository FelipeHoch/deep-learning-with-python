from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential(name="deep_dive")

model.add(layers.Dense(64, activation="relu", name="first_layer"))
model.add(layers.Dense(10, activation="softmax", name="last_layer"))

model.build((None, 3))

print(model.summary())