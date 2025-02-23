from tensorflow import keras
from tensorflow.keras import layers


inputs = keras.Input(shape=(3,), name="my_input")

features = layers.Dense(64, activation="relu", name="first_layer")(inputs)

outputs = layers.Dense(10, activation="softmax", name="last_layer")(features)

model = keras.Model(inputs=inputs, outputs=outputs)

print(model.summary())
