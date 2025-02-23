import keras
from keras import layers,activations

vocabulary_size = 10000

num_tags = 100

num_departaments = 4

# Defining the input data to our neural network.
title = keras.Input(shape=(vocabulary_size,), name="title")

text_body = keras.Input(shape=(vocabulary_size,), name="text_body")

tags = keras.Input(shape=(num_tags,), name="tags")

# Merging the symboluc tensors in a richer feature.
features = layers.Concatenate()([title, text_body, tags])

features = layers.Dense(64, activation=activations.relu)(features)

# Defining the model outputs.
priority = layers.Dense(1, activation=activations.sigmoid, name="priority")(features)

department = layers.Dense(
    num_departaments, activation=activations.softmax, name="department"
)(features)

# Creating the model.
model = keras.Model(
    inputs=[title, text_body, tags],
    outputs=[priority, department])

print(model.summary())



