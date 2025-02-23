import keras
from keras import layers, activations, optimizers, losses, metrics
import numpy as np

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

# Generating synthetical data.
num_samples = 1280

title_data = np.random.randint(0, 2, size=(num_samples, vocabulary_size))

text_body_data = np.random.randint(0, 2, size=(num_samples, vocabulary_size))

tags_data = np.random.randint(0, 2, size=(num_samples, num_tags))

priority_data = np.random.random(size=(num_samples, 1))

department_data = np.random.randint(0, 2, size=(num_samples, num_departaments))

# Compiling model.
model.compile(
    optimizer=keras.optimizers.RMSprop(),
    loss=[losses.mean_squared_error, losses.categorical_crossentropy],
    metrics=[[metrics.mean_absolute_error], [metrics.Accuracy]]
)

model.fit(
    [title_data, text_body_data, tags_data],
    [priority_data, department_data],
    epochs=1
)

model.evaluate(
    [title_data, text_body_data, tags_data],
    [priority_data, department_data]    
)

priority_preds, department_preds = model.predict(
    [title_data, text_body_data, tags_data]
)

# Plotting the neural network architeture.
keras.utils.plot_model(model, "support_ticket_ranker.png")

keras.utils.plot_model(model, "support_ticket_ranker_with_shapes.png", show_shapes=True)

# Reusing layers on new models.
features = model.layers[4].output

difficulty = layers.Dense(3, activation=activations.softmax, name="difficulty")(features)

new_model = keras.Model(
    inputs=[title, text_body, tags],
    outputs=[priority, department, difficulty]
)

keras.utils.plot_model(new_model, "support_ticket_ranker_with_difficulty_output.png", show_shapes=True)