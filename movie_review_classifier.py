from tensorflow.keras.datasets import imdb
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Num_words = We are selecting olny top 10.000 words that appears more frequently.
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

def vectorize_dataset(sequences, dimension=10000):
    # The shape will be length of the dataset and max words that one sample can contain (50000(reviews=rows), 10000(words per review=columns)).
    results = create_np_zero_array_with_dataset_shape(sequences, dimension)

    return set_to_one_index_that_contain_words(results, sequences)        

def create_np_zero_array_with_dataset_shape(dataset, max_words):
    return np.zeros((len(dataset), max_words))

def set_to_one_index_that_contain_words(np_array, dataset):
    for datasetIndex, review in enumerate(dataset):
        for wordIndex in review:
            np_array[datasetIndex, wordIndex] = 1

    return np_array

vectorized_train_data = vectorize_dataset(train_data)

vectorized_test_data = vectorize_dataset(test_data)

vectorized_train_labels = np.asarray(train_labels).astype("float32")

vectorized_test_labels = np.asarray(test_labels).astype("float32")

# Model definition
model = keras.Sequential([
    layers.Dense(128, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

# Model compiling
model.compile(
    optimizer="rmsprop",
    loss="mse",
    metrics=["accuracy"]
)

validation_train_data = vectorized_train_data[:10000]

partial_train_data = vectorized_train_data[10000:]

validation_train_labels = vectorized_train_labels[:10000]

partial_train_labels = vectorized_train_labels[10000:]

# # Execute the training
# train_history = model.fit(
#     partial_train_data,
#     partial_train_labels,
#     epochs=20,
#     batch_size=512,
#     validation_data=(validation_train_data, validation_train_labels
#     )
# )

train_history = model.fit(
    vectorized_train_data,
    vectorized_train_labels,
    epochs=4,
    batch_size=512,
)

# train_history_dict = train_history.history

# # Plotting train history data
# loss_values = train_history_dict["loss"]

# validation_loss_values = train_history_dict["val_loss"]

# epochs = range(1, len(loss_values) + 1)

# plt.plot(epochs, loss_values, "bo", label="Training loss")

# plt.plot(epochs, validation_loss_values, "b", label="Validation loss")

# plt.title("Training and validation loss")

# plt.xlabel("Epochs")

# plt.ylabel("Loss")

# plt.legend()

# plt.savefig("training_validation_loss.png")

# # Plotting train accuracy

# plt.clf()

# accuracy = train_history_dict["accuracy"]

# validation_accuracy = train_history_dict["val_accuracy"]

# plt.plot(epochs, accuracy, "bo", label="Accuracy")

# plt.plot(epochs, validation_accuracy, "b", label="Validation Accuracy")

# plt.title("Training and validation accuracy")

# plt.xlabel("Epochs")

# plt.ylabel("Accuracy")

# plt.legend()

# plt.savefig("training_accuracy.png")


results = model.evaluate(vectorized_test_data, vectorized_test_labels)

print(results)






