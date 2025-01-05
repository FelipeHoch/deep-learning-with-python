from tensorflow.keras.datasets import imdb
import tensorflow as tf
import numpy as np

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
