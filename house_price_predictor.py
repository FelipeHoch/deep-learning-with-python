from tensorflow.keras.datasets import boston_housing

# Handling the data points
def _z_score_normalization(dataset):
    mean = dataset.mean(axis=0)    

    dataset -= mean

    standard_deviation = dataset.std(axis=0)

    dataset /= standard_deviation

    return dataset

(train_data, train_targets), (test_data, test_targets) = (boston_housing.load_data())

train_data = _z_score_normalization(train_data)

test_data = _z_score_normalization(test_data)

# Handling the neural network definitions

