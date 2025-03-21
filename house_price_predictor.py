from tensorflow.keras.datasets import boston_housing
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

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
def _build_model():
    model = keras.Sequential([
        layers.Dense(64, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(1)
    ])

    # Model compiling
    model.compile(
        optimizer="rmsprop",
        loss="mse",
        metrics=["mae"]
    )
    
    return model

def _get_init_of_partition(partition, num_of_samples):
    return partition * num_of_samples

def _get_correct_partition(dataset, actual_partition, num_samples):
    next_partition = actual_partition + 1    

    return dataset[:next_partition * num_samples]

def _k_fold_validation(train_data, train_targets):
    k = 4

    num_val_samples = len(train_data) // k
    
    num_epochs = 1000

    all_scores = []

    all_mse_histories = []

    for partition in range(k):
        print(f"Processing fold #{partition}")

        next_partition = partition + 1

        val_data = _get_correct_partition(train_data, partition, num_val_samples)

        val_targets = _get_correct_partition(train_targets, partition, num_val_samples)

        init_of_partition = _get_init_of_partition(partition, num_val_samples)

        end_of_partition = _get_init_of_partition(next_partition, num_val_samples)

        partial_train_data = np.concatenate([
            train_data[:init_of_partition],
            train_data[end_of_partition:]
        ], axis=0)

        partial_train_targets = np.concatenate(
            [train_targets[:init_of_partition],
            train_targets[end_of_partition:]], axis=0)

        model = _build_model()

        history = model.fit(
            partial_train_data,
            partial_train_targets,
            validation_data=(val_data, val_targets),
            epochs=num_epochs,            
            batch_size=16,
            verbose=0
        )

        mae_history = history.history["val_mae"]

        all_mse_histories.append(mae_history)

        val_mse, val_mae = model.evaluate(
            val_data,
            val_targets,
            verbose=0
        )

        all_scores.append(val_mae)

    return (all_scores, all_mse_histories)

scores, history = _k_fold_validation(train_data, train_targets)

print(scores)

print(np.mean(scores))

average_mae_history = [
    np.mean([x[i] for x in history]) for i in range(1000)
]

average_mae_history = [float(x) for x in average_mae_history]

average_mae_history = average_mae_history[10:] 

print(average_mae_history)

plt.plot(range(1, len(average_mae_history)+1), average_mae_history)

plt.xlabel("Epochs")

plt.ylabel("Validation MAE")

plt.savefig("validation_mae.png")