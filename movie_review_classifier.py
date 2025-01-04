from tensorflow.keras.datasets import imdb
import tensorflow as tf

# Num_words = We are selecting olny top 10.000 words that appears more frequently.
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# Listar todos os dispositivos físicos disponíveis
print("Dispositivos físicos:", tf.config.list_physical_devices())

# Listar especificamente as GPUs disponíveis
print("GPUs disponíveis:", tf.config.list_physical_devices('GPU'))