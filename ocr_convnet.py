import keras
from keras import layers, activations, datasets, optimizers, losses


inputs = keras.Input(shape=(28,28,1))

x = layers.Conv2D(filters=32, kernel_size=3, activation=activations.relu)(inputs)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=64, kernel_size=3, activation=activations.relu)(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=128, kernel_size=3, activation=activations.relu)(x)
x = layers.Flatten()(x)

outputs = layers.Dense(10, activation=activations.softmax)(x)

model = keras.Model(inputs=inputs, outputs=outputs)

print(model.summary())

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype("float32") / 255

test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype("float32") / 255

model.compile(optimizer=optimizers.RMSprop(),
              loss=losses.sparse_categorical_crossentropy,
              metrics=["accuracy"])

model.fit(train_images, train_labels, epochs=5, batch_size=64)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print(f"Acc: {test_acc:.3f}")