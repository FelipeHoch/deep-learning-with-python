import os
from matplotlib import pyplot 
import keras
from keras import utils, layers, activations, optimizers, losses, callbacks
import numpy as np
import random

input_dir = r"/home/ai/images/"

target_dir = r"/home/ai/annotations/trimaps/"

input_img_paths = sorted(
    [os.path.join(input_dir, fname)
        for fname in os.listdir(input_dir)
            if fname.endswith(".jpg")
    ]
)

target_paths = sorted(
    [os.path.join(target_dir, fname)
        for fname in os.listdir(target_dir)
            if fname.endswith(".png") and not fname.startswith(".")
    ]
)

pyplot.axis("off")

pyplot.imshow(utils.load_img(input_img_paths[9]), url="./")

img_size = (200, 200)

num_imgs = len(input_img_paths)

random.Random(1337).shuffle(input_img_paths)
random.Random(1337).shuffle(target_paths)

def path_to_input_image(path):
    return utils.img_to_array(utils.load_img(path, target_size=img_size))

def path_to_targey(path):
    img = utils.img_to_array(utils.load_img(path, target_size=img_size, color_mode="grayscale"))

    img = img.astype("uint8")

    return img

input_imgs = np.zeros((num_imgs,) + img_size + (3,), dtype="float32")

targets = np.zeros((num_imgs,) + img_size + (1,), dtype="uint8")

for i in range(num_imgs):
    input_imgs[i] = path_to_input_image(input_img_paths[i])

    targets[i] = path_to_targey(target_paths[i])

num_val_samples = 1000

tran_input_imgs = input_imgs[:-num_val_samples]

train_targets = target_paths[:-num_val_samples]

val_input_imgs = input_imgs[-num_val_samples]

val_targets = targets[-num_val_samples:]

def get_model(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (3,))

    x = layers.Rescaling(1./255)(inputs)

    x = layers.Conv2D(64, 3, strides=2, activation=activations.relu, padding="same")(x)
    x = layers.Conv2D(64, 3, activation=activations.relu, padding="same")(x)
    x = layers.Conv2D(128, 3, strides=2, activation=activations.relu, padding="same")(x)
    x = layers.Conv2D(128, 3, activation=activations.relu, padding="same")(x)
    x = layers.Conv2D(256, 3, strides=2, activation=activations.relu, padding="same")(x)
    x = layers.Conv2D(256, 3, activation=activations.relu, padding="same")(x)

    x = layers.Conv2DTranspose(256, 3, activation=activations.relu, padding="same")(x)
    x = layers.Conv2DTranspose(256, 3, activation=activations.relu, padding="same", strides=2)(x)
    x = layers.Conv2DTranspose(128, 3, activation=activations.relu, padding="same")(x)
    x = layers.Conv2DTranspose(128, 3, activation=activations.relu, padding="same", strides=2)(x)
    x = layers.Conv2DTranspose(64, 3, activation=activations.relu, padding="same")(x)
    x = layers.Conv2DTranspose(64, 3, activation=activations.relu, padding="same", strides=2)(x)

    outputs = layers.Conv2D(num_classes, 3, activation=activations.softmax, padding="same")(x)

    model = keras.Model(inputs, outputs)

    return model

model = get_model(img_size=img_size, num_classes=3)

print(model.summary())

model.compile(optimizer=optimizers.RMSprop(), loss=losses.sparse_categorical_crossentropy)

callbacks_arr = [
    callbacks.ModelCheckpoint("oxford_segmentation.keras", save_best_only=True)
]

history = model.fit(
    tran_input_imgs,
    train_targets,
    epochs=50,
    callbacks=callbacks_arr,
    batch_size=64,
    validation_data=(val_input_imgs, val_targets)
)
