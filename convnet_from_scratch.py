import keras
from keras import layers, activations, optimizers, losses, utils, callbacks

import os, shutil, pathlib

original_dir = pathlib.Path(r"/home/ai/PetImages")

new_base_dir = pathlib.Path(r"/home/ai/cats_vs_dogs_small")

def make_subset(subset_name, start_index, end_index):
    for category in ("Cat", "Dog"):
        dir = new_base_dir / subset_name / category

        os.makedirs(dir)

        fnames = [f"{i}.jpg"
                    for i in range(start_index, end_index)]
        
        for fname in fnames:
            shutil.copyfile(src=original_dir / category / fname, dst=dir / fname)

# make_subset("train", start_index=0, end_index=1000)

# make_subset("validation", start_index=1000, end_index=1500)

# make_subset("test", start_index=1500, end_index=2500)

inputs = keras.Input(shape=(180,180,3))

x = layers.Rescaling(1./255)(inputs)
x = layers.Conv2D(filters=32, kernel_size=3, activation=activations.relu)(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=64, kernel_size=3, activation=activations.relu)(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=128, kernel_size=3, activation=activations.relu)(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=256, kernel_size=3, activation=activations.relu)(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=256, kernel_size=3, activation=activations.relu)(x)
x = layers.Flatten()(x)

outputs = layers.Dense(1, activation=activations.sigmoid)(x)

model = keras.Model(inputs=inputs, outputs=outputs)

print(model.summary())

model.compile(optimizer=optimizers.RMSprop(),
              loss=losses.binary_crossentropy,
              metrics=["accuracy"])

train_dataset = utils.image_dataset_from_directory(
    new_base_dir / "train",
    image_size=(180, 180),
    batch_size=32
)

validation_dataset = utils.image_dataset_from_directory(
    new_base_dir / "validation",
    image_size=(180, 180),
    batch_size=32,

)

test_dataset = utils.image_dataset_from_directory(
    new_base_dir / "test",
    image_size=(180, 180),
    batch_size=32
)

for data_batch, labels_batch in train_dataset:
    print("data batch shape:", data_batch.shape)

    print("labels batch shape:", labels_batch.shape)

    break


callbacks_arr = [
    callbacks.ModelCheckpoint(
        filepath="convnet_from_scratch.keras",
        save_best_only=True,
        monitor="val_loss"
    )
]

history = model.fit(
    x=train_dataset,
    epochs=30,
    validation_data=validation_dataset,
    callbacks=callbacks_arr
)