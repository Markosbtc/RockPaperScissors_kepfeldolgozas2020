import tensorflow as tf
import keras_preprocessing
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os
import numpy

TRAINING_DIR = "..\\datasets\\1"

# rescale: multiply the data by the value provided (after applying all other transformations).
# rotation_range: Degree range for random rotations.
# width_shift_range: shift the image to the left or right(horizontal shifts, the percentage of total width as range)
# height_shift_range: shift vertically (up or down)
# zoom_range: Range for random zoom
# horizontal_flip: Randomly flip inputs horizontally.
# validation_split: Fraction of images reserved for validation (strictly between 0 and 1).
train_datagen = ImageDataGenerator(rescale=1.0/255,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   validation_split=0.2)

train_generator = train_datagen.flow_from_directory(TRAINING_DIR,
                                                    batch_size=32,
                                                    target_size=(40, 60),
                                                    classes=["paper", "rock", "scissors"],
                                                    subset='training')

imgs, labels = next(train_generator)
plt.imshow(imgs[0])
plt.axis('Off')
plt.show()

validation_datagen = ImageDataGenerator(rescale=1.0/255,
                                        validation_split=0.2)

validation_generator = validation_datagen.flow_from_directory(TRAINING_DIR,
                                                              batch_size=32,
                                                              target_size=(40, 60),
                                                              classes=["paper", "rock", "scissors"],
                                                              subset='validation')

imgs, labels = next(validation_generator)
plt.imshow(imgs[0])
plt.axis('Off')
plt.show()

model = tf.keras.models.Sequential([
    # 40x60 images, 3 bytes of RGB color
    tf.keras.layers.Conv2D(64, (5,5), activation=tf.nn.relu,input_shape=(40, 60, 3)),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(64, (3,3), activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.Dense(3, activation = tf.nn.softmax) # 3 classes
])

model.summary()

model.compile(loss = 'categorical_crossentropy', optimizer= tf.keras.optimizers.Adam(), metrics=['acc'])

history = model.fit(train_generator,
                    epochs = 25,
                    verbose = 1,
                   validation_data = validation_generator)

model.save("conv_model.h5")
