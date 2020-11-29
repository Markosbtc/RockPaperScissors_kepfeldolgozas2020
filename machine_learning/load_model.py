import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image

labels = ["Rock", "Paper", "Scissors"]

def get_knn_result(idx):
    knn_result = {
        0: 1,
        1: 0,
        2: 2,
    }
    return knn_result.get(idx)

def image_to_array(file_path):
    img = Image.open(file_path)
    img = img.resize((40, 60))
    data = np.asarray(img, dtype='float32')
    return data


def load_model(model_file_name):
    print("Load neural network model")
    return tf.keras.models.load_model(model_file_name)

def make_prediction(img, model):
    expanded_img = np.expand_dims(img, 0)

    values = ImageDataGenerator(rescale=1.0/255).flow(expanded_img,
                                                      y=None,
                                                      batch_size=32,
                                                      subset=None)

    result = model.predict(values)
    result = result.argmax(axis=-1)
    sorted_labels = sorted(labels)
    predicted_label = sorted_labels[result[0]]
    print("Predicted label: ", predicted_label)
    return result[0]


def get_prediction(image_path):
    print("Get neural network prediction")
    model = load_model("conv_model.h5")
    img = image_to_array(image_path)
    prediction = make_prediction(img, model)
    return get_knn_result(prediction)
