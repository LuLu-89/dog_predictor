# Dependencies
import os
from IPython.display import Image, SVG
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.preprocessing import image
from keras.applications.vgg19 import (
    VGG19,
    preprocess_input,
    decode_predictions
)
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import (
    VGG19,
    preprocess_input,
    decode_predictions
)


def dog_predictor():

    # Define image size
    image_size = (224, 224)


    # Bring in images
    image_path = os.path.join("Images", "Affenpinscher", "Affenpinscher_233.jpg")
    img = image.load_img(image_path, target_size=image_size)
    plt.imshow(img)

    # Model
    model = VGG19(include_top=True, weights='imagenet')

    # Preprocess image for model prediction
    # This step handles scaling and normalization for VGG19

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Make predictions

    predictions = model.predict(x)
    print('Predicted:', decode_predictions(predictions, top=3))
    plt.imshow(img)

    # Refactor above steps into reusable function


    def predict(image_path):
        """Use VGG19 to label image"""
        img = image.load_img(image_path, target_size=image_size)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        predictions = model.predict(x)
        plt.imshow(img)
        print('Predicted:', decode_predictions(predictions, top=3))


    # Predicting some pure breeds to test model:
    image_path = os.path.join("dog_images_to_predict", "Boxer_69.jpg")
    predict(image_path)

    image_path = os.path.join("dog_images_to_predict", "Pembroke_209.jpg")
    predict(image_path)

    image_path = os.path.join("dog_images_to_predict", "Chihuahua_2973.jpg")
    predict(image_path)

    image_path = os.path.join("dog_images_to_predict", "Labrador_retriever_57.jpg")
    predict(image_path)

    # Using model to predict my dogs:
    image_path = os.path.join("dog_images_to_predict", "Kai_02.jpg")
    predict(image_path)

    image_path = os.path.join("dog_images_to_predict", "Samson_02.jpg")
    predict(image_path)
