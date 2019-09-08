# Dependencies
import os
from IPython.display import Image, SVG
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.preprocessing import image
from keras.applications.vgg19 import (VGG19, preprocess_input,
                                      decode_predictions)
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import (VGG19, preprocess_input,
                                                 decode_predictions)
from PIL import Image
import io

# Define image size

#   File "C:\ProgramData\Anaconda3\lib\site-packages\matplotlib\image.py", line 646, in set_data
#     raise TypeError("Invalid dimensions for image data")
# TypeError: Invalid dimensions for image data
image_size = (224, 224)

# https://github.com/jrosebr1/simple-keras-rest-api/issues/5#issuecomment-413461944
# Model
model = VGG19(include_top=True, weights='imagenet')


def dog_predictor():

    # Bring in images
    image_path = os.path.join("Images", "Affenpinscher",
                              "Affenpinscher_233.jpg")
    img = image.load_img(image_path, target_size=image_size)
    plt.imshow(img)

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

    image_path = os.path.join("dog_images_to_predict",
                              "Labrador_retriever_209.jpg")
    predict(image_path)

    # Using model to predict my dogs:
    image_path = os.path.join("dog_images_to_predict", "Kai_02.jpg")
    predict(image_path)

    image_path = os.path.join("dog_images_to_predict", "Samson_07.jpg")
    predict(image_path)


#the function must be at the top level, (aka dedented all the way to the left)
#for you to use it in another file (like app.py)
def predict_from_file(image_file):
    # https://blog.keras.io/building-a-simple-keras-deep-learning-rest-api.html
    pil_image = image_file.read()
    pil_image = Image.open(io.BytesIO(pil_image))
    # if the image mode is not RGB, convert it
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    # resize the input image and preprocess it
    pil_image = pil_image.resize(image_size)
    # print ('pil_image = image.img_to_array(pil_image)')
    pil_image = image.img_to_array(pil_image)
    # print('pil_image = np.expand_dims(pil_image, axis=0)')
    pil_image = np.expand_dims(pil_image, axis=0)
    # print('pil_image = preprocess_input(pil_image)')
    pil_image = preprocess_input(pil_image)
    # print('predictions = model.predict(pil_image)')
    predictions = model.predict(pil_image)
    # I don't think you want this code: `plt.imshow(img)`
    
    # File "C:\ProgramData\Anaconda3\lib\site-packages\matplotlib\image.py", line 646, in set_data
    #     raise TypeError("Invalid dimensions for image data")
    # TypeError: Invalid dimensions for image data
    # 127.0.0.1 - - [07/Sep/2019 22:17:49] "POST /upload HTTP/1.1" 500 -
    # plt.imshow(pil_image)
    decoded_predictions = decode_predictions(predictions, top=3)
    print('Predicted:', decoded_predictions)
    return decoded_predictions
