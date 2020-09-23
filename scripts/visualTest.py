#!/usr/bin/python3
import os, sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import time
import cv2

gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)

# Define Path
model_path = '../models/conv-5-3-64.h5'
model_weights_path = '../models/conv-5-3-64-weights.h5'
test_path = '../samples/visual_test'

# Load the pre-trained models
model = load_model(model_path)
model.load_weights(model_weights_path)

# Define image parameters
img_width, img_height = 32, 32

# Prediction from image
def predict(file):
    x = load_img(file, target_size=(
        img_width, img_height), color_mode='grayscale')
    x = img_to_array(x) / 255
    x = np.array([x])
    # PREDICT
    array = model(x)
    result = array[0]

    # GET TOP 3 GUESSES
    top1 = None
    top2 = None
    top3 = None
    for i, prob in enumerate(result):
        prob = prob.numpy()
        if top1 is None:
            top1 = (i, prob)
        elif prob > top1[1]:
            top2 = top1
            top1 = (i, prob)
        elif top2 is None or prob > top2[1]:
            top2 = (i, prob)
        elif top3 is None or prob > top3[1]:
            top3 = (i, prob)

    print(top1, top2, top3)
    return (top1, top2, top3)

def test(tpath):
    # Walk the directory for every image
    for i, ret in enumerate(os.walk(tpath)):
        for i, filename in enumerate(ret[2]):
            if filename.startswith("."):
                continue

            path = ret[0] + '/' + filename
            print(path)
            result = predict(path)
            letters = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
                    '8': 8, '9': 9, '10': 'A', '11': 'B', '12': 'C', '13': 'D', '14': 'E',
                    '15': 'F', '16': 'G', '17': 'H', '18': 'I', '19': 'J', '20': 'K', '21': 'L',
                    '22': 'M', '23': 'N', '24': 'O', '25': 'P', '26': 'R', '27': 'S', '28': 'T',
                    '29': 'U', '30': 'V', '31': 'Y', '32': 'Z'}
            print(letters[str(result[0][0])], letters[str(
                result[1][0])], letters[str(result[2][0])])
            img = cv2.imread(path)
            cv2.imshow("Image", img)
            cv2.waitKey(0)

if __name__ == '__main__':
    args = sys.argv
    if len(args) > 1:
        test(args[1])
    else:
        test(test_path)