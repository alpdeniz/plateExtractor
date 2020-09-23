#!/usr/bin/python3
import sys
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dropout, Flatten, Dense, Activation, BatchNormalization
from tensorflow.keras.layers import MaxPooling2D, Conv2D
from tensorflow.keras import callbacks
from datetime import datetime
import time

gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)

"""
Parameters
"""
epochs = 12
img_width, img_height = 32, 32
batch_size = 16
steps_per_epoch = 451
validation_steps = 50
nb_filters1 = 32
nb_filters2 = 64
classes_num = 33
lr = 0.001

# letter locations
train_data_path = '../samples/train'
validation_data_path = '../samples/valid'
test_data_path = '../samples/test'


# The KERAS Model
def createModel(nb_filters1=64, nb_filters2=128):
    model = Sequential()

    # First Convolution Layer
    model.add(Conv2D (nb_filters1, kernel_size=5, activation='relu', input_shape=(img_width, img_height, 1)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))

    # Second Convolution Layer
    model.add(Conv2D (nb_filters2, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    # 2D to 1D
    model.add(Flatten())
    # Output Layer
    model.add(Dense(classes_num, activation='softmax'))

    model.compile(loss='categorical_crossentropy',# tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                optimizer=optimizers.RMSprop(lr=lr), #optimizers.SGD(lr=lr,momentum=0.1), 
                metrics=[tf.keras.metrics.CategoricalAccuracy()])

    return model


def lr_schedule(epoch):
    lrate = 0.001
    if epoch > 1:
        lrate = 0.0005
    if epoch > 2:
        lrate = 0.0001
    if epoch > 3:
        lrate = 0.00005
    if epoch > 6:
        lrate = 0.00001
    return lrate


def setupData():
    train_datagen = ImageDataGenerator(rescale=1. / 255,
        rotation_range=8,
        zoom_range=0.02,
        shear_range=8,
        fill_mode="nearest"
    )
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_path,
        classes=['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','R','S','T','U','V','Y','Z'],
        target_size=(img_height, img_width),
        batch_size=batch_size,
        color_mode='grayscale',
        # class_mode='sparse'
    )
    print(train_generator.class_indices)
    print(train_generator.allowed_class_modes)

    validation_generator = test_datagen.flow_from_directory(
        validation_data_path,
        classes=['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','R','S','T','U','V','Y','Z'],
        target_size=(img_height, img_width),
        batch_size=batch_size,
        color_mode='grayscale',
        # class_mode='sparse'
    )

    test_generator = test_datagen.flow_from_directory(
        test_data_path,
        classes=['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','R','S','T','U','V','Y','Z'],
        target_size=(img_height, img_width),
        batch_size=batch_size,
        color_mode='grayscale',
        # class_mode='sparse'
    )

    return (train_generator, validation_generator, test_generator)


def train(model, data):

    ## EARLY STOPPING 
    es_cb = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0, patience=3, verbose=1, mode='auto',
        baseline=None, restore_best_weights=True
    )

    model.fit(
        data[0],
        # steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=data[1],
        callbacks=[es_cb, tf.keras.callbacks.LearningRateScheduler(lr_schedule)]
        # validation_steps=validation_steps
    )
    return model

def testModel(model, testData):
    model.evaluate(testData)

def saveModel(model):
    target_dir = '../models/'
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    model.save('../models/model.h5')
    model.save_weights('../models/weights.h5')

def printModelSummary(model):
    print(model.summary())


if __name__ == '__main__':
    start = time.time()
    
    model = createModel()
    printModelSummary(model)
    data = setupData()
    model = train(model, data)
    testModel(model, data[2])
    saveModel(model)

    #Calculate execution time
    dur = time.time() - start
    print("Execution Time:", dur, "seconds")