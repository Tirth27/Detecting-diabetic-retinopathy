from keras.models import load_model
from keras.preprocessing import image
from keras.utils import normalize
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from skimage import io
import numpy as np
from scipy.misc import imresize, imread
import cv2
import os
import json, codecs
import matplotlib.pyplot as plt


def prediction(saved_weights, image_path):
    saved_weights = saved_weights
    #saved_weights = '/Users/tirth/Documents/Diabetic Retinopathy/saved_weights/Other_model_and_logs/Model_Arch_Change/best_model/best_model_weights_Epoch_11-ValLoss_0.94.h5'
    save_model = load_model(saved_weights)

    sgd = SGD(lr = 0.0003, momentum = 0.9, decay = 0.0, nesterov = True)
    save_model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    img_dir = img_dir
    #img_dir = r'/Users/tirth/Documents/Diabetic Retinopathy/How to predict new input(Mnist)/dr_image/'
    test_datagen = ImageDataGenerator(rescale = 1./255.)
    batch_size = 1
    test_generator = test_datagen.flow_from_directory(directory = img_dir, target_size = (256, 256),
                                        color_mode = 'rgb', class_mode = 'categorical',
                                        batch_size = batch_size, seed = 42, shuffle = False)
    test_generator.reset()
    STEP_SIZE_TEST = test_generator.n // test_generator.batch_size
    y_pred = save_model.predict_generator(test_generator, verbose = 1)
    print('Gene:- ',y_pred)
    y_pred_max = np.argmax(y_pred, axis = 1)
    print("Argmax:- ", y_pred_max)

    return y_pred


#1
#imgs = io.imread(img_path)
img = image.load_img(img_path)
img = image.img_to_array(img)
#cv2.imshow('Image',img)
img = np.expand_dims(img, axis = 0)
print('Original image shape:- ', img.shape)

#2
img2 = imread(img_path)
#img2 = normalize(img2, axis = -1)
#img = imresize(img, (28, 28))
#img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
print('Image shape after resize-',img2.shape)
#cv2.imshow('Image',img2)
img2 = np.reshape(img2, [1, 256, 256, 3])

img_dir = r'/Users/tirth/Documents/Diabetic Retinopathy/How to predict new input(Mnist)/dr_image/'
test_datagen = ImageDataGenerator(rescale = 1./255.)
batch_size = 1
test_generator = test_datagen.flow_from_directory(directory = img_dir, target_size = (256, 256),
                                    color_mode = 'rgb', class_mode = 'categorical',
                                    batch_size = batch_size, seed = 42, shuffle = False)
test_generator.reset()
STEP_SIZE_TEST = test_generator.n // test_generator.batch_size
y_pred = save_model.predict_generator(test_generator, verbose = 1)
print('Gene:- ',y_pred)
y_pred = np.argmax(y_pred, axis = 1)
print("Argmax:- ", y_pred)


#Predict
result = save_model.predict(img)
result = np.array_str(np.argmax(result, axis=1))
print('Predict Class:- ',result)

prob = save_model.predict_proba(img2)
print('predict probability:- ',prob)

cl = save_model.predict_classes(img2)
print('predict Class(Method):-',cl)
