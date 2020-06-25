#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 16:21:55 2019

@author: tirth
"""
import Model as modl

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import regularizers
from keras.utils import normalize, np_utils, multi_gpu_model, to_categorical
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score
from sklearn.utils import class_weight

# Reproduce Result
np.random.seed(1337)
tf.set_random_seed(1337)

def save_model(model, score, model_name):
    if score >= 0.75:
        print('Saving Model')
        model.save('saved_model/' + model_name + '_recall_' + str(round(score, 4)) + '.h5')
    else:
        print("Model Score is Low, Score:-", score)
        model.save('saved_model/' + model_name + '_LowScore_' +'_recall_' + str(round(score, 4)) + '.h5')

if __name__ == '__main__':

    #Hyperparameters.
    img_height, img_width = 256, 256
    channels = 3
    nb_classes = 5

    nb_gpus = 2
    nb_epoch = 1
    #batch_size = 32 #default
    batch_size = 15

    ## Value of alpha for LeakyReLU
    #leakiness = 0.5 #JeffreyDf
    leakiness = 0.01

    # Value of lamda for regularizers
    w_lambdaa = 0.01 #Weights Regularizer
    b_lambdaa = 0.01 #Biases Regularizer
    w_regu = regularizers.l2(w_lambdaa)
    b_regu = regularizers.l2(b_lambdaa)

    # Type of kernel_initializer(https://faroit.github.io/keras-docs/1.2.2/initializations/)
    #initializer = 'golort_normal' # Default -> 'glorot_uniform'
    initializer = 'he_normal'

    #Test/Train Directory path
    train_data_dir = '/Users/tirth/Documents/Diabetic Retinopathy/Model/Sample Dataset'
    train_labels = '/Users/tirth/Documents/Diabetic Retinopathy/Model/sample.csv'

    test_data_dir = '/Users/tirth/Documents/Diabetic Retinopathy/Model/Sample Dataset'
    test_labels = '/Users/tirth/Documents/Diabetic Retinopathy/Model/sample.csv'

    #Saved Model Name
    model_name = 'CNN_Model_DR'

    #Labels
    labels_train = pd.read_csv(train_labels)
    #labels_train = labels_train.loc[:,('level')]
    y_train = np.array(labels_train['level'])

    labels_test = pd.read_csv(test_labels)
    y_test = np.array(labels_test['level'])
    y_test = to_categorical(y_test, nb_classes)

    #Class Weights (For imbalanced classes)
    cl_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)

    #Train
    model, validation_generator, train_generator = \
						modl.cnn_model(channels = channels, nb_epoch = nb_epoch,
                     batch_size = batch_size, nb_classes = nb_classes, nb_gpus = nb_gpus,
                     cl_weights = cl_weights, leakiness = leakiness, w_regu = w_regu,
                     b_regu = b_regu, initializer = initializer, img_height = img_height,
                     img_width = img_width, labels_train = labels_train, train_data_dir = train_data_dir)

    test_datagen = ImageDataGenerator(rescale = 1./255.)

    batch_size = 1
    test_generator = test_datagen.flow_from_dataframe(labels_test, test_data_dir,
									x_col = 'train_image_name', y_col = 'level',
                                    has_ext = True, target_size = (img_height, img_width),
                                    color_mode = 'rgb', class_mode = 'categorical',
                                    batch_size = batch_size, seed = 42, shuffle = False)

    STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size

    score = model.evaluate_generator(validation_generator, steps = STEP_SIZE_TRAIN, verbose = 1)
    print("Test score: ", score[0])
    print("Test Accuracy: ", score[1])

    #Generates output predictions for input samples
    test_generator.reset()
    STEP_SIZE_TEST = test_generator.n // test_generator.batch_size
    #y_pred = model.predict_generator(test_generator, steps = STEP_SIZE_TEST, verbose = 1)
    y_pred = model.predict_generator(test_generator, verbose = 1)

    y_test = np.argmax(y_test, axis = 1)
    y_pred = np.argmax(y_pred, axis = 1)

    ##The best value is 1 and the worst value is 0
    #Recall is intuitively the ability of classifer not to label as postive a sample that is negative.
    precision = precision_score(y_test, y_pred, average = 'micro')
    #Recall is intuitively the ability of classifer to find all postive samples.
    recall = recall_score(y_test, y_pred, average = 'micro')
    print("Precision: ", precision)
    print("Recall: ", recall)

    #save model
    print("Saving Model.")
    save_model(model = model, score = recall, model_name = model_name)

    #The best value is 1 and the worst value is 0
    f1 = f1_score(y_test, y_pred, average = 'micro')
    #cohen_kappa is level of agreement between two annotators on a classification problem.
    cohen_kappa = cohen_kappa_score(y_test, y_pred)
    cohen_kappa_qua = cohen_kappa_score(y_test, y_pred, weights = 'quadratic')
    #quad_kappa = kappa(y_test, y_pred, weights='quadratic')

    print("F1: ", f1)
    print("Cohen Kappa Score: ", cohen_kappa)
    print("Cohen Kappa Score(Weights:- Quadratic): ", cohen_kappa_qua)
    #print("Quadratic Kappa: ", quad_kappa)

    print("Completed")
