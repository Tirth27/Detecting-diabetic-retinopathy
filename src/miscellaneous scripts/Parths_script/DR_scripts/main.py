#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 16:21:55 2019

@author: tirth
"""
import Model_Ben_Parth as modl

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import regularizers
from keras.utils import normalize, np_utils, multi_gpu_model, to_categorical
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score, confusion_matrix
from sklearn.utils import class_weight

# Reproduce Result
#np.random.seed(1337)
#tf.set_random_seed(1337)

def save_model(model, epoch, score, model_name):
    if score >= 0.75:
        print('Saving Model')
        model.save('saved_model/' + model_name + '_Epoch_' + str(epoch) + '_recall_' + str(round(score, 4)) + '.h5')
    else:
        print("Model Score is Low, Score:-", score)
        model.save('saved_model/' + model_name + '_Epoch_' + str(epoch) + '_LowScore_' + '_recall_' + str(round(score, 4)) + '.h5')

if __name__ == '__main__':

    #Hyperparameters.
    img_height, img_width = 512, 512
    channels = 3
    nb_classes = 5

    nb_gpus = 2
    nb_epoch = 50
    #batch_size = 32 #default
    batch_size = 6

    ## Value of alpha for LeakyReLU
    #leakiness = 0.5 #JeffreyDf
    leakiness = 0.01

    # Value of lamda for regularizers
    w_lambdaa = 0.0005 #Weights Regularizer
    b_lambdaa = 0.0005 #Biases Regularizer
    w_regu = regularizers.l2(w_lambdaa)
    b_regu = regularizers.l2(b_lambdaa)
    #w_regu = None
    #b_regu = None

    #Fractional MaxPool
    pooling_ratio = [1.0, 1.44, 1.73, 1.0]

    # Type of kernel_initializer(https://faroit.github.io/keras-docs/1.2.2/initializations/)
    #initializer = 'golort_normal' # Default -> 'glorot_uniform'
    #initializer = 'he_normal'
    initializer = 'glorot_uniform'

    #Test/Train Directory path
    train_data_dir = r"D:\Final Year Project\crops\denoise_512"
    train_labels = r"D:\Final Year Project\label\trainlabel_master_v2.csv"

    test_data_dir = r"D:\Final Year Project\crops\denoise_test_512"
    test_labels = r"D:\Final Year Project\label\testLables.csv"

    #Saved Model Name
    model_name = 'Model_Ben_rmBoundary512'

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
                     img_width = img_width, labels_train = labels_train, train_data_dir = train_data_dir,
                    pooling_ratio = pooling_ratio)

    test_datagen = ImageDataGenerator(rescale = 1./255.)
    
    batch_size = 1
    test_generator = test_datagen.flow_from_dataframe(labels_test, test_data_dir,
									x_col = 'image', y_col = 'level',
                                    has_ext = False, target_size = (img_height, img_width),
                                    color_mode = 'rgb', class_mode = 'categorical',
                                    batch_size = batch_size,
#                                                      seed = 42,
                                                      shuffle = False)

    STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size

    loss, accuracy = model.evaluate_generator(validation_generator, steps = STEP_SIZE_TRAIN, verbose = 1)
    print("Test Loss: ", loss)
    print("Test Accuracy: ", accuracy)

    #Generates output predictions for input samples
    #batch_size = 1
    test_generator.reset()
    STEP_SIZE_TEST = test_generator.n // test_generator.batch_size
    y_pred = model.predict_generator(test_generator, steps = STEP_SIZE_TEST, verbose = 1)
    #y_pred = model.predict_generator(test_generator, verbose = 1)

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
    save_model(model = model, epoch = nb_epoch, score = recall, model_name = model_name)

    #The best value is 1 and the worst value is 0
    f1 = f1_score(y_test, y_pred, average = 'micro')
    #cohen_kappa is level of agreement between two annotators on a classification problem.
    cohen_kappa = cohen_kappa_score(y_test, y_pred)
    quadratic_kappa = cohen_kappa_score(y_test, y_pred, weights = 'quadratic')

    print("F1: ", f1)
    print("Cohen Kappa Score: ", cohen_kappa)
    print("Quadratic Kappa Score: ", quadratic_kappa)
    
    #Calculate and Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    modl.plot_confusion_matrix(cm  = cm, 
                      normalize    = False,
                      target_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR'],
                      title        = "Confusion_Matrix/Confusion Matrix")
    #Normalized Confusion Matrix.
    modl.plot_confusion_matrix(cm  = cm, 
                      normalize    = True,
                      target_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR'],
                      title        = "Confusion_Matrix/Confusion Matrix(Normalized)")

    print("Completed")
