#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 12:20:48 2019

@author: tirth
"""
from keras.models import load_model
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, cohen_kappa_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools

model_path = "F:\\Project\\Test Model\\Other_model_and_logs\\Best Confusiom Matrix\\best_model_weights_Epoch_61-ValLoss_0.43.h5"

test_data_dir = "/home/ritesh/tirth/Test"
validation_data_dir = "/home/ritesh/tirth/Validation"

img_height, img_width = 512, 512

model = load_model(model_path)

sgd = SGD(lr = 0.0001, momentum = 0.9, decay = 1e-6, nesterov = True)
model.compile(optimizer=sgd, loss='mean_squared_error', metrics=['accuracy'])

test_datagen = ImageDataGenerator(rescale = 1./255)

batch_size = 1
test_generator = test_datagen.flow_from_directory(test_data_dir,									
                                                target_size = (img_height, img_width),
                                                color_mode = 'rgb',
                                                class_mode = 'categorical',
                                                batch_size = batch_size,
                                                shuffle = False)
test_generator.reset()

STEP_SIZE_TEST = test_generator.n // test_generator.batch_size
y_pred = model.predict_generator(test_generator, steps = STEP_SIZE_TEST, verbose = 1)

y_pred = np.argmax(y_pred, axis = 1)

precision = precision_score(test_generator.classes[test_generator.index_array], y_pred, average = 'micro')
recall = recall_score(test_generator.classes[test_generator.index_array], y_pred, average = 'micro')
print("Precision_1: ", precision)
print("Recall_1: ", recall)
    
f1 = f1_score(test_generator.classes[test_generator.index_array], y_pred, average = 'micro')
cohen_kappa = cohen_kappa_score(test_generator.classes[test_generator.index_array], y_pred)
quadratic_kappa = cohen_kappa_score(test_generator.classes[test_generator.index_array], y_pred, weights = 'quadratic')
print("F1: ", f1)
print("Cohen Kappa Score: ", cohen_kappa)
print("Quadratic Kappa Score: ", quadratic_kappa)

cm = confusion_matrix(test_generator.classes[test_generator.index_array], y_pred)

def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=None, normalize=True):
    
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    #plt.show()
    plt.savefig(str(title) + '.png')
    
plot_confusion_matrix(cm           = cm, 
                      normalize    = False,
                      target_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR'],
                      title        = "Confusion Matrix")


            
