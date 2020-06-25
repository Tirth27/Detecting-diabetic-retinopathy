#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 16:22:32 2019

@author: tirth
"""
import os
import math
import numpy as np
import tensorflow as tf
from keras.callbacks import Callback
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.callbacks import TensorBoard, EarlyStopping, LearningRateScheduler
from keras import regularizers
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Reproduce Result
np.random.seed(1337)
tf.set_random_seed(1337)

# learning rate schedule
def step_decay(nb_epoch):
	initial_lr = 0.1
	drop = 0.5
	epochs_drop = 10.0
	lrate = initial_lr * math.pow(drop, math.floor((1+nb_epoch)/epochs_drop))
	print('lr: ',lrate)
	return lrate

# learning rate schedule
class LRFinder(Callback):
    '''
    A simple callback for finding the optimal learning rate range for your model + dataset.

    # Usage
        ```python
            lr_finder = LRFinder(min_lr=1e-5,
                                 max_lr=1e-2,
                                 steps_per_epoch=np.ceil(epoch_size/batch_size),
                                 epochs=3)
            model.fit(X_train, Y_train, callbacks=[lr_finder])

            lr_finder.plot_loss()
        ```

    # Arguments
        min_lr: The lower bound of the learning rate range for the experiment.
        max_lr: The upper bound of the learning rate range for the experiment.
        steps_per_epoch: Number of mini-batches in the dataset. Calculated as `np.ceil(epoch_size/batch_size)`.
        epochs: Number of epochs to run experiment. Usually between 2 and 4 epochs is sufficient.

    # References
        Blog post: jeremyjordan.me/nn-learning-rate
        Original paper: https://arxiv.org/abs/1506.01186
    '''

    def __init__(self, min_lr=1e-5, max_lr=1e-2, steps_per_epoch=None, epochs=None):
        super().__init__()

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.total_iterations = steps_per_epoch * epochs
        self.iteration = 0
        self.history = {}

    def clr(self):
        '''Calculate the learning rate.'''
        x = self.iteration / self.total_iterations
        l_r = self.min_lr + (self.max_lr-self.min_lr) * x
        print('lr: ', l_r)
        return l_r

    def on_train_begin(self, logs=None):
        '''Initialize the learning rate to the minimum value at the start of training.'''
        logs = logs or {}
        K.set_value(self.model.optimizer.lr, self.min_lr)

    def on_batch_end(self, epoch, logs=None):
        '''Record previous batch statistics and update the learning rate.'''
        logs = logs or {}
        self.iteration += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.iteration)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        K.set_value(self.model.optimizer.lr, self.clr())

    def plot_lr(self):
        '''Helper function to quickly inspect the learning rate schedule.'''
        plt.plot(self.history['iterations'], self.history['lr'])
        plt.yscale('log')
        plt.xlabel('Iteration')
        plt.ylabel('Learning rate')
        plt.show()

    def plot_loss(self):
        '''Helper function to quickly observe the learning rate experiment results.'''
        plt.plot(self.history['lr'], self.history['loss'])
        plt.xscale('log')
        plt.xlabel('Learning rate')
        plt.ylabel('Loss')
        plt.show()

# step decay schedule
def step_decay_schedule(initial_lr=1e-3, decay_factor=0.75, step_size=10):
    '''
    Wrapper function to create a LearningRateScheduler with step decay schedule.
    '''
    def schedule(epoch):
        sd = initial_lr * (decay_factor ** np.floor(epoch/step_size))
        print('sd: ', sd)
        return sd

    return LearningRateScheduler(schedule)


def cnn_model(channels, nb_epoch, batch_size, nb_classes, nb_gpus, cl_weights,
              leakiness, w_regu, b_regu, initializer, img_height, img_width,
              labels_train, train_data_dir):

    print("Class Weights:- ", cl_weights, "\n # of Classes:- ", nb_classes)
    print("Channels:- ", channels, "\n # of gpus:- ", nb_gpus)
    print("Batch Size:- ", batch_size, "\n # of epoch:- ", nb_epoch)
    print("Bias regularizers:- ", b_regu, "\nInitializer:- ", initializer)
    print("Leakiness:- ", leakiness, "\nWeight regularizers:- ", w_regu)
    print("Image Height:- ", img_height, "\nImage Width:- ", img_width)

    if not os.path.exists('saved_model'):
        os.makedirs('saved_model')
        print("Saved Model Folder Created!")

    model = Sequential()
    model.add(Conv2D(32, (3,3), padding = 'same', strides = (1,1),
                        input_shape = (img_height, img_width, channels),
                        kernel_initializer = initializer,
                        kernel_regularizer = w_regu,
                        bias_regularizer = b_regu))
    model.add(LeakyReLU(alpha = leakiness))
    model.add(BatchNormalization(axis = -1))
    model.add(Conv2D(32, (3, 3), strides = (1, 1), padding = 'same',
                    kernel_initializer = initializer,
                    kernel_regularizer = w_regu,
                    bias_regularizer = b_regu))
    model.add(LeakyReLU(alpha = leakiness))
    model.add(BatchNormalization(axis = -1))
    model.add(MaxPooling2D(pool_size = (3, 3), strides = (2, 2)))

    model.add(Conv2D(64, (3, 3), strides = (1, 1), padding = 'same',
                    kernel_initializer = initializer,
                    kernel_regularizer = w_regu,
                    bias_regularizer = b_regu))
    model.add(LeakyReLU(alpha = leakiness))
    model.add(BatchNormalization(axis = -1))
    model.add(Conv2D(64, (3, 3), strides = (1, 1), padding = 'same',
                    kernel_initializer = initializer,
                    kernel_regularizer = w_regu,
                    bias_regularizer = b_regu))
    model.add(LeakyReLU(alpha = leakiness))
    model.add(BatchNormalization(axis = -1))
    model.add(MaxPooling2D(pool_size = (3, 3), strides = (2, 2)))

    model.add(Conv2D(128, (3, 3), strides = (1, 1), padding = 'same',
                    kernel_initializer = initializer,
                    kernel_regularizer = w_regu,
                    bias_regularizer = b_regu))
    model.add(LeakyReLU(alpha = leakiness))
    model.add(BatchNormalization(axis = -1))
    model.add(Conv2D(128, (3, 3), strides = (1, 1), padding = 'same',
                    kernel_initializer = initializer,
                    kernel_regularizer = w_regu,
                    bias_regularizer = b_regu))
    model.add(LeakyReLU(alpha = leakiness))
    model.add(BatchNormalization(axis = -1))
    model.add(MaxPooling2D(pool_size = (3, 3), strides = (2, 2)))

    model.add(Conv2D(256, (3, 3), strides = (1, 1), padding = 'same',
                    kernel_initializer = initializer,
                    kernel_regularizer = w_regu,
                    bias_regularizer = b_regu))
    model.add(LeakyReLU(alpha = leakiness))
    #model.add(BatchNormalization(axis = -1))
    model.add(Conv2D(256, (3, 3), strides = (1, 1), padding = 'same',
                    kernel_initializer = initializer,
                    kernel_regularizer = w_regu,
                    bias_regularizer = b_regu))
    model.add(LeakyReLU(alpha = leakiness))
    model.add(BatchNormalization(axis = -1))
    model.add(MaxPooling2D(pool_size = (3, 3), strides = (2, 2)))

    model.add(Conv2D(512, (3, 3), strides = (1, 1), padding = 'same',
                    kernel_initializer = initializer,
                    kernel_regularizer = w_regu,
                    bias_regularizer = b_regu))
    model.add(LeakyReLU(alpha = leakiness))
    #model.add(BatchNormalization(axis = -1))
    model.add(Conv2D(512, (3, 3), strides = (1, 1), padding = 'same',
                    kernel_initializer = initializer,
                    kernel_regularizer = w_regu,
                    bias_regularizer = b_regu))
    model.add(LeakyReLU(alpha = leakiness))
    model.add(BatchNormalization(axis = -1))
    model.add(MaxPooling2D(pool_size = (3, 3), strides = (2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(1024, kernel_initializer = initializer))
    model.add(LeakyReLU(alpha = leakiness))
    model.add(Dropout(0.5))

    model.add(Dense(1024, kernel_initializer = initializer))
    model.add(LeakyReLU(alpha = leakiness))

    model.add(Dense(nb_classes, activation = 'softmax'))

    #model.summary()

    train_datagen = ImageDataGenerator(rescale = 1./255., validation_split = 0.2)

    train_generator = train_datagen.flow_from_dataframe(labels_train, train_data_dir,
                                    x_col = 'train_image_name', y_col = 'level',
                                    has_ext = True, target_size = (img_height, img_width),
                                    color_mode = 'rgb', class_mode = 'categorical',
                                    batch_size = batch_size, seed = 42,
									subset = 'training')

    validation_generator = train_datagen.flow_from_dataframe(labels_train, train_data_dir,
                                    x_col = 'train_image_name', y_col = 'level',
                                    has_ext = True, target_size = (img_height, img_width),
                                    color_mode = 'rgb', class_mode = 'categorical',
                                    batch_size = batch_size, seed = 42,
									subset = 'validation')

    #initially 0 lr
    sgd = SGD(lr = 0.001, momentum = 0.0, decay = 0.0, nesterov=False)
    model.compile(optimizer = sgd , loss = 'categorical_crossentropy', metrics = ['accuracy'])

    tensorboard = TensorBoard(log_dir = 'log/', histogram_freq = 0, write_graph = True,
                              write_images = True)
    stop = EarlyStopping(monitor = 'val_loss', min_delta = 0.001, patience = 2,
                         verbose=2, mode='auto')

    #lr scheduler callback
    lrate = LearningRateScheduler(step_decay)
    lr_sched = step_decay_schedule(initial_lr=1e-4, decay_factor=0.75, step_size=2)
    lr_finder = LRFinder(min_lr=1e-5, max_lr=1e-2, steps_per_epoch=np.ceil(nb_epoch/batch_size),
                         epochs=3)

    #print(".n:- ", train_generator.n)
    #print(".batch_size:- ", train_generator.batch_size)
    #print(".samples:- ", train_generator.samples)

    STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
    STEP_SIZE_VALID = validation_generator.n // validation_generator.batch_size

    model.fit_generator(train_generator,
						steps_per_epoch = STEP_SIZE_TRAIN,
						validation_data = validation_generator,
						validation_steps = STEP_SIZE_VALID,
                        epochs = nb_epoch,
						class_weight = cl_weights,
						verbose = 1,
                        callbacks = [stop, tensorboard, lrate])

    #lr_finder.plot_loss()
    #lr_finder.plot_lr()

    return model, validation_generator, train_generator
