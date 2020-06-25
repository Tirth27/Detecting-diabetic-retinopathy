import numpy as np
import pandas as pd
import os
import math
import tensorflow as tf
from keras import backend as K
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.utils import normalize, np_utils, multi_gpu_model, to_categorical
from keras.callbacks import TensorBoard, EarlyStopping, LearningRateScheduler
from keras.callbacks import Callback
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
from skll.metrics import kappa

np.random.seed(1337)
tf.set_random_seed(1337)

# learning rate schedule
def step_decay(nb_epoch):
	initial_lr = 0.1
	drop = 0.5
	epochs_drop = 10.0
	lrate = initial_lr * math.pow(drop, math.floor((1+nb_epoch)/epochs_drop))
	return lrate

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
        return self.min_lr + (self.max_lr-self.min_lr) * x

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

def step_decay_schedule(initial_lr=1e-3, decay_factor=0.75, step_size=10):
    '''
    Wrapper function to create a LearningRateScheduler with step decay schedule.
    '''
    def schedule(epoch):
        return initial_lr * (decay_factor ** np.floor(epoch/step_size))

    return LearningRateScheduler(schedule)

def cnn_model(channels, nb_epoch, batch_size, nb_classes,
                nb_gpus, cl_weights, leakiness, w_lambdaa , b_lambdaa, initializer):

    print("\nClass Weights:- ", cl_weights)
    print("channels:- ", channels, "\nnb_gpus:- ", nb_gpus, "\nLeakiness:- ", leakiness)
    print("batch_size:- ", batch_size, "\nnb_epoch:- ", nb_epoch, "\nWeight lambda:- ", w_lambdaa)
    print("Bias lambda:- ", b_lambdaa, "\nInitializer:- ", initializer)

    if not os.path.exists('saved_model'):
        os.makedirs('saved_model')
        print("Path Set!")

    model = Sequential()
    model.add(Conv2D(32, (3,3), padding = 'same', strides = (1,1),
                        input_shape = (img_rows, img_cols, channels),
                        kernel_initializer = initializer,
                        kernel_regularizer = regularizers.l2(w_lambdaa),
                        bias_regularizer = regularizers.l2(b_lambdaa)))
    model.add(LeakyReLU(alpha = leakiness))
    model.add(BatchNormalization(axis = -1))
    model.add(Conv2D(32, (3, 3), strides = (1, 1), padding = 'same',
                    kernel_initializer = initializer,
                    kernel_regularizer = regularizers.l2(w_lambdaa),
                    bias_regularizer = regularizers.l2(b_lambdaa)))
    model.add(LeakyReLU(alpha = leakiness))
    model.add(BatchNormalization(axis = -1))
    model.add(MaxPooling2D(pool_size = (3, 3), strides = (2, 2)))

    model.add(Conv2D(64, (3, 3), strides = (1, 1), padding = 'same',
                    kernel_initializer = initializer,
                    kernel_regularizer = regularizers.l2(w_lambdaa),
                    bias_regularizer = regularizers.l2(b_lambdaa)))
    model.add(LeakyReLU(alpha = leakiness))
    model.add(BatchNormalization(axis = -1))
    model.add(Conv2D(64, (3, 3), strides = (1, 1), padding = 'same',
                    kernel_initializer = initializer,
                    kernel_regularizer = regularizers.l2(w_lambdaa),
                    bias_regularizer = regularizers.l2(b_lambdaa)))
    model.add(LeakyReLU(alpha = leakiness))
    model.add(BatchNormalization(axis = -1))
    model.add(MaxPooling2D(pool_size = (3, 3), strides = (2, 2)))

    model.add(Conv2D(128, (3, 3), strides = (1, 1), padding = 'same',
                    kernel_initializer = initializer,
                    kernel_regularizer = regularizers.l2(w_lambdaa),
                    bias_regularizer = regularizers.l2(b_lambdaa)))
    model.add(LeakyReLU(alpha = leakiness))
    model.add(BatchNormalization(axis = -1))
    model.add(Conv2D(128, (3, 3), strides = (1, 1), padding = 'same',
                    kernel_initializer = initializer,
                    kernel_regularizer = regularizers.l2(w_lambdaa),
                    bias_regularizer = regularizers.l2(b_lambdaa)))
    model.add(LeakyReLU(alpha = leakiness))
    model.add(BatchNormalization(axis = -1))
    model.add(MaxPooling2D(pool_size = (3, 3), strides = (2, 2)))

    model.add(Conv2D(256, (3, 3), strides = (1, 1), padding = 'same',
                    kernel_initializer = initializer,
                    kernel_regularizer = regularizers.l2(w_lambdaa),
                    bias_regularizer = regularizers.l2(b_lambdaa)))
    model.add(LeakyReLU(alpha = leakiness))
    #model.add(BatchNormalization(axis = -1))
    model.add(Conv2D(256, (3, 3), strides = (1, 1), padding = 'same',
                    kernel_initializer = initializer,
                    kernel_regularizer = regularizers.l2(w_lambdaa),
                    bias_regularizer = regularizers.l2(b_lambdaa)))
    model.add(LeakyReLU(alpha = leakiness))
    model.add(BatchNormalization(axis = -1))
    model.add(MaxPooling2D(pool_size = (3, 3), strides = (2, 2)))

    model.add(Conv2D(512, (3, 3), strides = (1, 1), padding = 'same',
                    kernel_initializer = initializer,
                    kernel_regularizer = regularizers.l2(w_lambdaa),
                    bias_regularizer = regularizers.l2(b_lambdaa)))
    model.add(LeakyReLU(alpha = leakiness))
    #model.add(BatchNormalization(axis = -1))
    model.add(Conv2D(512, (3, 3), strides = (1, 1), padding = 'same',
                    kernel_initializer = initializer,
                    kernel_regularizer = regularizers.l2(w_lambdaa),
                    bias_regularizer = regularizers.l2(b_lambdaa)))
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

    model.summary()

    train_datagen = ImageDataGenerator(rescale = 1./255.,
									validation_split = 0.2)

    train_generator = train_datagen.flow_from_dataframe(labels, train_data_dir,
                                    x_col = 'train_image_name', y_col = 'level',
                                    has_ext = True, target_size = (img_rows, img_cols),
                                    color_mode = 'rgb', class_mode = 'categorical',
                                    batch_size = batch_size, seed = 42,
									subset = 'training')

    validation_generator = train_datagen.flow_from_dataframe(labels, train_data_dir,
                                    x_col = 'train_image_name', y_col = 'level',
                                    has_ext = True, target_size = (img_rows, img_cols),
                                    color_mode = 'rgb', class_mode = 'categorical',
                                    batch_size = batch_size, seed = 42,
									subset = 'validation')

    test_datagen = ImageDataGenerator(rescale = 1./255.)

    test_generator = test_datagen.flow_from_dataframe(labels, test_data_dir,
									x_col = 'train_image_name', y_col = 'level',
                                    has_ext = True, target_size = (img_rows, img_cols),
                                    color_mode = 'rgb', class_mode = 'categorical',
                                    batch_size = batch_size, seed = 42, shuffle = False)

    sgd = SGD(lr = 0.01, momentum = 0.0, decay = 0.0, nesterov=False) #initially 0 lr
    model.compile(optimizer = sgd , loss = 'categorical_crossentropy', metrics = ['accuracy'])


    tensorboard = TensorBoard(log_dir = 'log/', histogram_freq = 0, write_graph = True, write_images = True)
    stop = EarlyStopping(monitor = 'val_loss', min_delta = 0.001, patience = 2, verbose=2, mode='auto')
    #lrate = LearningRateScheduler(step_decay) #lr scheduler callback
    lr_sched = step_decay_schedule(initial_lr=1e-4, decay_factor=0.75, step_size=2)
    lr_finder = LRFinder(min_lr=1e-5, max_lr=1e-2, steps_per_epoch=np.ceil(nb_epoch/batch_size), epochs=3)
    #print(".n", train_generator.n)
    #print(".batch_size", train_generator.batch_size)
    #print(".samples", train_generator.samples)

    STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
    STEP_SIZE_VALID = validation_generator.n // validation_generator.batch_size

    model.fit_generator(train_generator,
						steps_per_epoch = STEP_SIZE_TRAIN,
						validation_data = validation_generator,
						validation_steps = STEP_SIZE_VALID,
                        epochs = nb_epoch,
						class_weight = cl_weights,
						verbose = 1,
                        callbacks = [stop, tensorboard, lr_sched, lr_finder])
    lr_finder.plot_loss()

    return model, validation_generator, test_generator, train_generator


def save_model(model, score, model_name):
    if score >= 0.75:
        print('Saving Model')
        model.save('saved_model/' + model_name + '_recall_' + str(round(score, 4)) + '.h5')
    else:
        print("Model Score is Low, Score:-", score)
        model.save('saved_model/' + model_name + '_LowScore_' +'_recall_' + str(round(score, 4)) + '.h5')


if __name__ == '__main__':

    img_rows, img_cols = 256, 256
    channels = 3
    nb_classes = 5
    nb_gpus = 2

    train_data_dir = '/Users/tirth/Documents/Diabetic Retinopathy/Model/Sample Dataset'
    train_labels = '/Users/tirth/Documents/Diabetic Retinopathy/Model/sample.csv'
    test_data_dir = '/Users/tirth/Documents/Diabetic Retinopathy/Model/Sample Dataset'
    test_labels = train_labels
    
    model_name = 'cnn_for_dr'

    nb_epoch = 1
    #batch_size = 32 #default
    batch_size = 10

    # Value of alpha for LeakyReLU
    #leakiness = 0.5 #JeffreyDf
    leakiness = 0.01

    # Value of lamda for regularizers
    w_lambdaa = 0.01
    b_lambdaa = 0.01

    # Type of kernel_initializer(https://faroit.github.io/keras-docs/1.2.2/initializations/)
    #initializer = 'golort_normal'
    initializer = 'he_normal'

    #Data
    labels = pd.read_csv(train_labels)
    #labels = labels.loc[:,('level')]
    y = np.array(labels['level'])
    
    y_test = pd.read_csv(test_labels)
    y_test = np.array(y_test['level'])
    y_test = to_categorical(y_test, nb_classes)

    #Class Weights (For imbalanced classes)
    cl_weights = class_weight.compute_class_weight('balanced', np.unique(y), y)

    model, validation_generator, test_generator, train_generator = \
						cnn_model(channels = channels,
                        nb_epoch = nb_epoch, batch_size = batch_size,
                        nb_classes = nb_classes, nb_gpus = nb_gpus, cl_weights = cl_weights,
                        leakiness = leakiness, w_lambdaa = w_lambdaa, b_lambdaa = b_lambdaa,
                        initializer = initializer)

	#Return loss value & metrics values for the model in test mode.
    STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
    score = model.evaluate_generator(validation_generator, steps = STEP_SIZE_TRAIN, verbose = 0)
    print("Test score: ", score[0])
    print("Test Accuracy: ", score[1])

    #Generates output predictions for input samples
    test_generator.reset()
    batch_size = 1
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
