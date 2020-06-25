import pandas as pd
import numpy as np
import os
import math
import tensorflow as tf
from keras import backend as K
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, MaxoutDense, Dropout, Flatten, BatchNormalization, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.utils import normalize, np_utils, multi_gpu_model, to_categorical
from keras.callbacks import TensorBoard, EarlyStopping, LearningRateScheduler
from keras.optimizers import SGD,Adagrad
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score
from sklearn.utils import class_weight

img_rows, img_cols = 512,512 #i/p image rez
img_height, img_width = 512,512 #i/p image rez
channels = 3 #RGB image
classes = 5 #o/p classes

#Train data+labels path
train_data_dir = r"D:\Final Year Project\crops\denoise_512"
train_labels = r"D:\Final Year Project\label\trainlabel_master_v2.csv"
#Test data+labels path
test_data_dir = r"D:\Final Year Project\crops\denoise_test_512"
test_labels = r"D:\Final Year Project\label\testLables.csv"
'''
#Sample data+labels path
sample_data_dir = r"D:\Final Year Project\sample_cnn\sample.npy"
sample_label = r"D:\Final Year Project\sample_cnn\sample.csv"

sample_data= np.load(sample_data_dir)
sam_labels = pd.read_csv(sample_label)
sam_labels = sam_labels.values
'''
nb_epoch = 3 # 70 for final model
batch_size = 6 # 7 for my pc and 32 for cloud
lambdaa = 0.01 # regularization parameter
leakiness = 0.01 # Value of alpha for LeakyReLU

input_shape = (img_height,img_width,channels)
'''
Conv2D takes a 4D tensor as input_shape but we need to pass only
3D while keras takes care of batch size on its own
so pass (img_height,img_width,channels) not (batch_size,img_height,img_width,channels)
'''

initializer = 'he_normal' # For initializing weights 
#Saved Model Name
model_name = 'DRcnn'

#getting train labels
labels_train = pd.read_csv(train_labels)
#convert 'level' from train_labels to numpy array and store it to y_train
y_train = np.array(labels_train['level'])
#getting test labels
labels_test = pd.read_csv(test_labels)
#convert 'level' from test_labels to numpy array and store it to y_test
y_test = np.array(labels_test['level'])
#convert y_test to catagorical
y_test = to_categorical(y_test, classes)

#Class Weights (For imbalanced classes)
cl_weights = class_weight.compute_class_weight('balanced',
                                               np.unique(y_train),
                                               y_train)


model = Sequential()

model.add(Conv2D(32,
                 (4,4),
                 strides=(2,2),
                 input_shape = input_shape,
                 padding='same',
                 ))
model.add(LeakyReLU(alpha = leakiness))
model.add(Conv2D(32,
                 (4,4),
                 strides=(1,1),
                 padding='same',
                 ))
model.add(LeakyReLU(alpha = leakiness))
#model.add(BatchNormalization(axis = -1))
model.add(MaxPooling2D(pool_size = (3, 3),
                       strides = (2, 2)))
#model.add(Dropout(0.15))

model.add(Conv2D(64,
                 (4, 4),
                 strides=(2, 2),
                 padding = 'same',
                 ))
model.add(LeakyReLU(alpha = leakiness))
model.add(Conv2D(64,
                 (4, 4),
                 strides=(1, 1),
                 padding = 'same',
                 ))
model.add(LeakyReLU(alpha = leakiness))
#model.add(BatchNormalization(axis = -1))
model.add(MaxPooling2D(pool_size = (3, 3),
                       strides = (2, 2)))
#model.add(Dropout(0.20))

model.add(Conv2D(128,
                 (4, 4),
                 strides = (1, 1),
                 padding = 'same',
                 ))
model.add(LeakyReLU(alpha = leakiness))
model.add(Conv2D(128,
                 (4, 4),
                 strides = (1, 1),
                 padding = 'same',
                 ))
model.add(LeakyReLU(alpha = leakiness))
#model.add(BatchNormalization(axis = -1))
model.add(MaxPooling2D(pool_size = (3, 3),
                       strides = (2, 2)))
#model.add(Dropout(0.20))

model.add(Conv2D(256,
                 (4, 4),
                 strides = (1, 1),
                 padding = 'same',
                 ))
model.add(LeakyReLU(alpha = leakiness))

#model.add(BatchNormalization(axis = -1))
model.add(MaxPooling2D(pool_size = (3, 3),
                       strides = (2, 2)))
model.add(Conv2D(384,
                 (4, 4),
                 strides = (1, 1),
                 padding = 'same',
                 ))
model.add(LeakyReLU(alpha = leakiness))

#model.add(BatchNormalization(axis = -1))
model.add(MaxPooling2D(pool_size = (3, 3),
                       strides = (2, 2)))
#model.add(Dropout(0.4))
model.add(Conv2D(512,
                 (4, 4),
                 strides = (1, 1),
                 padding = 'same',
                 ))
model.add(LeakyReLU(alpha = leakiness))
#model.add(BatchNormalization(axis = -1))
model.add(MaxPooling2D(pool_size = (3, 3),
                       strides = (2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(1024))
model.add(LeakyReLU(alpha = leakiness))
model.add(Dropout(0.5))
model.add(Dense(1024))
model.add(LeakyReLU(alpha = leakiness))

model.add(Dense(classes,activation = 'softmax'))

model.summary()

sgd = SGD(lr = 0.0001,momentum=0.9, decay= 1e-6 ,nesterov=True)

model.compile(optimizer = sgd,
              loss = 'mean_squared_error',
              metrics = ['accuracy'])


train_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_dataframe(labels_train,
                                                    train_data_dir,
                                                    x_col = 'train_image_name',
                                                    y_col = 'level',
                                                    has_ext = True,
                                                    target_size = (img_height, img_width),
                                                    class_mode = 'categorical',
                                                    batch_size = batch_size,
                                                    subset = 'training')
validation_generator = train_datagen.flow_from_dataframe(labels_train,
                                                         train_data_dir,
                                                         x_col = 'train_image_name',
                                                         y_col = 'level',
                                                         has_ext = True,
                                                         target_size = (img_height, img_width),                            
                                                         class_mode = 'categorical',
                                                         batch_size = batch_size,
                                                         subset = 'validation')
STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
STEP_SIZE_VALID = validation_generator.n // validation_generator.batch_size
model.fit_generator(train_generator,
                    steps_per_epoch = STEP_SIZE_TRAIN,
                    validation_data = validation_generator,
                    validation_steps = STEP_SIZE_VALID,
                    epochs = nb_epoch,
                    class_weight = cl_weights,
                    verbose = 1)




#'''
