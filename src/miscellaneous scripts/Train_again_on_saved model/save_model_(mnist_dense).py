from keras.datasets import mnist
import tensorflow.keras as keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Conv2D, MaxPooling2D
from keras.utils import normalize
from keras.callbacks import TensorBoard, EarlyStopping
import matplotlib.pyplot as plt
from keras.optimizers import SGD
from keras.callbacks import Callback
from keras import backend as K
import tensorflow as tf

model_Name = "Mnist_Dense"
tensorboard_name = 'dense(512-256)/'

class change_lr(Callback):
    def __init__(self):
        super().__init__()
    '''
    def clr(self):
        Calculate the learning rate.
        l_r = 0.0003
        return l_r

    def on_epoch_begin(self, epoch, logs=None):
        #print(K.get_value(self.model.optimizer.lr))
        if epoch >= 4:
            return K.set_value(self.model.optimizer.lr, self.clr())
    '''
    
    def on_epoch_end(self, epoch, logs=None):
        print('\nlr:- ', K.eval(self.model.optimizer.lr))
        print('decay:- ', K.eval(self.model.optimizer.decay))

        lr = self.model.optimizer.lr
        decay = self.model.optimizer.decay
        iterations = self.model.optimizer.iterations
        lr_with_decay = lr / (1. + decay * K.cast(iterations, K.dtype(decay)))
        print('Method_1_decay_applied:- ', K.eval(lr_with_decay))
        
        '''
        optimizer = self.model.optimizer
        _lr = tf.to_float(optimizer.lr, name='ToFloat') 
        _decay = tf.to_float(optimizer.decay, name='ToFloat')
        _iter = tf.to_float(optimizer.iterations, name='ToFloat')
        lr_decay = K.eval(_lr * (1. / (1. + _decay * _iter)))
        print('Method_2_LR: {:.20f}\n'.format(lr_decay))
        '''

        with open("test.txt", "a") as myfile:
            myfile.write("\n\nEpoch:- " + str(epoch))
            myfile.write("\nInitial_lr:- " + str(K.eval(self.model.optimizer.lr)))
            myfile.write("\nDecay:- " + str(K.eval(self.model.optimizer.decay)))
            myfile.write("\nlr_with_decay" + str(K.eval(lr_with_decay)))
            #myfile.write("\n" + str(lr_decay))

#Load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#Normalise the data
x_train = normalize(x_train, axis=-1)
x_test = normalize(x_test, axis=-1)

#Reshape Data
#1-for dense layer
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000,784)

#model = load_model("/Users/tirth/Documents/Diabetic Retinopathy/Model/train on saved model/Saved model/saved2.h5")
model = Sequential()


model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

model.load_weights("C:\\Tirth\\Diabetic Retinopathy\\Model\\Train_again_on_saved model\\saved_model\\saved_epoch_10.h5")

#compile model
sgd = SGD(lr = 9.953318e-05, momentum = 0.9, decay = 1e-6, nesterov = True)
model.compile(optimizer=sgd, loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#Tensorboard for visualise
#tensorboard = TensorBoard(log_dir='Mnist_log/' + tensorboard_name,
#                          histogram_freq=30)
#stop = EarlyStopping(monitor="val_acc", min_delta=0.001, patience=2, verbose=0,
#                     mode="auto")

lrChange = change_lr()

#Feed the data
model.fit(x_train, y_train, epochs=6, batch_size=128,
          validation_data=(x_test, y_test), verbose=2, callbacks=[lrChange])

model.save("C:\\Tirth\\Diabetic Retinopathy\\Model\\Train_again_on_saved model\\saved_model\\Final_saved.h5")
