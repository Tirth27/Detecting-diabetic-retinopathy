import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, MaxPooling2D, AveragePooling1D
from keras.layers import InputLayer, concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.utils import normalize, np_utils, multi_gpu_model
from keras.callbacks import TensorBoard, EarlyStopping
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score
from sklearn.utils import class_weight

#from skll.metrics import kappa

np.random.seed(1337)

def cnn_model(X_train, y_train, channels, nb_gpus, nb_classes, batch_size, nb_epoch, cl_weights, leakiness):

    print("X_train:- ", X_train.shape, "\ny_train:- ", y_train.shape, "\nClass Weights:- ", cl_weights)
    print("channels:- ", channels, "\nnb_gpus:- ", nb_gpus, "\nLeakiness:- ", leakiness)
    print("batch_size:- ", batch_size, "\nnb_epoch:- ", nb_epoch)

    if not os.path.exists('saved_model'):
        os.makedirs('saved_model')
        print("Path Set!")

    input_layer = InputLayer(input_shape = (64, 2))

    model = Sequential()
    #Retinet has one Convolve layer with 7x7 filter but it is expensive so Convert
    #it into two 3x3 filter
    model.add(Conv2D(32, (7,7), padding = 'same', strides = (2,2),
                        input_shape = (img_rows, img_cols, channels)))
    model.add(LeakyReLU(alpha = leakiness))
    model.add(MaxPooling2D(pool_size = (3, 3), strides = (2, 2)))

    model.add(Conv2D(32, (3, 3), strides = (1, 1), padding = 'same'))
    model.add(LeakyReLU(alpha = leakiness))
    model.add(Conv2D(32, (3, 3), strides = (1, 1), padding = 'same'))
    model.add(LeakyReLU(alpha = leakiness))
    #model.add(Conv2D(32, (3, 3), strides = (1, 1), padding = 'same'))
    #model.add(LeakyReLU(alpha = leakiness))
    model.add(MaxPooling2D(pool_size = (3, 3), strides = (2, 2)))

    model.add(Conv2D(64, (3, 3), strides = (1, 1), padding = 'same'))
    model.add(LeakyReLU(alpha = leakiness))
    model.add(Conv2D(64, (3, 3), strides = (1, 1), padding = 'same'))
    model.add(LeakyReLU(alpha = leakiness))
    #model.add(Conv2D(64, (3, 3), strides = (1, 1), padding = 'same'))
    #model.add(LeakyReLU(alpha = leakiness))
    model.add(MaxPooling2D(pool_size = (3, 3), strides = (2, 2)))

    model.add(Conv2D(128, (3, 3), strides = (1, 1), padding = 'same'))
    model.add(LeakyReLU(alpha = leakiness))
    model.add(Conv2D(128, (3, 3), strides = (1, 1), padding = 'same'))
    model.add(LeakyReLU(alpha = leakiness))
    model.add(Conv2D(128, (3, 3), strides = (1, 1), padding = 'same'))
    model.add(LeakyReLU(alpha = leakiness))
    model.add(Conv2D(128, (3, 3), strides = (1, 1), padding = 'same'))
    model.add(LeakyReLU(alpha = leakiness))
    model.add(MaxPooling2D(pool_size = (3, 3), strides = (2, 2)))

    model.add(Conv2D(256, (3, 3), strides = (1, 1), padding = 'same'))
    model.add(LeakyReLU(alpha = leakiness))
    model.add(Conv2D(256, (3, 3), strides = (1, 1), padding = 'same'))
    model.add(LeakyReLU(alpha = leakiness))
    model.add(Conv2D(256, (3, 3), strides = (1, 1), padding = 'same'))
    model.add(LeakyReLU(alpha = leakiness))
    model.add(Conv2D(256, (3, 3), strides = (1, 1), padding = 'same'))
    model.add(LeakyReLU(alpha = leakiness))
    model.add(MaxPooling2D(pool_size = (3, 3), strides = (2, 2)))

    #model.add(AveragePooling1D(pool_size = 2))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha = leakiness))
    model.add(Dropout(0.5))

    model.add(Dense(1024))
    model.add(LeakyReLU(alpha = leakiness))
    model.add(Dropout(0.5))

    model.add(Dense(nb_classes, activation = 'softmax'))

    #model = multi_gpu_model(model, gpus=nb_gpus)
    model.summary()

    sgd = SGD(lr = 0.01, momentum = 0.9, decay = 1e-6, nesterov = False)
    model.compile(loss = 'categorical_crossentropy', optimizer = sgd,
                        metrics = ['accuracy'])

    stop = EarlyStopping(monitor="val_loss", min_delta = 0.001, patience=2, verbose=0, mode="auto")
    tensorboard = TensorBoard(log_dir='tensorboard_logs/', histogram_freq = 0,
                                write_graph = True, write_images = True)

    model.fit(X_train, y_train, batch_size = batch_size,
                   epochs = nb_epoch,
                   verbose = 1,
                   validation_split = 0.2,
                   class_weight = cl_weights,
                   callbacks = [stop, tensorboard])

    return model


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
    nb_epoch = 70
    batch_size = 128
    leakiness = 0.5

    #Data
    labels = pd.read_csv("/Users/tirth/Documents/Diabetic Retinopathy/Model/sample.csv")
    labels = labels.loc[:26599,('level')]

    X = np.load("/Users/tirth/Documents/Diabetic Retinopathy/Model/sample.npy")
    y = np.array(labels)

    #Class Weights (For imbalanced classes)
    cl_weights = class_weight.compute_class_weight('balanced', np.unique(y), y)

    #Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    #Reshape
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, channels)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, channels)
    print("X_train Shape: ", X_train.shape)
    print("X_test Shape: ", X_test.shape)

    #Normalize
    X_train = X_train.astype('float32')
    X_train /= 255

    X_test = X_test.astype('float32')
    X_test /=255

    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)
    print("y_train Shape: ", y_train.shape)
    print("y_test Shape: ", y_test.shape)

    #Training
    model = cnn_model(X_train = X_train, y_train = y_train, channels = channels,
                        nb_epoch = nb_epoch, batch_size = batch_size,
                        nb_classes = nb_classes, nb_gpus = nb_gpus, cl_weights = cl_weights,
                        leakiness = leakiness)
    

    #Generates output predictions for input samples
    y_pred = model.predict(X_test)

    #Return loss value & metrics values for the model in test mode.
    score = model.evaluate(X_test, y_test, verbose = 0)
    print("Test score: ", score[0])
    print("Test Accuracy: ", score[1])

    y_test = np.argmax(y_test, axis = 1)
    y_pred = np.argmax(y_pred, axis = 1)

    ##The best value is 1 and the worst value is 0
    #Recall is intuitively the ability of classifer not to label as postive a sample that is negative.
    precision = precision_score(y_test, y_pred, average = 'micro')
    #Recall is intuitively the ability of classifer to find all postive samples.
    recall = recall_score(y_test, y_pred, average = 'micro')

    print("Precision: ", precision)
    print("Recall: ", recall)

    #Compliment
    #The best value is 1 and the worst value is 0
    precision = precision_score(y_test, y_pred, average = 'micro')
    recall_s = recall_score(y_test, y_pred, average = 'micro')
    f1 = f1_score(y_test, y_pred, average = 'micro')
    #cohen_kappa is level of agreement between two annotators on a classification problem.
    cohen_kappa = cohen_kappa_score(y_test, y_pred)
    #quad_kappa = kappa(y_test, y_pred, weights='quadratic')

    print("----------------Compliment----------------")
    print("Precision: ", precision)
    print("Recall: ", recall_s)
    print("F1: ", f1)
    print("Cohen Kappa Score", cohen_kappa)
    #print("Quadratic Kappa: ", quad_kappa)

    save_model(model = model, score = recall, model_name = "cnn_retinet")
    print("Completed")
