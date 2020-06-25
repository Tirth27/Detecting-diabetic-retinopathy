import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.utils import normalize, np_utils, multi_gpu_model
from keras.callbacks import TensorBoard, EarlyStopping
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score
from sklearn.utils import class_weight
#from skll.metrics import kappa

np.random.seed(1337)

def cnn_model(X_train, y_train, nb_filters, channels, nb_gpus, nb_classes, kernel_size, batch_size, nb_epoch):

    print("X_train:- ", X_train.shape, "\ny_train:- ", y_train.shape, "\nnb_filters:- ", nb_filters)
    print("channels:- ", channels, "\nnb_gpus:- ", nb_gpus, "\nkernel_size:- " , kernel_size)
    print("batch_size:- ", batch_size, "\nnb_epoch:- ", nb_epoch)

    if not os.path.exists('saved_model'):
        os.makedirs('saved_model')
        print("Path Set!")

    model = Sequential()
    model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1]), padding = 'valid', strides = 1,
                        input_shape = (img_rows, img_cols, channels),
                        activation = 'relu'))

    model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1]), activation = 'relu'))

    model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1]), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))

    model.add(Flatten())
    #print('Model Flatten out to: ', model.output_shape)

    model.add(Dense(128, activation = 'relu'))
    model.add(Dropout(0.25))

    model.add(Dense(nb_classes, activation = 'softmax'))

    #model = multi_gpu_model(model, gpus=nb_gpus)

    model.summary()

    #Load weights
    model.load_weights("/floyd/input/output3/saved_model/PART_3_cnn_normal_optimizer_SGD_recall_0.625.h5")

    sgd = SGD(lr = 0.001, momentum = 0.9, decay = 1e-6, nesterov = False)
    model.compile(loss = 'categorical_crossentropy', optimizer = sgd,
                        metrics = ['accuracy'])

    stop = EarlyStopping(monitor="val_acc", min_delta=0.001, patience=2, verbose=0,
                         mode="auto")
    tensorboard = TensorBoard(log_dir='tensorboard_logs/', histogram_freq = 0,
                                write_graph = True, write_images = True)

    model.fit(X_train, y_train, batch_size = batch_size,
                   epochs = nb_epoch,
                   verbose = 1,
                   validation_split = 0.2,
                   class_weight = 'auto',
                   callbacks = [stop, tensorboard])

    return model


def save_model(model, score, model_name):
    if score >= 0.27:
        print('Saving Model')
        model.save('saved_model/' + model_name + '_recall_' + str(round(score, 4)) + '.h5')
    else:
        print("Model Score is Low, Score:-", score)
        model.save('saved_model/' + model_name + '_LowScore_' +'_recall_' + str(round(score, 4)) + '.h5')


if __name__ == '__main__':

    img_rows, img_cols = 256, 256
    channels = 3
    nb_classes = 5

    nb_filters = 32
    kernel_size = (8, 8)
    #kernal_size = (4, 4)
    nb_gpus = 2
    nb_epoch = 30
    #nb_epoch = 50
    batch_size = 128

    #Data
    labels = pd.read_csv("/floyd/input/data1/trainLabels_master_256_v2.csv")
    X = np.load("/floyd/input/data/4_uncomp_save.npy")
    y = np.array(labels['level'][79800:])

    #Class Weights (For imbalanced classes)
    #cl_weights = class_weight.compute_class_weight('balanced', np.unique(y), y)

    #Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    #print(X_train.shape, X_test.shape, y_train, y_test)

    #Reshape
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, channels)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, channels)
    print("X_train Shape: ", X_train.shape)
    print("X_test Shape: ", X_test.shape)

    #Normalize
    #Method 1(data)
    X_train = X_train.astype('float32')
    X_train /= 255

    X_test = X_test.astype('float32')
    X_test /=255

    #Method 2(data)
    #X_train = normalize(X_train, axis=-1)
    #X_test = normalize(X_test, axis=-1)

    #(label)
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)
    print("y_train Shape: ", y_train.shape)
    print("y_test Shape: ", y_test.shape)

    #Training
    model = cnn_model(X_train = X_train, y_train = y_train, kernel_size = kernel_size,
                    nb_filters = nb_filters, channels = channels, nb_epoch = nb_epoch,
                    batch_size = batch_size, nb_classes = nb_classes, nb_gpus = nb_gpus)

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

    save_model(model = model, score = recall, model_name = "PART_4_cnn_normal_optimizer_SGD")
    print("Completed")
