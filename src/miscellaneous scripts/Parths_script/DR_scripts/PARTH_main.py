import PARTH_model as modl

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
np.random.seed(1337)
tf.set_random_seed(1337)

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
    nb_epoch = 1 # 70 for final model
    #batch_size = 32 #default
    batch_size = 6# 7 for my pc and 32 for cloud
    
 
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

    
    # Type of kernel_initializer(https://faroit.github.io/keras-docs/1.2.2/initializations/)
    #initializer = 'golort_normal' # Default -> 'glorot_uniform'
    #initializer = 'he_normal'
    initializer = 'glorot_uniform'

    #Test/Train Directory path
    train_data_dir = "D:\\Final Year Project\\crops\\Train"
    #train_labels = r"D:\Final Year Project\label\trainlabel_master_v2.csv"

    test_data_dir = "D:\\Final Year Project\\crops\\Test"
    test_labels = r"D:\Final Year Project\label\testLables.csv"

    validation_data_dir = "D:\\Final Year Project\\crops\\Validation"

    '''
    #Sample data+labels path
    sample_data_dir = r"D:\Final Year Project\sample_cnn\sample.npy"
    sample_label = r"D:\Final Year Project\sample_cnn\sample.csv"

    sample_data= np.load(sample_data_dir)
    sam_labels = pd.read_csv(sample_label)
    sam_labels = sam_labels.values
    '''

    #Saved Model Name
    model_name = 'FFD_512'
##
##    #Labels
##    labels_train = pd.read_csv(train_labels)
##    #labels_train = labels_train.loc[:,('level')]
##    y_train = np.array(labels_train['level'])
##
    labels_test = pd.read_csv(test_labels)
    y_test = np.array(labels_test['level'])
    y_test = to_categorical(y_test, nb_classes)
##
##    #Class Weights (For imbalanced classes)
##    cl_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)

    #Train
    model, validation_generator, train_generator = \
						modl.cnn_model(channels = channels, nb_epoch = nb_epoch,
                     batch_size = batch_size, nb_classes = nb_classes, nb_gpus = nb_gpus,
                     leakiness = leakiness, w_regu = w_regu, img_width = img_width, 
                     b_regu = b_regu, initializer = initializer, img_height = img_height,
                     validation_data_dir = validation_data_dir , train_data_dir = train_data_dir,
                     )

    test_datagen = ImageDataGenerator(rescale = 1./255)
    
    batch_size = 1
    test_generator = test_datagen.flow_from_directory(test_data_dir,									
                                                      target_size = (img_height, img_width),
                                                      color_mode = 'rgb',
                                                      class_mode = 'categorical',
                                                      batch_size = batch_size,
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
    precision = precision_score(test_generator.classes[test_generator.index_array], y_pred, average = 'micro')
    #Recall is intuitively the ability of classifer to find all postive samples.
    recall = recall_score(test_generator.classes[test_generator.index_array], y_pred, average = 'micro')
    print("Precision: ", precision)
    print("Recall: ", recall)

    #save model
    print("Saving Model.")
    save_model(model = model, epoch = nb_epoch, score = recall, model_name = model_name)

    #The best value is 1 and the worst value is 0
    f1 = f1_score(test_generator.classes[test_generator.index_array], y_pred, average = 'micro')
    #cohen_kappa is level of agreement between two annotators on a classification problem.
    cohen_kappa = cohen_kappa_score(test_generator.classes[test_generator.index_array], y_pred)
    quadratic_kappa = cohen_kappa_score(test_generator.classes[test_generator.index_array], y_pred, weights = 'quadratic')

    print("F1: ", f1)
    print("Cohen Kappa Score: ", cohen_kappa)
    print("Quadratic Kappa Score: ", quadratic_kappa)
    
    #Calculate and Plot Confusion Matrix
    cm = confusion_matrix(test_generator.classes[test_generator.index_array], y_pred)
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
