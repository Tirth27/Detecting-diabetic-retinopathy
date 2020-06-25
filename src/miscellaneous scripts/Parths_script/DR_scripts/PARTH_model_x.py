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
from keras.callbacks import TensorBoard, EarlyStopping, LearningRateScheduler, ModelCheckpoint
from keras.optimizers import SGD,Adagrad
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score
from sklearn.utils import class_weight

img_rows, img_cols = 512,512 #i/p image rez
img_height, img_width = 512,512 #i/p image rez
channels = 3 #RGB image
classes = 5 #o/p classes
def cnn_model(channels, nb_epoch, batch_size, nb_classes, nb_gpus, cl_weights,
              leakiness, w_regu, b_regu, initializer, img_height, img_width,
              labels_train, train_data_dir):

  '''
  #Sample data+labels path
  sample_data_dir = r"D:\Final Year Project\sample_cnn\sample.npy"
  sample_label = r"D:\Final Year Project\sample_cnn\sample.csv"

  sample_data= np.load(sample_data_dir)
  sam_labels = pd.read_csv(sample_label)
  sam_labels = sam_labels.values
  '''


  input_shape = (img_height,img_width,channels)
  '''
  Conv2D takes a 4D tensor as input_shape but we need to pass only
  3D while keras takes care of batch size on its own
  so pass (img_height,img_width,channels) not (batch_size,img_height,img_width,channels)
  '''

  model = Sequential()

  model.add(Conv2D(32,
                 (3,3),
                 strides=(1,1),
                 input_shape = input_shape,
                 padding='same',
                 ))
  model.add(BatchNormalization(axis = -1))
  model.add(LeakyReLU(alpha = leakiness))
  model.add(MaxPooling2D(pool_size = (2, 2),
                       strides = (2, 2)))

  model.add(Conv2D(64,
                 (3, 3),
                 strides=(1, 1),
                 padding = 'same',
                 ))
  model.add(BatchNormalization(axis = -1))
  model.add(LeakyReLU(alpha = leakiness))
  model.add(MaxPooling2D(pool_size = (2, 2),
                       strides = (2, 2)))
  #model.add(Dropout(0.20))

  model.add(Conv2D(128,
                 (3, 3),
                 strides = (1, 1),
                 padding = 'same',
                 ))
  model.add(BatchNormalization(axis = -1))
  model.add(LeakyReLU(alpha = leakiness))
  
  #model.add(Dropout(0.20))
  model.add(Conv2D(64,
                 (1, 1),
                 strides=(1, 1),
                 padding = 'same',
                 ))
  model.add(BatchNormalization(axis = -1))
  model.add(LeakyReLU(alpha = leakiness))

  model.add(Conv2D(128,
                 (3, 3),
                 strides = (1, 1),
                 padding = 'same',
                 ))
  model.add(BatchNormalization(axis = -1))
  model.add(LeakyReLU(alpha = leakiness))

  model.add(MaxPooling2D(pool_size = (2, 2),
                       strides = (2, 2)))

  model.add(Conv2D(256,
                 (3, 3),
                 strides = (1, 1),
                 padding = 'same',
                 ))
  model.add(BatchNormalization(axis = -1))
  model.add(LeakyReLU(alpha = leakiness))

  model.add(Conv2D(128,
                 (1, 1),
                 strides = (1, 1),
                 padding = 'same',
                 ))
  model.add(BatchNormalization(axis = -1))
  model.add(LeakyReLU(alpha = leakiness))

  #model.add(Dropout(0.20))

  model.add(Conv2D(256,
                 (3, 3),
                 strides = (1, 1),
                 padding = 'same',
                 ))
  model.add(BatchNormalization(axis = -1))
  model.add(LeakyReLU(alpha = leakiness))
  model.add(MaxPooling2D(pool_size = (3, 3),
                       strides = (2, 2)))
  
  #model.add(Dropout(0.4))
  model.add(Conv2D(512,
                 (3, 3),
                 strides = (1, 1),
                 padding = 'same',
                 ))
  model.add(BatchNormalization(axis = -1))
  model.add(LeakyReLU(alpha = leakiness))

  model.add(MaxPooling2D(pool_size = (3, 3),
                       strides = (2, 2)))
##  model.add(Dropout(0.25))

  model.add(Flatten())

  model.add(Dense(1024))
  model.add(LeakyReLU(alpha = leakiness))
  model.add(Dropout(0.3))
  model.add(Dense(1024))
  model.add(LeakyReLU(alpha = leakiness))
  model.add(Dropout(0.3))

  model.add(Dense(classes,activation = 'softmax'))

  model.summary()
  ggwp
  

  sgd = SGD(lr = 0.0001,momentum=0.9, decay= 0,nesterov=True)

  model.compile(optimizer = sgd,
              loss = 'mean_squared_error',
              metrics = ['accuracy'])
  tensorboard = TensorBoard(log_dir = 'log/', histogram_freq = 0, write_graph = True,
                              write_images = True)
  stop = EarlyStopping(monitor = 'loss', patience = 0, verbose = 2, mode = 'auto')
  model_chkpt = ModelCheckpoint(filepath='saved_model/best_model/best_model_weights_Epoch_{epoch:02d}-ValLoss_{val_loss:.2f}.h5', monitor='val_loss', save_best_only=True)


  train_datagen = ImageDataGenerator(rescale = 1./255, validation_split=0.2)

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
                    verbose = 1,
                    callbacks = [stop, tensorboard, model_chkpt])

  return model, validation_generator, train_generator



def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    if not os.path.exists('Confusion_Matrix'):
        os.makedirs('Confusion_Matrix')
        print("Confusion Matrix Folder Created!")

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(6, 8))
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





