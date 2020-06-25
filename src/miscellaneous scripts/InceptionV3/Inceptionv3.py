from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import SGD
from keras.utils import to_categorical
from keras.callbacks import TensorBoard, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score
from sklearn.utils import class_weight
from skll.metrics import kappa
import numpy as np
import pandas as pd
import os

np.random.seed(1337)

#Hyperparameter
img_rows, img_cols = 256, 256
channels = 3
last_layer_epoch = 70
nb_classes = 5
batch_size = 128

#Import Data
labels = pd.read_csv("/Users/tirth/Documents/Diabetic Retinopathy/Model/sample.csv")
labels = labels.loc[:26599,('level')]

X = np.load("/Users/tirth/Documents/Diabetic Retinopathy/Model/sample.npy")
y = np.array(labels)

#Class Weights (For imbalanced classes)
cl_weights = class_weight.compute_class_weight('balanced', np.unique(y), y)

#Splitting Data
print('Splitting Dataset into Training/Testing set.')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

#Reshape Data
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, channels)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, channels)
print("X_train Shape: ", X_train.shape)
print("X_test Shape: ", X_test.shape)

#Normalize
print("Normalizing Data")
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

y_train = to_categorical(y_train, nb_classes)
y_test = to_categorical(y_test, nb_classes)
print("y_train Shape: ", y_train.shape)
print("y_test Shape: ", y_test.shape)

#Load Inception-V3
print("Training")
base_model = InceptionV3(weights = 'imagenet', include_top = False, input_shape = (img_rows, img_cols, channels))
#model.summary()
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(2048, activation = 'relu')(x)
predictions = Dense(nb_classes, activation = 'softmax')(x)

model = Model(inputs = base_model.input, outputs = predictions)
model.summary()

#Freeze all convolutional Incpetion-V3 layers
for layer in base_model.layers:
    layer.trainable = False

tensorboard = TensorBoard(log_dir = 'log/', histogram_freq = 0, write_graph = True, write_images = True)
stop = EarlyStopping(monitor = 'val_loss', min_delta = 0.001, patience = 2, verbose=0, mode='auto')

model.compile(loss = 'categorical_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])
model.fit(X_train, y_train, batch_size = batch_size, epochs = last_layer_epoch,
            validation_split = 0.2, callbacks = [stop, tensorboard], class_weight = cl_weights)

# We choose to train the top 2 incpetion blocks i.e we will Freeze
# the first 249 layers and unfreeze the rest.
print('Freezing till layer # 249.')
for layers in model.layers[:249]:
    layer.trainable = False

print('Unreezing after layer # 249.')
for layers in model.layers[249:]:
    layers.trainable = True


print('FineTuning the model')
sgd = SGD(lr = 0.0001, momentum = 0.9, decay = 1e-6, nesterov = False)
model.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])
model.fit(X_train, y_train, batch_size = batch_size, epochs = last_layer_epoch,
            validation_split = 0.2, callbacks = [stop, tensorboard], class_weight = cl_weights)

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

if not os.path.exists('saved_model'):
        os.makedirs('saved_model')
        print("Path Set!")

def save_model(model, score, model_name):
    if score >= 0.75:
        print('Saving Model')
        model.save('saved_model/' + model_name + '_recall_' + str(round(score, 4)) + '.h5')
    else:
        print("Model Score is Low, Score:-", score)
        model.save('saved_model/' + model_name + '_LowScore_' +'_recall_' + str(round(score, 4)) + '.h5')

save_model(model = model, score = recall, model_name = "cnn_normal_optimizer_SGD")
print("Completed")
