from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Conv2D, MaxPooling2D
from keras.utils import normalize
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import keras

X  = np.load("/Users/tirth/Documents/Diabetic Retinopathy/Model/sample.npy")
print(X.shape)

y = pd.read_csv("/Users/tirth/Documents/Diabetic Retinopathy/Model/sample.csv")
y = y['level']
print(y.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
print(X_train.shape, X_test.shape)

# Coverts a class vector(integers) to binary classes
# i.e 2 = [0, 0, 1, 0, 0]
y_train = keras.utils.to_categorical(y_train, 5)
y_test = keras.utils.to_categorical(y_test, 5)

#print("y_train(class vector): ", y_train.shape)
#print('y_test(class vector): ', y_test.shape)
#print("y_train(class vector): ", y_train)
#print('y_test(class vector): ', y_test)

# Normalise
# Method 1
X_train = keras.utils.normalize(X_train, axis=-1)
X_test = keras.utils.normalize(X_test, axis=-1)
print(X_train.shape)

# Method 2
#X_train = X_train.astype('float32')
#X_test = X_test.astype('float32')

#X_train /= 255
#X_test /= 255
#print(X_train[0])
#plt.imshow(X_train[0])
#plt.show()

model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same', input_shape=(256, 256, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), padding='same', activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(512, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(5, activation='softmax'))

model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy', 'categorical_accuracy'])
model.fit(X_train, y_train, epochs=2, batch_size=10, shuffle=True)


model.save("/Users/tirth/Documents/Diabetic Retinopathy/Model/train on saved model/Saved model/saved.h5")
print("Done Training")
