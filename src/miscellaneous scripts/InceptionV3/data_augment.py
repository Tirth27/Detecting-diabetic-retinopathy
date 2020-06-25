import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from sklearn.model_selection import train_test_split
import pandas as pd

X = np.load("/Users/tirth/Documents/Diabetic Retinopathy/Model/sample.npy")
y = pd.read_csv('/Users/tirth/Documents/Diabetic Retinopathy/Model/sample.csv')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

datagen = ImageDataGenerator(
        rotation_range = 20,
        width_shift_range = 0.2,
        height_shift_range = 0.2,
        rescale = 1./255,
        brightness_range = [0.2, 0.8],
        shear_range = 0.2,
        zoom_range = 0.2,
        channel_shift_range = 0.2,
        vertical_flip = True,
        horizontal_flip = True,
        featurewise_center = True,
        fill_mode = 'constant'
)

X_train = X_train[0]
x = X_train.reshape((1,) + X_train.shape)

datagen.fit(x)

i = 0
for batch in datagen.flow(x, batch_size = 1, save_to_dir = 'Data_augment_preview',
                            save_prefix = 'eye', save_format = 'jpeg'):
    i += 1
    if i > 20:
        break
