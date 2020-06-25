from keras.models import load_model
from keras.preprocessing import image
from keras.utils import normalize
from keras.optimizers import SGD
from skimage import io
import numpy as np
from scipy.misc import imresize, imread
import cv2
import os
import json, codecs

save_model = load_model('/Users/tirth/Documents/Diabetic Retinopathy/Model/split data MAIN PROJECT/1/PART_1_cnn_binary_class/saved_model/PART_1_cnn_binary_class_recall_1.0.h5')

img_path = '/Users/tirth/Documents/Diabetic Retinopathy/crop256/10_left_mir.jpeg'
'''
['0.png', '1.1.jpg', '1.JPG', '2.2.png', '2.png', '5.png', '6.png', '7.1.png',
'7.jpg', '9.1.png', '9.3.jpg', '9.png', 'mnist_2conv.py', 'Mnist_log', 'model.py',
'modelsaved_mnist.h5']
'''

#print(os.listdir(img_path))
imgs = io.imread(img_path)
#print(imgs.shape)
#save_model.summary()

#test_image = model.fit(test_image)
sgd = SGD(lr = 0.001, momentum = 0.9, decay = 1e-6, nesterov = False)
save_model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])
#1
test_image = image.load_img(img_path, target_size=(imgs.shape[0], imgs.shape[1]))

test_image = imresize(test_image, (256, 256))
#test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY) #Convert to Black and White

test_image = image.img_to_array(test_image).reshape((256,256,3))
test_image = np.expand_dims(test_image, axis=0)
print('image shape-',test_image.shape)

#test_image = imresize(test_image, 28, 28)

#2
#tm = imread(img_path, mode = 'L')
#tm = np.invert(tm)
#tm = imresize(tm, 28, 28)
#test_image = test_image.reshape(1, 28, 28, 1)

#test_image = preprocess_input(test_image)
#print(test_image)

#3
img = imread(img_path)
img = normalize(img, axis=-1)
img = imresize(img, (256, 256))
#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #Convert Black and White
print('Image shape after resize-',img.shape)
cv2.imshow('Image',img)
img = np.reshape(img, [1, 256, 256, 3])

'''
1st and 3rd both methods are working just replace 'img' with 'test_image'
to use 1st method
'''
result = save_model.predict(img)
result = np.array_str(np.argmax(result, axis=1))
print('Predict digit-',result)

prob = save_model.predict_proba(img)
print('predict probability',prob)

cl = save_model.predict_classes(img)
print('predict class',cl)

#gn = save_model.predict_generator(img)
#print('predict class',gn)

'''
#List into JSON
#1
with open('list.json', 'w') as F:
    #F.write(json.dump(prob.tolist(), F))
    json.dump(prob.tolist(), F)
    #F.close

#2
#a = np.arange(10).reshape(2,5) #eg:- 2 by 5 array
b = prob.tolist() # nested lists with same data, indices
print(b)
file_path = "path.json" ## your path variable
json.dump(b, codecs.open(file_path, 'w', encoding='utf-8'), separators=(',', ':'),
 sort_keys=True, indent=4) ### this saves the array in .json format
'''
