import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd

#labels
df = pd.read_csv("/Users/tirth/Documents/Diabetic Retinopathy/Model/sample.csv")

#load uncompressed npy
x = np.load("/Users/tirth/Documents/Diabetic Retinopathy/Sample dataset/array/uncomp_save.npy")
print('x ->', x.shape)

#load compressed .npz
y = np.load("/Users/tirth/Documents/Diabetic Retinopathy/Sample dataset/array/comp_savez_comp.npz")
print("y ->", y.files)
#load uncompressed .npz
z = np.load("/Users/tirth/Documents/Diabetic Retinopathy/Sample dataset/array/uncomp_savez.npz")
print("z ->", z.files)

#unzip .npz
yy = y['arr_0']
zz = z['arr_0']

#get shape
print('y ->', yy.shape)
print('z ->', zz.shape)
#plt.imshow(zz[0])
#plt.show()

#split dataset
X_train, X_test, y_train, y_test = train_test_split(yy, df, test_size=0.2, random_state=42)
print(X_train.shape)
print(X_test.shape)
#plt.imshow(X_train[0])
#plt.show()
