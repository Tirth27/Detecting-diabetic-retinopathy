#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 20:08:10 2019

@author: tirth
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

img_read = '/home/tirth/Diabetic Retinopathy/Model/Sample Dataset/13_right_mir.jpeg'
img_write = '/home/tirth/Diabetic Retinopathy/Model/Sample Dataset/opencv_test/CLAHE_2_10_left_mir.jpeg'

#1
img = cv2.imread(img_read,0)
hist,bins = np.histogram(img.flatten(),256,[0,256])
cdf = hist.cumsum()
cdf_normalized = cdf * hist.max()/ cdf.max()
#plt.plot(cdf_normalized, color = 'b')
#plt.hist(img.flatten(),256,[0,256], color = 'r')
#plt.xlim([0,256])
#plt.legend(('cdf','histogram'), loc = 'upper left')
#plt.show()

#2
img = cv2.imread(img_read,0)
equ = cv2.equalizeHist(img)
res = np.hstack((img,equ)) #stacking images side-by-side
#cv2.imwrite(img_write,res)

#3
img = cv2.imread(img_read)
img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
# equalize the histogram of the Y channel
img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
# convert the YUV image back to RGB format
img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
#cv2.imshow('Color input image', img)
#cv2.imshow('Histogram equalized', img_output)
ress = np.hstack((img,img_output))
cv2.imshow('Histogram equalized 3', ress)
#cv2.imwrite(img_write,ress)


#4 (Efficient) 
# https://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html
img = cv2.imread(img_read)
img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
# create a CLAHE object (Arguments are optional).
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
ress = np.hstack((img,img_output))
#cv2.imwrite(img_write,ress)
