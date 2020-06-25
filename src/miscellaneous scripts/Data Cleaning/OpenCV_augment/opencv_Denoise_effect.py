#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 09:30:10 2019

@author: tirth
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt


img_read = '/home/tirth/Diabetic Retinopathy/Model/Sample Dataset/10_left_mir.jpeg'
img_write = '/home/tirth/Diabetic Retinopathy/Model/Sample Dataset/opencv_test/denoise_10_left_mir.jpeg'

img = cv2.imread(img_read)
dst = cv2.fastNlMeansDenoisingColored(img,None,1,1,7,21)
cv2.imwrite(img_write,dst)
#plt.subplot(121),plt.imshow(img)
#plt.subplot(122),plt.imshow(dst)
#plt.show()
