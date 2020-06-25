#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 19:40:49 2019

@author: tirth
"""
import cv2
import os
import time
import numpy as np

#start the timer
start_time_first = time.time()

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def denoise_and_CLAHE(path, new_path):

    create_directory(new_path)
    dire = [l for l in os.listdir(path) if l != '.DS_Store'] #For MacUsers

    for item in dire:
        img = cv2.imread(path + item)

        #Denoise (https://docs.opencv.org/3.3.1/d5/d69/tutorial_py_non_local_means.html)
        img = cv2.fastNlMeansDenoisingColored(img, None, 1, 1, 7, 21)

        #CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # https://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8, 8))
        img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
        img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

        cv2.imwrite(str(new_path + item), img)

    # time elapsed
    print("--- %s seconds ---" % (time.time() - start_time_first))

#by doing this the code cannot be use as module
if __name__ == "__main__":
    print("Denoising And Adjusting Contrast -> 256x256")
    denoise_and_CLAHE(path = 'E:/Tirth/test/crop_test_256/crop_test_256/',
                      new_path = 'E:/Tirth/train/denoise_test_256/')

    print("Denoising And Adjusting Contrast -> 512x512")
    denoise_and_CLAHE(path = 'E:/Tirth/test/crop_test_512/crop_test_512/',
                      new_path = 'E:/Tirth/train/denoise_test_512/')
