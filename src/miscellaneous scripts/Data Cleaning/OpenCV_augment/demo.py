import cv2 
import numpy as np 
from matplotlib import pyplot as plt

img_read = '/home/tirth/Diabetic Retinopathy/Model/Sample Dataset/10_right_mir.jpeg'
img_write = '/home/tirth/Diabetic Retinopathy/Model/opencv_test/test/2.jpeg'

img = cv2.imread(img_read)

#Denoise (https://docs.opencv.org/3.3.1/d5/d69/tutorial_py_non_local_means.html)
dst = cv2.fastNlMeansDenoisingColored(img,None,1,1,7,21)
de_comp = np.hstack((img, dst))
cv2.imshow('Denoise Compare',de_comp)

#CLAHE (Contrast Limited Adaptive Histogram Equalization)
# https://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html
img_yuv = cv2.cvtColor(dst, cv2.COLOR_BGR2YUV)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
clah_comp = np.hstack((img, img_output))
cv2.imshow('CLAHE Compare',clah_comp)
#cv2.imwrite(img_write, de_clahe)

#Find Optic Disk (https://www.pyimagesearch.com/2014/09/29/finding-brightest-spot-image-using-python-opencv/)
radius = 31 #must be an odd value
orig = img_output.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (radius, radius), 0)
(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
img = orig.copy()
cv2.circle(img, maxLoc, radius, (255, 0, 0), 2)
cv2.imshow("Find Bright Spot", img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

'''
g_image = cv2.cvtColor(img_output,cv2.COLOR_BGR2GRAY)
cv2.imshow('image',g_image)
b = img_output[:,:,0]
g = img_output[:,:,1]
r = img_output[:,:,2]
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
# opening = cv2.morphologyEx(b, cv2.MORPH_OPEN,kernel,5)
# opening_1 = cv2.morphologyEx(b, cv2.MORPH_OPEN,kernel,3)
# plt.subplot(2,3,1),plt.imshow(b),plt.title('Blue Layer')
# plt.subplot(2,3,2),plt.imshow(r),plt.title('red Layer')
# plt.subplot(2,3,3),plt.imshow(g),plt.title('green Layer')
#plt.subplot(2,3,1),plt.imshow(b,'gray'),plt.title('Blue Layer')
#plt.subplot(2,3,2),plt.imshow(r,'gray'),plt.title('red Layer')
#plt.subplot(2,3,3),plt.imshow(g,'gray'),plt.title('green Layer')
# plt.subplot(1,2,1),plt.imshow(opening,'gray'),
# plt.subplot(1,2,2),plt.imshow(opening_1,'gray'),
retval, threshold = cv2.threshold(g, 127, 255, cv2.THRESH_BINARY)
im2, contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#f = cv2.cvtColor(threshold,cv2.COLOR_LAB2RGB)
cv2.imshow('original',img)
cv2.imshow('im2',im2)
#cv2.imshow('th',f)
plt.show()
# plt.imshow(opening)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
