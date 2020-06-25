import numpy as np
import cv2
from matplotlib import pyplot as plt

loc = "C:\\Users\\parth\\test_pp\\1350_left.jpeg"
img = cv2.imread(loc)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
#marker = cv2.watershed(img,thresh)

plt.imshow(thresh)
plt.show()
