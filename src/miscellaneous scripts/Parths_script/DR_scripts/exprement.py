from tqdm import tqdm
import glob
import numpy as np
import cv2
import shutil
from imgaug import augmenters as iaa

seq = iaa.Sequential([iaa.Flipud(0.5)])
loc = "C:\\Users\\parth\\test_pp"
new_loc = "C:\\Users\\parth\\test_pp\\"
imglist = []
print('reading')
for file in tqdm(glob.glob(loc+"\\"+"*.jpeg")):   
  img = cv2.imread(file) 
  imglist.append(img)

images_aug = seq.augment_images(imglist)

print('writing')
q=0
for i in tqdm(range(len(imglist))):
  cv2.imwrite(new_loc+ '-' + str(q) +'.jpeg' , images_aug[i])
  q+=1 
