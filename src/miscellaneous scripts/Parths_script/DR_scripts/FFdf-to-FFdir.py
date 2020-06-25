'''
we have data in 5 catagories
0 - No DR
1 - Mild
2 - Moderate
3 - Severe
4 - Proliferative DR

workflow of separating images according to their
severity and storing them in separate folder:

1: load csv and store labels in a separate list
   according to catagories.
2: move the listed label's photos to new catagorical
   folders
'''
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import shutil

print(__doc__)

csv_dir = r"D:\Final Year Project\label\trainlabel_master_v2.csv"

train_dir = "D:\\Final Year Project\\crops\\crop_512\\"
##train_dir = r"D:\Final Year Project\crops\remove_boundary_512"
'''
While coding this i came across a problem where crop_512 and remove_boundary_512 have
exactly same naming of images but the images are differently preprocessed so while copying
images using shutil it just replaces images.
--------------------------------------------
so we need to rename images bcz ultimately image name dosen't matter in FFdir
---------------------------------------------------------------------------
we'll use rename method from os to do this
rename every image to DR0,DR1,DR2,DR3 and DR4.
run this block of code for the same
'''
##i=0
##for filename in os.listdir(train_dir):
##  dst ="DR" + str(i) + ".jpeg"
##  src = train_dir + filename
##  dst = train_dir + dst
##
##  # rename() function will
##  # rename all the files
##  os.rename(src, dst)
##  i += 1


No_DR_Dir = r"D:\Final Year Project\crops\Train\0"
Mild_DR_Dir = r"D:\Final Year Project\crops\Train\1"
Moderate_DR_Dir = r"D:\Final Year Project\crops\Train\2"
Severe_DR_Dir = r"D:\Final Year Project\crops\Train\3"
Proliferative_DR_Dir = r"D:\Final Year Project\crops\Train\4"


pd_csv = pd.read_csv(csv_dir)
#print(pd_csv.image)

'''
make list of train_image_name with their respective level
'''
list_of_0 = []
list_of_1 = []
list_of_2 = []
list_of_3 = []
list_of_4 = []

##save train_image_name(label name) according to their level in lists

for index, row in pd_csv.iterrows():
    if row['level']==4:
        list_of_4.append(row['train_image_name'])
    elif row['level']==3:
        list_of_3.append(row['train_image_name'])
    elif row['level']==2:
        list_of_2.append(row['train_image_name'])
    elif row['level']==1:
        list_of_1.append(row['train_image_name'])
    elif row['level']==0:
        list_of_0.append(row['train_image_name'])
    else:
        pass
'''
You can call this fn to rename images
'''
# def rename_image(new_label,i):
#     # print('''
#     #         Label name will be converted as per
#     #         YourLabelName+0,1,2...
#     #         In this case label name is DR4_
#     #         so rename will be DR4_0,DR4_1...
#     #         ''')
#     # new_label = input("Enter label name => ")
#     q=0
#     pre_dst = new_label + str(q) + ".jpeg"
#     src = train_dir + i
#     dst = train_dir + pre_dst
#     os.rename(src, dst)
#     q += 1
#     return pre_dst

q=0
for i in list_of_4:
 pre_dst ="DR4_" + str(q) + ".jpeg"
 src = train_dir + i
 dst = train_dir + pre_dst
 os.rename(src, dst)
 q += 1
 shutil.copy(os.path.join(train_dir,pre_dst),Proliferative_DR_Dir)
print('done for 4')

for i in tqdm(list_of_3):
 pre_dst ="DR3_" + str(q) + ".jpeg"
 src = train_dir + i
 dst = train_dir + pre_dst
 os.rename(src, dst)
 q += 1
 shutil.copy(os.path.join(train_dir,pre_dst),Severe_DR_Dir)
print('done for 3')
q=0

for i in tqdm(list_of_2):
 pre_dst ="DR2_" + str(q) + ".jpeg"
 src = train_dir + i
 dst = train_dir + pre_dst
 os.rename(src, dst)
 q += 1
 shutil.copy(os.path.join(train_dir,pre_dst),Moderate_DR_Dir)
print('done for 2')
q=0

for i in tqdm(list_of_1):
  pre_dst ="DR1-" + str(q) + ".jpeg"
  src = train_dir + i
  dst = train_dir + pre_dst
  os.rename(src, dst)
  q += 1
  shutil.copy(os.path.join(train_dir,pre_dst),Mild_DR_Dir)
print('done for 1')

for i in list_of_0:
 shutil.copy(os.path.join(train_dir,i),No_DR_Dir)
print('done for 0')

print('all done GTG')
