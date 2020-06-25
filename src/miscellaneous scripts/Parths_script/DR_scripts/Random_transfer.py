'''
this code transfers random images from train set to validation set
'''
import os
import random
import shutil
import glob
from tqdm import tqdm


def rand_gen(source):

  src = source
  
  files = os.listdir(src)
  list_files = []
  for i in range(int(len(files)/8)):
    index = random.randrange(0, len(files))
    list_files.append(index)
  rand_files=[]
  for i in list_files:
    rand_files.append(files[i])

  return rand_files
print(__doc__)
src_0 =  "D:\\Final Year Project\\crops\\Train\\0\\"
src_1 =  "D:\\Final Year Project\\crops\\Train\\1\\"
src_2 =  "D:\\Final Year Project\\crops\\Train\\2\\"
src_3 =  "D:\\Final Year Project\\crops\\Train\\3\\"
src_4 =  "D:\\Final Year Project\\crops\\Train\\4\\"

dst_0 = "D:\\Final Year Project\\crops\\Validation\\0"
dst_1 = "D:\\Final Year Project\\crops\\Validation\\1"
dst_2 = "D:\\Final Year Project\\crops\\Validation\\2"
dst_3 = "D:\\Final Year Project\\crops\\Validation\\3"
dst_4 = "D:\\Final Year Project\\crops\\Validation\\4"

rand_files_0 = rand_gen(src_0)
for img in tqdm(rand_files_0):
  shutil.copy(os.path.join(src_0, img), dst_0)
print('done for 0')

rand_files_1 = rand_gen(src_1)
for img in tqdm(rand_files_1):
  shutil.copy(os.path.join(src_1, img), dst_1)
print('done for 1')

rand_files_2 = rand_gen(src_2)
for img in tqdm(rand_files_2):
  shutil.copy(os.path.join(src_2, img), dst_2)
print('done for 2')

rand_files_3 = rand_gen(src_3)
for img in tqdm(rand_files_3):
  shutil.copy(os.path.join(src_3, img), dst_3)
print('done for 3')

rand_files_4 = rand_gen(src_4)
for img in tqdm(rand_files_4):
  shutil.copy(os.path.join(src_4, img), dst_4)
print('done for 4')

