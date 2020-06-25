import pandas as pd
import sys
import tqdm
import numpy as np
from tqdm import tqdm
import os
import shutil

csv_dir = "D:\\Final Year Project\\label\\testLables.csv"

train_dir = "D:\\Final Year Project\\crops\\crop_test_512\\"

dst_0 = "D:\\Final Year Project\\crops\\Test\\0"
dst_1 = "D:\\Final Year Project\\crops\\Test\\1"
dst_2 = "D:\\Final Year Project\\crops\\Test\\2"
dst_3 = "D:\\Final Year Project\\crops\\Test\\3"
dst_4 = "D:\\Final Year Project\\crops\\Test\\4"

pd_csv = pd.read_csv(csv_dir)

list_of_0 = []
list_of_1 = []
list_of_2 = []
list_of_3 = []
list_of_4 = []

for index, row in pd_csv.iterrows():
  if row['level']==4:
    list_of_4.append(row['image'])
  elif row['level']==3:
    list_of_3.append(row['image'])
  elif row['level']==2:
    list_of_2.append(row['image'])
  elif row['level']==1:
    list_of_1.append(row['image'])
  elif row['level']==0:
    list_of_0.append(row['image'])
  else:
    pass

for i in tqdm(list_of_4):
  shutil.copy(os.path.join(train_dir,i +'.jpeg'),dst_4)
print('done for 4')
for i in tqdm(list_of_3):
  shutil.copy(os.path.join(train_dir,i +'.jpeg'),dst_3)
print('done for 3')
for i in tqdm(list_of_2):
  shutil.copy(os.path.join(train_dir,i +'.jpeg'),dst_2)
print('done for 2')
for i in tqdm(list_of_1):
  shutil.copy(os.path.join(train_dir,i +'.jpeg'),dst_1)
print('done for 1')
for i in tqdm(list_of_0):
  shutil.copy(os.path.join(train_dir,i +'.jpeg'),dst_0)
print('done for 0')
