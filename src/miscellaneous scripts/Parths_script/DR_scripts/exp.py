import pandas as pd
import sys
import tqdm
import numpy as np
from tqdm import tqdm
import os
import shutil


csv_dir = r"D:\Final Year Project\label\trainlabel_master_v2.csv"
train_dir = "D:\\Final Year Project\\crops\\denoise_256\\"
list_of_4 = []
Proliferative_DR_Dir = "E:\\sample_1"
pd_csv = pd.read_csv(csv_dir)




for index, row in pd_csv.iterrows():
  if row['level']==4:
    list_of_4.append(row['train_image_name'])

q=0
for i in tqdm(list_of_4): 
  pre_dst ="DR4_" + str(q) + ".jpeg"
  src = train_dir + i 
  dst = train_dir + pre_dst 
  os.rename(src, dst) 
  q += 1
  shutil.copy(os.path.join(train_dir,pre_dst),Proliferative_DR_Dir)
print('done for 4')
