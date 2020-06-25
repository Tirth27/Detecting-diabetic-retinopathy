from tqdm import tqdm
import shutil
import glob
import os

dst = "D:\\Final Year Project\\crops\\Train\\4"
src = "D:\\Final Year Project\\crops\\trmp\\" ##remember to add \\ as we'r accessing folder
for file in tqdm(glob.glob(src + "*.jpeg")):
  shutil.copy(file,dst)
print("done")
