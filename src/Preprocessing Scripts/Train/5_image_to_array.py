from PIL import Image
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
import time
#np.set_printoptions(threshold=np.nan)

#da = 'floyd run --cpu --data tirth/datasets/cropimg/1:data "python image_to_array(1).py"'

def change_image_name(df, column):
    return [i + '.jpeg' for i in df[column]]

def image_to_array(file_path, df):
    image_list = [l for l in df['train_image_name']]
    return np.array([np.array(Image.open(file_path + img)) for img in image_list])

def save_to_array(arr_name, arr_object):
    return np.save(arr_name, arr_object)

if __name__ == '__main__':
    start_time = time.time()
    
    array_name = 'E:/Tirth/train/Array/master_crop_512.npy'
    
    #read csv (INCLUDE trainlabel_master_v2.csv)
    labels = pd.read_csv("E:/Tirth/trainlabel_zip/trainLabels_csv/trainlabel_master_v2.csv")

    print("Converting Image To Array")
    array = image_to_array(file_path = "E:/Tirth/train/crop_512/", df = labels)
    print(array.shape)
    
    print("Saving Train Array")
    save_to_array(arr_name = array_name , arr_object = array)
    
    print("--- %s seconds ---" % (time.time() - start_time))

    '''
    print("Saving to '.npy'")
    np.save("/Users/tirth/Documents/Diabetic Retinopathy/Sample dataset/array/uncomp_save.npy", image_array)

    print("Saving compressed '.npz' using -> savez_compressed()")
    np.savez_compressed("/Users/tirth/Documents/Diabetic Retinopathy/Sample dataset/array/comp_savez_comp.npz", image_array)

    print("Saving uncompressed '.npz' using -> savez()")
    np.savez("/Users/tirth/Documents/Diabetic Retinopathy/Sample dataset/array/uncomp_savez.npz", image_array)
    '''
    
