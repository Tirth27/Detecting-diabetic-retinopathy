import cv2
import numpy as np
import os
import time

#start the timer
start_time_first = time.time()

def create_directory(directory):
    '''
    Creates a new folder in the specified directory if the folder doesn't exist.
    INPUT
        directory: Folder to be created, called as "folder/".
    OUTPUT
        New folder in the current directory.
    '''
    if not os.path.exists(directory):
        os.makedirs(directory)

def scaleRadius(img, scale):
    x = img[img.shape[0] / 2,:,:].sum(1)
    r = (x>x.mean()/10).sum()/2
    #print r
    s = (scale * 1.0)/r
    
    return cv2.resize(img, (0, 0), fx = s , fy = s )

def remove_boundary_effect(path, new_path, scale):
    '''
    Crops, resizes, and stores all images from a directory in a new directory.
    INPUT
        path: Path where the current, unscaled images are contained.
        new_path: Path to save the resized images.
        cropx, cropy: Initial size for cropping.
        img_size: New size for the rescaled images.
    OUTPUT
        All images cropped, resized, and saved from the old folder to the new folder.
    '''    
    create_directory(new_path)
    dire = [l for l in os.listdir(path) if l != '.DS_Store'] #For MacUsers
    
    for item in dire[22928:]:
        #print item
        a = cv2.imread(path + item)
        #scale img to a given radius
        a = scaleRadius(a, scale)
        #subtract local mean color
        a = cv2.addWeighted(a, 4, cv2.GaussianBlur(a, (0, 0), scale/30), -4, 128)
        #remove outer 10%
        b = np.zeros(a.shape)
        cv2.circle(b, (a.shape[1]/2, a.shape[0]/2), int(scale*0.9), (1, 1, 1), -1 , 8, 0)
        a = a*b + 128 * (1 - b)
        cv2.imwrite(str(new_path + item), a)
    
    # time elapsed    
    print("--- %s seconds ---" % (time.time() - start_time_first))
    
#by doing this the code cannot be moduled
if __name__ == "__main__":
    remove_boundary_effect(path = 'E:\\Tirth\\data\\Test\\test\\test\\',
                           scale = 500, new_path = 'E:\\Tirth\\test\\remove_boundary__test_Original\\')
    
