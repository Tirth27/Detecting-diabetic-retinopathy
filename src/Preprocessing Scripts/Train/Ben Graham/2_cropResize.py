#convert images to 1800x1800 pixels
import PIL
from skimage import io
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True      #to overcome truncation error.
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

def crop_and_resize(path, new_path, crop_size, resize_size):
    '''
    Crop images from directory in a new directory.
    
    Input-
        path:- Images which are being processed
        new_path:- Place where images are saved after processing 
        crop_size:- DEFAULT-750, can changes as per need
        resize_size:- resize cropped images to given pixels
        
    Output-
        All images cropped, resized, and saved from the old folder to the new folder. 
    '''
    create_directory(new_path)
    dire = [l for l in os.listdir(path) if l != '.DS_Store'] #For MacUsers
    
    for items in dire:
        #reading the image
        image_read = io.imread(path + items) #skimage
        image = Image.open(path + items) #PIL
        
        #return dimensions of image        
        #    y,     x, channel = image_read.shape
        height, width, channel = image_read.shape
        
        '''formula to center the image'''
        #upper left corner
        left = width//2 - crop_size//2
        top = height//2 - crop_size//2
        #lower right corner
        bottom = width//2 + crop_size//2
        right = height//2 + crop_size//2
        
        #crop the image to given pixels
        image = image.crop((left, top, bottom, right))
        
        #reize image
        image = image.resize((resize_size, resize_size), PIL.Image.LANCZOS)
        
        #saving the image
        image.save(str(new_path + items)) #PIL        
    
    #time elapsed    
    print("--- %s seconds ---" % (time.time() - start_time_first))
    
#by doing this the code cannot be use as module    
if __name__ == "__main__":
    #Resize to 256x256
    crop_and_resize(path = "E:\\Tirth\\train\\remove_boundary_Original\\",
         crop_size = 750, resize_size = 256,
         new_path = "E:\\Tirth\\train\\remove_boundary_256\\")
    
    #Resize to 512x512
    crop_and_resize(path = "E:\\Tirth\\train\\remove_boundary_Original\\",
         crop_size = 750, resize_size = 512,
         new_path = "E:\\Tirth\\train\\remove_boundary_512\\")
