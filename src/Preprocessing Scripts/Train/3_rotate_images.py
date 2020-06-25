'''
 NOTE:- store the images after rotation to same folder from where it 
        fetched i.e MAKE "new_path_to_store = old_path" or simply copy
        and paste value from old_path to new_path_to_store(I HAVE TO TELL YOU
        EVERYTHING, SO NOOB)
'''

import pandas as pd
import time
from skimage import io
from skimage.transform import rotate
import cv2


def rotate_images(image_path, degrees_of_rotation, list_images):
    '''
        Rotate the images to specified Degrees°
        
        Input:
            path:- where  original images are stored
            new_path:- location to save images after rotation
            label_DR:- contains label for images
            degrees_of_rotation:- degree to which image is rotated
        
        Output:
            Images rotated with degrees specified  
    '''
    for label in list_images:
        #get image
        image = io.imread(image_path + str(label) + '.jpeg')
        #rotate to specified degree 
        image = rotate(image, degrees_of_rotation)
        #save it
        io.imsave(image_path + str(label) + '_' + str(degrees_of_rotation) + '.jpeg',
                 image)
    
    
def mirror_images(image_path, list_image):
    '''
        Mirror images
        
        Input:
            path:- location where original images are stored
            new_path:- location to save images after mirror
            label_DR:- contains label for images
            
        Output:
            Mirrored images
    '''
    for label in list_image:
        #get image
        image = cv2.imread(image_path + str(label) + '.jpeg')
        #flip it
        image = cv2.flip(image, 1)
        #save it
        cv2.imwrite(image_path + str(label) + '_mir' + '.jpeg', image)

if __name__ == "__main__":
    #start the timer
    start_time_Final = time.time()
    #read csv(INCLUDE trainLabels_find_black_images.csv)
    trainLabels = pd.read_csv("E:/Tirth/trainlabel_zip/trainLabels_csv/trainLabels_find_black_images.csv")
    
    #add .jpeg
    trainLabels['image'] = trainLabels['image'].str.rstrip('.jpeg')
    #labels which doesn't have DR
    trainLabels_no_DR = trainLabels[trainLabels['level'] == 0]
    #labels with DR
    trainLabels_DR = trainLabels[trainLabels['level'] >= 1]
    
    list_images_no_DR = [i for i in trainLabels_no_DR['image']]
    list_images_DR = [i for i in trainLabels_DR['image']]
    
    path_224 = 'E:/Tirth/train/crop_224/'
    path_256 = 'E:/Tirth/train/crop_256/'
    path_299 = 'E:/Tirth/train/crop_299/'
    path_512 = 'E:/Tirth/train/crop_512/'
    
    #mirror images having no-DR
    print("--224--Mirroring Non-DR Images")
    mirror_images(image_path = path_224, list_image = list_images_no_DR)
    print("--256--Mirroring Non-DR Images")
    mirror_images(image_path = path_256, list_image = list_images_no_DR)
    print("--299--Mirroring Non-DR Images")
    mirror_images(image_path = path_299, list_image = list_images_no_DR)
    print("--512--Mirroring Non-DR Images")
    mirror_images(image_path = path_512, list_image = list_images_no_DR)
    
    #rotate images having DR
    #rotate labels to 90°
    print("--224--Rotating DR Images to 90 Degrees") 
    rotate_images(image_path = path_224, degrees_of_rotation = 90, list_images = list_images_DR)
    print("--256--Rotating DR Images to 90 Degrees") 
    rotate_images(image_path = path_256, degrees_of_rotation = 90, list_images = list_images_DR)
    print("--299--Rotating DR Images to 90 Degrees") 
    rotate_images(image_path = path_299, degrees_of_rotation = 90, list_images = list_images_DR)
    print("--512--Rotating DR Images to 90 Degrees") 
    rotate_images(image_path = path_512, degrees_of_rotation = 90, list_images = list_images_DR)
    
    #rotate labels to 120°
    print("--224--Rotating DR Images to 120 Degrees")
    rotate_images(image_path = path_224, degrees_of_rotation = 120, list_images = list_images_DR)
    print("--256--Rotating DR Images to 120 Degrees")
    rotate_images(image_path = path_256, degrees_of_rotation = 120, list_images = list_images_DR)
    print("--299--Rotating DR Images to 120 Degrees")
    rotate_images(image_path = path_299, degrees_of_rotation = 120, list_images = list_images_DR)
    print("--512--Rotating DR Images to 120 Degrees")
    rotate_images(image_path = path_512, degrees_of_rotation = 120, list_images = list_images_DR)
    
    #rotate labels to 180°
    print("--224--Rotating DR Images to 180 Degrees")
    rotate_images(image_path = path_224, degrees_of_rotation = 180, list_images = list_images_DR)
    print("--256--Rotating DR Images to 180 Degrees")
    rotate_images(image_path = path_256, degrees_of_rotation = 180, list_images = list_images_DR)
    print("--299--Rotating DR Images to 180 Degrees")
    rotate_images(image_path = path_299, degrees_of_rotation = 180, list_images = list_images_DR)
    print("--512--Rotating DR Images to 180 Degrees")
    rotate_images(image_path = path_512, degrees_of_rotation = 180, list_images = list_images_DR)
    
    #rotate labels to 270°
    print("--224--Rotating DR Images to 270 Degrees")
    rotate_images(image_path = path_224, degrees_of_rotation = 270, list_images = list_images_DR)
    print("--256--Rotating DR Images to 270 Degrees")
    rotate_images(image_path = path_256, degrees_of_rotation = 270, list_images = list_images_DR)
    print("--299--Rotating DR Images to 270 Degrees")
    rotate_images(image_path = path_299, degrees_of_rotation = 270, list_images = list_images_DR)
    print("--512--Rotating DR Images to 270 Degrees")
    rotate_images(image_path = path_512, degrees_of_rotation = 270, list_images = list_images_DR)
    
    #mirror images having DR
    print("--224--Mirroring Images Having DR")
    mirror_images(image_path = path_224, list_image = list_images_DR)
    print("--256--Mirroring Images Having DR")
    mirror_images(image_path = path_256, list_image = list_images_DR)
    print("--299--Mirroring Images Having DR")
    mirror_images(image_path = path_299, list_image = list_images_DR)
    print("--512--Mirroring Images Having DR")
    mirror_images(image_path = path_512, list_image = list_images_DR)
    
    #time elapsed
    print("Done With It")    
    print("--- %s seconds ---" % (time.time() - start_time_Final))  
