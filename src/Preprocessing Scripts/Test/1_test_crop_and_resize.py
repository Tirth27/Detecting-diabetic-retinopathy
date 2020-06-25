from skimage import io
from PIL import ImageFile
from skimage.transform import resize
ImageFile.LOAD_TRUNCATED_IMAGES = True #to overcome truncation error.
import os
import time

#start the timer
start_time_first = time.time()

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def crop_and_resize_image(path, new_path, cropx, cropy, image_size):
    '''
    Crop, resize and store from directory in a new directory.

    Input-
        path:- Images which are being processed
        new_path:- Place where images are saved after processing
        crop_size:- DEFAULT-1800, can changes as per need

    Output-
        Cropped images having resolution 1800x1800 and resize with 256x256
    '''

    create_directory(new_path)
    dire = [l for l in os.listdir(path) if l != '.DS_Store'] #For MacUsers
    total = 0

    for item in dire:
        #reading the image
        image = io.imread(path + item) #skimage

        #return dimensions of image
        #    y,     x, channel = image.shape
        height, width, channel = image.shape

        #top
        startx = width//2 - (cropx//2)
        #bottom
        starty = height//2 - (cropy//2)

        image = image[starty: starty + cropy, startx: startx + cropx]

        #reize image
        image = resize(image, (image_size, image_size), mode = 'reflect')

        #saving the image
        io.imsave(str(new_path + item), image) #skimage

        total += 1

    # time elapsed
    print("--- %s seconds ---" % (time.time() - start_time_first))

#by doing this the code cannot be use as module
if __name__ == "__main__":
    #Resize to 256x256(dsi-capstone)
    crop_and_resize_image(path = "E:/Tirth/test/test/test/",
         new_path = "E:/Tirth/test/crop_test_256/",
         cropx = 1800, cropy = 1800, image_size = 256)

    #Resize to 224x244(VGG16)
    crop_and_resize_image(path = "E:/Tirth/test/test/test/",
         new_path = "E:/Tirth/test/crop_test_224/",
         cropx = 1800, cropy = 1800, image_size = 224)

    #Resize to 299x299(Inception-V3)
    crop_and_resize_image(path = "E:/Tirth/test/test/test/",
         new_path = "E:/Tirth/test/crop_test_299/",
         cropx = 1800, cropy = 1800, image_size = 299)

    #Resize to 512x512(Retinet)
    crop_and_resize_image(path = "E:/Tirth/test/test/test/",
         new_path = "E:/Tirth/test/crop_test_512/",
         cropx = 1800, cropy = 1800, image_size = 512)
