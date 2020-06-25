#crop and resize the image

from PIL import Image
import os
#for image read and save
from skimage import io
from skimage.transform import resize
import time

#scriptDir = os.path.dirname(__file__)
#imagePath = os.path.join(scriptDir, '/home/tirth/Diabetic Retinopathy/TirthSampleTest/126_right.jpeg')
start_time_first = time.time()


imagePath = "/home/tirth/Diabetic Retinopathy/Data Cleaning/sample/16_left.jpeg"

image = Image.open(imagePath)

img = io.imread(imagePath)
height, width, channel = img.shape

print('Before: ', height, width, channel)

cropp = 1800
#upper left corner
left = width//2 - cropp//2
top = height//2 - cropp//2

#lower right corner
bot = width//2 + cropp//2
right = height//2 + cropp//2

imageCropped = image.crop((left, top, bot, right))


extension = '.jpeg'

imageCropped.save('1800' + extension)
print("--- %s seconds ---" % (time.time() - start_time_first))
#img.show()

start_time_second = time.time()
cropx , cropy = 1800, 1800

img = io.imread(imagePath)
y,x,channel = img.shape
startx = x//2-(cropx//2)
starty = y//2-(cropy//2)
img = img[starty:starty+cropy,startx:startx+cropx]
img = resize(img, (256,256), mode='constant')
io.imsave('256.jpeg', img)
print("--- %s seconds ---" % (time.time() - start_time_second))
