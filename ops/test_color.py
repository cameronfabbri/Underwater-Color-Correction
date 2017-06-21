from scipy import misc
from skimage import color
import cv2
import numpy as np

import time
s = time.time()

color_imgs = np.empty((64, 218, 178, 3), dtype=np.float32)
gray_imgs = np.empty((64, 218, 178, 1), dtype=np.float32)

for i in range(1):
   # read image
   img = misc.imread('rgb.JPEG')
   img = misc.imresize(img, (256, 256))
   img = color.rgb2lab(img)
   # convert to gray
   #gray_img = color.rgb2gray(img)

   # resize
   #img = misc.imresize(img, (256, 256))
   #gray_img = misc.imresize(gray_img, (256, 256))
   #gray_img = np.expand_dims(gray_img, 2) 
   # convert to LAB
   #img = color.rgb2lab(img)

   # normalize
   img = img/127.5 - 1.
   #gray_img = gray_img/127.5 - 1.
  
   #color_imgs[i, ...] = img
   #gray_imgs[i, ...] = gray_img
   print img
   exit()
   img = (img+1.0)*127.5
   img = color.lab2rgb(np.float64(img))
   #img = 255.0*color.lab2rgb(np.float64(img))/img.max()


#print time.time()-s
#exit()
#misc.imsave('lab.jpg', img)
#img = misc.imread('lab.jpg')
#img = 255.0*color.lab2rgb(np.uint8(img))
misc.imsave('conv.jpg', img)
#misc.imsave('gray.JPEG', gray)

