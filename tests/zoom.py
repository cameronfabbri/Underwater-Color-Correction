'''

   Script to zoom in on parts of an image and save out

'''

import scipy.misc as misc
import numpy as np
import sys


#img  = 'n01496331_15872.png'
img  = 'n01496331_2629.png'
name = img.split('.')[0]
root = 'test_images/'
dirs = ['cgan/', 'original/', 'ugan_0.0/', 'ugan_1.0/']

for d in dirs:

   image = misc.imread(root+d+img)

   #########################################################
   y1 = 160
   y2 = 200
   x1 = 210
   x2 = 250
   # row, column (y, x), (0,0) at top left
   crop = image[y1:y2, x1:x2, :]
   crop = misc.imresize(crop, (64, 64))
   misc.imsave('result_images/'+name+'_'+d.split('/')[0]+'_crop_1.png', crop)
   # draw box around cropped area
   image[y1:y2, x1, :]   = [255, 0, 0] # left
   image[y1, x1:x2, :]   = [255, 0, 0] # top
   image[y1:y2, x2, :]   = [255, 0, 0] # right
   image[y2, x1:x2+1, :] = [255, 0, 0] # bottom


   #########################################################
   y1 = 225
   y2 = 245
   x1 = 80
   x2 = 100
   crop = image[y1:y2, x1:x2, :]
   crop = misc.imresize(crop, (64, 64))
   misc.imsave('result_images/'+name+'_'+d.split('/')[0]+'_crop_2.png', crop)
   image[y1:y2, x1, :]   = [0, 0, 255]
   image[y1, x1:x2, :]   = [0, 0, 255]
   image[y1:y2, x2, :]   = [0, 0, 255]
   image[y2, x1:x2+1, :] = [0, 0, 255]
   
   
   #########################################################
   y1 = 118+30-3
   y2 = 128+30-3
   x1 = 40-2
   x2 = 50-2
   crop = image[y1:y2, x1:x2, :]
   crop = misc.imresize(crop, (64, 64))
   misc.imsave('result_images/'+name+'_'+d.split('/')[0]+'_crop_3.png', crop)
   image[y1:y2, x1, :]   = [0, 255, 0]
   image[y1, x1:x2, :]   = [0, 255, 0]
   image[y1:y2, x2, :]   = [0, 255, 0]
   image[y2, x1:x2+1, :] = [0, 255, 0]

   #########################################################
   y1 = 110
   y2 = 140
   x1 = 125
   x2 = 155
   crop = image[y1:y2, x1:x2, :]
   crop = misc.imresize(crop, (64, 64))
   misc.imsave('result_images/'+name+'_'+d.split('/')[0]+'_crop_4.png', crop)
   image[y1:y2, x1, :]   = [255, 255, 0]
   image[y1, x1:x2, :]   = [255, 255, 0]
   image[y1:y2, x2, :]   = [255, 255, 0]
   image[y2, x1:x2+1, :] = [255, 255, 0]
   
   misc.imsave('result_images/'+name+'_'+d.split('/')[0]+'.png', image)


