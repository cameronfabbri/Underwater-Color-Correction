'''

   Script to zoom in on parts of an image and save out

   This will be used for comparing to CycleGAN images

   Might also be used for comparing local image stats

'''

import scipy.misc as misc
import numpy as np
import sys


#img  = 'n01496331_15872.png'
#img  = 'n01496331_2629.png'

# possible one to use - stingray
#img = '2425754085_bcfa73d160_o.png'

# lots of stuff
#img = '6959211760_295f0e300e_o.png'

# pretty good fish and corral
img = '8563897707_0a8be6d5f1_o.png'

# orange coral
#img = '5538756669_6a5627ea9c_o.png'

#img = '8564948684_dbcf221d1e_o.png'

name = img.split('.')[0]
root = 'test_images/'
#dirs = ['cgan/', 'original/', 'ugan_fl_0.0/', 'ugan__fl_1.0/']
dirs = ['cgan/', 'flickr/', 'ugan_fl_0.0/', 'ugan_fl_1.0/']

for d in dirs:

   image = misc.imread(root+d+img)

   name = d.split('/')[0]   

   name = name.replace('ugan_fl_0.0','ugan')
   name = name.replace('ugan_fl_1.0','uganp')

   # TODO when I am comparing metrics, need to save out crops without the borders
   ######################################################### red box
   y1 = 125
   y2 = 155
   x1 = 120
   x2 = 150
   # row, column (y, x), (0,0) at top left
   crop = image[y1:y2, x1:x2, :]
   crop = misc.imresize(crop, (64, 64))
   # save out before coloring edges
   misc.imsave('result_images/'+name+'_crop1_og.png', crop)
   # coloring the crops to match the boxes
   crop[0:,0,:]  = [255,0,0] # left
   crop[0:,63,:] = [255,0,0] # right
   crop[0,0:,:]  = [255,0,0] # top
   crop[63,0:,:] = [255,0,0] # bot
   misc.imsave('result_images/'+name+'_crop1.png', crop)
   # draw box around cropped area
   image[y1:y2, x1, :]   = [255, 0, 0] # left
   image[y1, x1:x2, :]   = [255, 0, 0] # top
   image[y1:y2, x2, :]   = [255, 0, 0] # right
   image[y2, x1:x2+1, :] = [255, 0, 0] # bottom


   ######################################################### blue box
   y1 = 170
   y2 = 200
   x1 = 60
   x2 = 90
   crop = image[y1:y2, x1:x2, :]
   crop = misc.imresize(crop, (64, 64))
   misc.imsave('result_images/'+name+'_crop2_og.png', crop)
   crop[0:,0,:]  = [0,0,255] # left
   crop[0:,63,:] = [0,0,255] # right
   crop[0,0:,:]  = [0,0,255] # top
   crop[63,0:,:] = [0,0,255] # bot
   misc.imsave('result_images/'+name+'_crop2.png', crop)
   image[y1:y2, x1, :]   = [0, 0, 255]
   image[y1, x1:x2, :]   = [0, 0, 255]
   image[y1:y2, x2, :]   = [0, 0, 255]
   image[y2, x1:x2+1, :] = [0, 0, 255]
   
   
   ######################################################### green box
   y1 = 75
   y2 = 90
   x1 = 90
   x2 = 105
   crop = image[y1:y2, x1:x2, :]
   crop = misc.imresize(crop, (64, 64))
   misc.imsave('result_images/'+name+'_crop3_og.png', crop)
   crop[0:,0,:]  = [0,255,0] # left
   crop[0:,63,:] = [0,255,0] # right
   crop[0,0:,:]  = [0,255,0] # top
   crop[63,0:,:] = [0,255,0] # bot
   misc.imsave('result_images/'+name+'_crop3.png', crop)
   image[y1:y2, x1, :]   = [0, 255, 0]
   image[y1, x1:x2, :]   = [0, 255, 0]
   image[y1:y2, x2, :]   = [0, 255, 0]
   image[y2, x1:x2+1, :] = [0, 255, 0]

   ######################################################### orange box
   y1 = 180
   y2 = 210
   x1 = 175
   x2 = 205
   crop = image[y1:y2, x1:x2, :]
   crop = misc.imresize(crop, (64, 64))
   misc.imsave('result_images/'+name+'_crop4_og.png', crop)
   crop[0:,0,:]  = [255,140,0] # left
   crop[0:,63,:] = [255,140,0] # right
   crop[0,0:,:]  = [255,140,0] # top
   crop[63,0:,:] = [255,140,0] # bot
   misc.imsave('result_images/'+name+'_crop4.png', crop)
   image[y1:y2, x1, :]   = [255, 140, 0]
   image[y1, x1:x2, :]   = [255, 140, 0]
   image[y1:y2, x2, :]   = [255, 140, 0]
   image[y2, x1:x2+1, :] = [255, 140, 0]
   
   misc.imsave('result_images/'+name+'_cmp.png', image)

   
   ######################################################### orange box
   '''
   y1 = 50
   y2 = 80
   x1 = 145
   x2 = 175
   crop = image[y1:y2, x1:x2, :]
   crop = misc.imresize(crop, (64, 64))
   crop[0:,0,:]  = [255,140,0] # left
   crop[0:,63,:] = [255,140,0] # right
   crop[0,0:,:]  = [255,140,0] # top
   crop[63,0:,:] = [255,140,0] # bot
   misc.imsave('result_images/'+name+'_crop5.png', crop)
   image[y1:y2, x1, :]   = [255, 140, 0]
   image[y1, x1:x2, :]   = [255, 140, 0]
   image[y1:y2, x2, :]   = [255, 140, 0]
   image[y2, x1:x2+1, :] = [255, 140, 0]


   misc.imsave('result_images/'+name+'.png', image)
   '''

