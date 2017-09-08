'''

Script to create a nxm matrix of images

'''

import cv2
import numpy as np
import scipy.misc as misc
from glob import glob

if __name__ == '__main__':

   n = 2 # cols
   m = 4  # rows

   num_images = n*m

   img_size = (256, 256, 3)

   canvas = 255*np.ones((m*img_size[0]+(10*m)+10, n*img_size[1]+(10*n)+10, 3), dtype=np.uint8)
   
   start_x = 10
   start_y = 10

   x = 0
   y = 0

   image_paths = sorted(glob('result_images/*.png'))

   count = 0
   for impath in image_paths:

      print impath
      if count == 7: break
      count += 1
   exit()

   for img in gen_imgs:

      end_x = start_x+256
      end_y = start_y+256
      
      canvas[start_y:end_y, start_x:end_x, :] = img

      if x < n:
         start_x += 256+10
         x += 1
      if x == n:
         x       = 0
         start_x = 10
         start_y = end_y + 10
         end_y   = start_y+256

   misc.imsave('results.jpg', canvas)
   #cv2.imwrite('results.jpg', canvas)
   #cv2.imshow('canvas', canvas)
   #cv2.waitKey(0)
   #cv2.destroyAllWindows()
