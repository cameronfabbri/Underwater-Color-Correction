import scipy.misc as misc
import math
import numpy as np
import cv2

for i in range(1,5):

   flickr = cv2.imread('result_images/flickr_crop'+str(i)+'_og.png')/255.0
   cgan   = cv2.imread('result_images/cgan_crop'  +str(i)+'_og.png')/255.0
   ugan   = cv2.imread('result_images/ugan_crop'  +str(i)+'_og.png')/255.0
   uganp  = cv2.imread('result_images/uganp_crop' +str(i)+'_og.png')/255.0

   print 'crop',i
   print 'Variance'
   print 'real:  ',np.var(flickr)
   print 'cgan:  ',np.var(cgan)
   print 'ugan:  ',np.var(ugan)
   print 'uganp: ',np.var(uganp)
   print
   print 'Mean'
   print 'real:  ',np.mean(flickr)
   print 'cgan:  ',np.mean(cgan)
   print 'ugan:  ',np.mean(ugan)
   print 'uganp: ',np.mean(uganp)
   print
   print '---'
   print
