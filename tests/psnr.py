import cv2
import numpy as np
import math

def psnr(img1, img2):
   mse = np.mean( (img1 - img2) ** 2 )
   return 10 * math.log10(255.0**2/mse)

for i in range(1,5):

   flickr = cv2.imread('result_images/flickr_crop'+str(i)+'_og.png')
   cgan   = cv2.imread('result_images/cgan_crop'  +str(i)+'_og.png')
   ugan   = cv2.imread('result_images/ugan_crop'  +str(i)+'_og.png')
   uganp  = cv2.imread('result_images/uganp_crop' +str(i)+'_og.png')


   print 'crop  ',i
   print 'cgan: ',psnr(flickr,cgan)
   print 'ugan: ',psnr(flickr,ugan)
   print 'uganp:',psnr(flickr,uganp)
   print



