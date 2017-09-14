import cv2
import numpy as np

for i in range(1,5):

   # 
   flickr = cv2.imread('result_images/flickr_crop'+str(i)+'_og.png')
   cgan   = cv2.imread('result_images/cgan_crop'  +str(i)+'_og.png')
   ugan   = cv2.imread('result_images/ugan_crop'  +str(i)+'_og.png')
   uganp  = cv2.imread('result_images/uganp_crop' +str(i)+'_og.png')

   flickr_hist = cv2.calcHist([flickr], [0,1,2], None, [8,8,8], [0,256,0,256,0,256])
   cgan_hist   = cv2.calcHist([cgan],   [0,1,2], None, [8,8,8], [0,256,0,256,0,256])
   ugan_hist   = cv2.calcHist([ugan],   [0,1,2], None, [8,8,8], [0,256,0,256,0,256])
   uganp_hist  = cv2.calcHist([uganp],  [0,1,2], None, [8,8,8], [0,256,0,256,0,256])

   a = cv2.compareHist(flickr_hist, cgan_hist,  cv2.HISTCMP_BHATTACHARYYA)
   b = cv2.compareHist(flickr_hist, ugan_hist,  cv2.HISTCMP_BHATTACHARYYA)
   c = cv2.compareHist(flickr_hist, uganp_hist, cv2.HISTCMP_BHATTACHARYYA)

   flickr = flickr/255.0
   cgan   = cgan/255.0
   ugan   = ugan/255.0
   uganp  = uganp/255.0

   print 'crop',i
   print
   print 'flickr_mean :',np.mean(flickr)
   print 'cgan_mean   :',np.mean(cgan)
   print 'ugan_mean   :',np.mean(ugan)
   print 'uganp_mean  :',np.mean(uganp)
   print
   print 'flickr_var  :',np.var(flickr)
   print 'cgan_var    :',np.var(cgan)
   print 'ugan_var    :',np.var(ugan)
   print 'uganp_var   :',np.var(uganp)
   print
   print 'flickr_cgan:',a
   print 'flickr_ugan:',b
   print 'flickr_uganp:',c
   print 
   print '---'
   print

