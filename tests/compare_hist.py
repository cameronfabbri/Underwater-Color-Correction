import cv2
import numpy as np




flickr = cv2.imread('result_images/flickr_crop4.png')
cgan   = cv2.imread('result_images/cgan_crop4.png')
ugan   = cv2.imread('result_images/ugan_crop4.png')
uganp  = cv2.imread('result_images/uganp_crop4.png')


flickr_hist = cv2.calcHist([flickr], [0,1,2], None, [8,8,8], [0,256,0,256,0,256])
cgan_hist   = cv2.calcHist([cgan],   [0,1,2], None, [8,8,8], [0,256,0,256,0,256])
ugan_hist   = cv2.calcHist([ugan],   [0,1,2], None, [8,8,8], [0,256,0,256,0,256])
uganp_hist  = cv2.calcHist([uganp],  [0,1,2], None, [8,8,8], [0,256,0,256,0,256])

a = cv2.compareHist(flickr_hist, cgan_hist,  cv2.HISTCMP_BHATTACHARYYA)
b = cv2.compareHist(flickr_hist, ugan_hist,  cv2.HISTCMP_BHATTACHARYYA)
c = cv2.compareHist(flickr_hist, uganp_hist, cv2.HISTCMP_BHATTACHARYYA)

print a
print b
print c


