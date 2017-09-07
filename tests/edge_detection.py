import cv2
import numpy as np
import glob
import ntpath
from tqdm import tqdm

# real images
imgs = []

# cyclegan images
cimgs = []

# our ugan images
uimgs = []

# all the image paths
real_paths   = sorted(glob.glob('test_images/original/*.png'))
cgan_paths   = sorted(glob.glob('test_images/cgan/*.png'))
ugan_0_paths = sorted(glob.glob('test_images/ugan_0.0/*.png'))
ugan_1_paths = sorted(glob.glob('test_images/ugan_1.0/*.png'))

# respective distances in image space of canny edge detection
cgan_dist   = []
ugan_0_dist = []
ugan_1_dist = []

# loop through all paths and open images
for r,c,u0,u1 in tqdm(zip(real_paths, cgan_paths, ugan_0_paths, ugan_1_paths)):

   image_name = ntpath.basename(r)

   #if image_name != 'n01496331_11938.png': continue
   if image_name != 'n01917289_4078.png': continue


   rimg  = cv2.imread(r)
   cimg  = cv2.imread(c)
   u0img = cv2.imread(u0)
   u1img = cv2.imread(u1)

   # edges of original
   edges = cv2.Canny(rimg,100,200)/255.0


   # edges of cyclegan
   cedges = cv2.Canny(cimg,100,200)/255.0

   # edges of ugan 0.0
   u0edges = cv2.Canny(u0img,100,200)/255.0

   # edges of ugan 1.0
   u1edges = cv2.Canny(u1img,100,200)/255.0

   cgan_dist.append(np.linalg.norm(edges-cedges))
   ugan_0_dist.append(np.linalg.norm(edges-u0edges))
   ugan_1_dist.append(np.linalg.norm(edges-u1edges))
   
   print 'cgan:   ',np.linalg.norm(edges-cedges)
   print 'ugan_0: ',np.linalg.norm(edges-u0edges)
   print 'ugan_1: ',np.linalg.norm(edges-u1edges)
   print

   # save a bunch of images for paper
   # original images
   cv2.imwrite('original.png', rimg)
   cv2.imwrite('cimg.png', cimg)
   cv2.imwrite('u0img.png', u0img)
   cv2.imwrite('u1img.png', u1img)

   # edges of original image
   cv2.imwrite('edges.png', 255.0*edges)
   
   # edges of cgan
   cv2.imwrite('cedges.png', 255.0*cedges)

   # edges of ugan 0.0
   cv2.imwrite('u0edges.png', 255.0*u0edges)
   
   # edges of ugan 1.0
   cv2.imwrite('u1edges.png', 255.0*u1edges)


   exit()


   #cv2.imwrite('img.png', i)
   #cv2.imwrite('edges.png', 255.0*edges)
   #cv2.imwrite('cedges.png', 255.0*cedges)
   #cv2.imwrite('uedges.png', 255.0*uedges)

print np.mean(cgan_dist)
print np.mean(ugan_0_dist)
print np.mean(ugan_1_dist)
# results
# 105.447125252
# 91.8003760136
# 91.8462828424

exit()


