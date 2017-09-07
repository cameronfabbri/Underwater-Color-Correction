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

   #print 'cyclegan: ',np.linalg.norm(edges-cedges)
   #print 'ugan_0:   ',np.linalg.norm(edges-u0edges)
   #print 'ugan_1:   ',np.linalg.norm(edges-u1edges)
   #print

   #cv2.imwrite('img.png', i)
   #cv2.imwrite('edges.png', 255.0*edges)
   #cv2.imwrite('cedges.png', 255.0*cedges)
   #cv2.imwrite('uedges.png', 255.0*uedges)

print np.mean(cgan_dist)
print np.mean(ugan_0_dist)
print np.mean(ugan_1_dist)

exit()


