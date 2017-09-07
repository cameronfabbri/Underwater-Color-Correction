import cv2
import numpy as np

'''
# original image
img = cv2.imread('images/n01917289_4982_real.png',0)

# cyclegan image
cimg = cv2.imread('images/n01917289_4982.png',0)

# our image
uimg = cv2.imread('images/n01917289_4982_gen_IG_0.0.png',0)
'''

import glob

# real images
imgs = []

# cyclegan images
cimgs = []

# our ugan images
uimgs = []

images = sorted(glob.glob('images/*.png'))
for i in images:
   img = cv2.imread(i)

   if 'real' in i: imgs.append(img)
   elif 'IG_0.0' in i: uimgs.append(img)
   elif 'real' not in i and 'IG_1.0' not in i and 'IG_0.0' not in i: cimgs.append(img)

imgs = np.asarray(imgs)
cimgs = np.asarray(cimgs)
uimgs = np.asarray(uimgs)

for i,c,u in zip(imgs, cimgs, uimgs):

   # edges of original
   edges = cv2.Canny(i,100,200)/255.0

   # edges of cyclegan
   cedges = cv2.Canny(c,100,200)/255.0

   # edges of ugan
   uedges = cv2.Canny(u,100,200)/255.0

   print 'cyclegan: ',np.linalg.norm(edges-cedges)
   print 'ugan:     ',np.linalg.norm(edges-uedges)
   print

   cv2.imwrite('img.png', i)
   cv2.imwrite('edges.png', 255.0*edges)
   cv2.imwrite('cedges.png', 255.0*cedges)
   cv2.imwrite('uedges.png', 255.0*uedges)

   exit()

cv2.imwrite('edges.png', edges)
cv2.imwrite('cedges.png', cedges)
cv2.imwrite('uedges.png', uedges)

edges = edges/255.0
cedges = cedges/255.0
uedges = uedges/255.0

print np.linalg.norm(edges-cedges)
print np.linalg.norm(edges-uedges)



