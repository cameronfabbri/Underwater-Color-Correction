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

save_images = ['n01496331_11763.png', 'n01496331_22079.png', 'n01496331_2629.png', 'n01914609_1607.png', 'n01664065_9210.png', 'n02607072_763.png']

largest = 0
l_img = ''

save = 0

# just open and close quick to wipe it
results_file = open('result_images/results.txt', 'w')
results_file.close()
results_file = open('result_images/results.txt', 'a')
results_file.write('image_name,cgan_dist,ugan_0_dist,ugan_1_dist\n')

# loop through all paths and open images
for r,c,u0,u1 in tqdm(zip(real_paths, cgan_paths, ugan_0_paths, ugan_1_paths)):

   if save == 1 and ntpath.basename(r) not in save_images: continue
   
   image_name = ntpath.basename(r)

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

   cgan_dist_   = np.linalg.norm(edges-cedges)
   ugan_0_dist_ = np.linalg.norm(edges-u0edges)
   ugan_1_dist_ = np.linalg.norm(edges-u1edges)

   cgan_dist.append(cgan_dist_)
   ugan_0_dist.append(ugan_0_dist_)
   ugan_1_dist.append(ugan_1_dist_)

   d = np.linalg.norm(edges-cedges) - np.linalg.norm(edges-u0edges)

   if d > largest:
      largest = d
      l_img = image_name

   if save == 1:

      print image_name
      print 'cgan:   ',np.linalg.norm(edges-cedges)
      print 'ugan_0: ',np.linalg.norm(edges-u0edges)
      print 'ugan_1: ',np.linalg.norm(edges-u1edges)
      print

      # save a bunch of images for paper
      # original images
      cv2.imwrite('result_images/'+image_name+'_original.png', rimg)
      cv2.imwrite('result_images/'+image_name+'_cimg.png', cimg)
      cv2.imwrite('result_images/'+image_name+'_u0img.png', u0img)
      cv2.imwrite('result_images/'+image_name+'_u1img.png', u1img)

      # edges of original image
      cv2.imwrite('result_images/'+image_name+'_edges.png', 255.0*edges)
      
      # edges of cgan
      cv2.imwrite('result_images/'+image_name+'_cedges.png', 255.0*cedges)

      # edges of ugan 0.0
      cv2.imwrite('result_images/'+image_name+'_u0edges.png', 255.0*u0edges)
      
      # edges of ugan 1.0
      cv2.imwrite('result_images/'+image_name+'_u1edges.png', 255.0*u1edges)

      results_file.write(image_name+','+str(cgan_dist_)+','+str(ugan_0_dist_)+','+str(ugan_1_dist_)+'\n')

results_file.close()
print 'done'
#print l_img
#print d

print np.mean(cgan_dist)
print np.mean(ugan_0_dist)
print np.mean(ugan_1_dist)
# results
# 105.447125252
# 91.8003760136
# 91.8462828424

exit()


