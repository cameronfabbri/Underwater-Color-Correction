import scipy.misc as misc
import os
import fnmatch

img_dir = 'datasets/'

pattern   = '*.jpg'
image_paths = []
for d, s, fList in os.walk(img_dir):
   for filename in fList:
      if fnmatch.fnmatch(filename, pattern):
         p = os.path.join(d, filename)

         img = misc.imread(p)

         if len(img.shape) < 3:
            #os.remove(p)
            print 'deleted',p

