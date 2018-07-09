'''

   Interpolates between two images

'''

import scipy.misc as misc
import numpy as np
import sys
   

if __name__ == '__main__':

   img1 = misc.imread(sys.argv[1])
   img2 = misc.imread(sys.argv[2])

   alpha = np.linspace(0,1,5)

   images = []
   for a in alpha:
      images.append( (img1*(1-a)+img2*a) )

   images = np.asarray(images)

   i = 0
   for img in images:
      misc.imsave('image_interp/image_interp_'+str(i)+'.png', img)
      i+=1
