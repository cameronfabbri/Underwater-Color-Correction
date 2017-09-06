import scipy.misc as misc
import math
import numpy as np

# our image
image1 = misc.imread('files/paper/images/n01496331_11938_gen_IG_0.0.png').astype('float32')
image1 = image1/255.0

#cyclegan image
image2 = misc.imread('files/paper/images/n01496331_11938.png').astype('float32')
image2 = image2/255.0

# real image
image3 = misc.imread('files/paper/images/n01496331_11938_real.png').astype('float32')
image3 = image3/255.0

img1 = image1[50:150, :100, :]
img2 = image2[50:150, :100, :]
img3 = image3[50:150, :100, :]

misc.imsave('original.png', image1)
misc.imsave('ours.png', img1)
misc.imsave('theirs.png', img2)
misc.imsave('real.png', img3)

print 'Variance'
print 'ours:     ',np.var(img1)
print 'cyclegan: ',np.var(img2)
print 'real:     ',np.var(img3)
print
print 'Mean'
print 'ours:     ',np.mean(img1)
print 'cyclegan: ',np.mean(img2)
print 'real:     ',np.mean(img3)


