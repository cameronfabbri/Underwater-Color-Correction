'''

   Script to zoom in on parts of an image and save out

'''

import scipy.misc as misc
import numpy as np
import sys


img = misc.imread(sys.argv[1])

crop = img[200:250, 200:250, :]

misc.imsave('crop.png', crop)

